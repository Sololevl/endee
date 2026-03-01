"""
web_server.py - ChatGPT-like web interface backend for Resume-Job Matcher

Run:
  uvicorn web_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from embedder import embed_text, get_embedding_dimension
from endee_client import EndeeClient
from ingest_jobs import create_job_text
from ingest_resumes import create_resume_text
from match import find_candidates_for_job, find_jobs_for_resume
from pipeline import (
    build_job_filter,
    build_resume_filter,
    get_existing_ids,
    parse_docx,
    parse_pdf,
    save_ingested_ids,
)
from rag import (
    analyze_job_posting,
    check_ollama,
    explain_match,
    list_models,
    suggest_resume_improvements,
)
from schema import validate_job, validate_resume

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEB_DIR = BASE_DIR / "web"

RESUME_INDEX = "resumes"
JOB_INDEX = "jobs"

client = EndeeClient()

SESSION_CONTEXT: Dict[str, Optional[str]] = {
    "last_resume_id": None,
    "last_job_id": None,
}


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    intent: str


app = FastAPI(title="Resume-Job Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def _ensure_index(index_name: str):
    try:
        client.create_index(
            index_name=index_name,
            dimension=get_embedding_dimension(),
            space_type="cosine",
            m=16,
            ef_con=200,
            precision="float32",
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise


def _load_lookup(subdir: str) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    folder = DATA_DIR / subdir
    if folder.exists():
        for fp in sorted(folder.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
                doc_id = doc.get("id")
                if doc_id:
                    lookup[doc_id] = doc
    return lookup


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
    return s.strip("_") or "doc"


def _infer_doc_type(doc: Dict[str, Any]) -> Optional[Literal["resume", "job"]]:
    if {"name", "skills", "summary"}.issubset(set(doc.keys())):
        return "resume"
    if {"required_skills", "description", "title"}.issubset(set(doc.keys())):
        return "job"
    return None


def _unwrap_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap common nested payload formats like {"resume": {...}} or {"job": {...}}."""
    if "resume" in doc and isinstance(doc["resume"], dict):
        return doc["resume"]
    if "job" in doc and isinstance(doc["job"], dict):
        return doc["job"]
    if "data" in doc and isinstance(doc["data"], dict):
        return doc["data"]
    return doc


def _ensure_id(doc: Dict[str, Any], doc_type: Literal["resume", "job"]) -> Dict[str, Any]:
    """Ensure uploaded docs always have IDs so they can be matched later."""
    if doc.get("id"):
        return doc

    stamp = int(time.time())
    if doc_type == "resume":
        base = _slug(str(doc.get("name", "uploaded_resume")))
        doc["id"] = f"resume_{base}_{stamp}"
    else:
        base = _slug(str(doc.get("title", "uploaded_job")))
        doc["id"] = f"job_{base}_{stamp}"
    return doc


def _upload_help_text() -> str:
    return (
        "Upload tips: use JSON/PDF/DOCX. For JSON, include resume fields "
        "[name,title,summary,skills] or job fields [title,description,required_skills]. "
        "If unsure, choose Document type explicitly instead of Auto detect."
    )


def _guess_type_from_filename(name: str) -> Optional[Literal["resume", "job"]]:
    n = name.lower()
    if any(k in n for k in ["resume", "cv", "candidate"]):
        return "resume"
    if any(k in n for k in ["job", "jd", "posting", "role"]):
        return "job"
    return None


def _extract_skills_from_text(text: str, doc_type: Literal["resume", "job"]) -> List[str]:
    known = [
        "python", "java", "javascript", "typescript", "react", "node", "sql",
        "tensorflow", "pytorch", "nlp", "docker", "kubernetes", "aws", "azure",
        "fastapi", "flask", "spring", "golang", "c++", "data engineering",
        "machine learning", "devops", "security", "linux",
    ]
    lower = text.lower()
    found = [k for k in known if k in lower]
    if not found:
        return ["general"] if doc_type == "job" else ["communication"]
    # preserve concise list
    return found[:10]


def _normalize_unstructured_text(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    seen = set()
    deduped: List[str] = []
    for ln in lines:
        key = re.sub(r"\s+", " ", ln.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(ln)

    collapsed = "\n".join(deduped)
    collapsed = re.sub(r"\b([A-Za-z][A-Za-z0-9 .,&+-]{18,}?)\s+\1\b", r"\1", collapsed)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return collapsed


def _extract_resume_sections(text: str) -> Dict[str, str]:
    low = text.lower()

    def section_after(header_patterns: List[str], fallback_len: int = 500) -> str:
        idx = -1
        for p in header_patterns:
            m = re.search(p, low)
            if m:
                idx = m.end()
                break
        if idx == -1:
            return text[:fallback_len]
        return text[idx: idx + fallback_len]

    skills_block = section_after([r"skills?\s*[:\-]", r"technical skills?\s*[:\-]"], 400)
    experience_block = section_after([r"experience\s*[:\-]", r"work experience\s*[:\-]"], 800)
    education_block = section_after([r"education\s*[:\-]"], 400)
    summary_block = text[:700]

    return {
        "summary": summary_block,
        "skills": skills_block,
        "experience": experience_block,
        "education": education_block,
    }


def _infer_years_experience(text: str) -> int:
    matches = re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)", text.lower())
    if not matches:
        return 0
    values = [int(m) for m in matches if m.isdigit()]
    return max(values) if values else 0


def _rerank_jobs_for_resume(results: List[list], resume_doc: Dict[str, Any]) -> List[list]:
    jobs = _load_lookup("jobs")
    resume_skills = {s.lower() for s in resume_doc.get("skills", []) if isinstance(s, str)}
    if not resume_skills:
        return results

    rescored = []
    for r in results:
        sim, jid = r[0], r[1]
        jdoc = jobs.get(jid, {})
        req = {s.lower() for s in jdoc.get("required_skills", []) if isinstance(s, str)}
        overlap = len(resume_skills & req)
        norm = overlap / max(1, len(req))
        score = sim + 0.12 * norm
        rescored.append((score, r))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in rescored]


def _rerank_candidates_for_job(results: List[list], job_doc: Dict[str, Any]) -> List[list]:
    resumes = _load_lookup("resumes")
    req = {s.lower() for s in job_doc.get("required_skills", []) if isinstance(s, str)}
    if not req:
        return results

    rescored = []
    for r in results:
        sim, rid = r[0], r[1]
        rdoc = resumes.get(rid, {})
        skills = {s.lower() for s in rdoc.get("skills", []) if isinstance(s, str)}
        overlap = len(skills & req)
        norm = overlap / max(1, len(req))
        score = sim + 0.12 * norm
        rescored.append((score, r))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in rescored]


def _build_doc_from_unstructured_text(
    text: str,
    doc_type: Literal["resume", "job"],
    filename: str,
) -> Dict[str, Any]:
    stem = _slug(Path(filename).stem)
    cleaned = _normalize_unstructured_text(text)
    snippet = " ".join(cleaned.split())[:1200] if cleaned else ""
    years = _infer_years_experience(cleaned)

    if doc_type == "resume":
        sections = _extract_resume_sections(cleaned)
        return {
            "id": f"resume_{stem}_{int(time.time())}",
            "name": Path(filename).stem.replace("_", " ").title(),
            "title": "Uploaded Resume",
            "summary": (sections.get("summary", snippet) or "Uploaded resume text")[:1200],
            "skills": _extract_skills_from_text(sections.get("skills", cleaned), "resume"),
            "location": "Unknown",
            "years_experience": years,
            "is_open_to_work": True,
            "experience": [
                {
                    "title": "Uploaded Experience",
                    "company": "Unknown",
                    "years": "N/A",
                    "description": sections.get("experience", "")[:900],
                }
            ] if sections.get("experience") else [],
            "education": [
                {
                    "degree": "Not found",
                    "field": "Not found",
                    "school": "Not found",
                    "year": 0,
                }
            ] if sections.get("education") else [],
        }

    return {
        "id": f"job_{stem}_{int(time.time())}",
        "title": f"Uploaded Job - {Path(filename).stem.replace('_', ' ').title()}",
        "company": "Unknown",
        "location": "Unknown",
        "min_experience": 0,
        "remote_friendly": True,
        "description": snippet or "Uploaded job description text",
        "required_skills": _extract_skills_from_text(cleaned, "job"),
        "preferred_skills": [],
        "responsibilities": [],
        "salary_range": "N/A",
    }


def _select_model(preferred: Optional[str]) -> Optional[str]:
    if not check_ollama():
        return None
    models = [m for m in list_models() if "tinyllama" not in m.lower()]
    if not models:
        return None

    low_mem_priority = [
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
        "llama3.2:1b",
        "phi3:mini",
        "gemma:2b",
    ]

    def pick_smallest(candidates: List[str]) -> Optional[str]:
        for hint in low_mem_priority:
            for m in candidates:
                if hint in m.lower():
                    return m
        return None

    if not preferred:
        small = pick_smallest(models)
        return small or models[0]
    if preferred in models:
        return preferred
    for m in models:
        if preferred in m or m.startswith(preferred):
            return m
    small = pick_smallest(models)
    return small or models[0]


def _format_job_results(results: List[list], top_k: int = 3) -> str:
    jobs = _load_lookup("jobs")
    lines: List[str] = []
    for rank, r in enumerate(results[:top_k], 1):
        sim, jid = r[0], r[1]
        job = jobs.get(jid, {})
        lines.append(
            f"{rank}. {job.get('title', jid)} @ {job.get('company', 'Unknown')} "
            f"({sim*100:.1f}%)"
        )
    return "\n".join(lines) if lines else "No matches found."


def _format_candidate_results(results: List[list], top_k: int = 3) -> str:
    resumes = _load_lookup("resumes")
    lines: List[str] = []
    for rank, r in enumerate(results[:top_k], 1):
        sim, rid = r[0], r[1]
        cand = resumes.get(rid, {})
        lines.append(
            f"{rank}. {cand.get('name', rid)} - {cand.get('title', 'Unknown')} "
            f"({sim*100:.1f}%)"
        )
    return "\n".join(lines) if lines else "No matches found."


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _clean_generated_text(text: str) -> str:
    if not text:
        return ""
    fixed = text.replace("Not found in Not found (0)", "None identified from parsed fields")
    fixed = fixed.replace("Skill Gaps: Not found", "Skill Gaps: None identified from parsed fields")

    lines = [ln.rstrip() for ln in fixed.splitlines()]
    cleaned: List[str] = []
    seen = set()
    for ln in lines:
        key = re.sub(r"\s+", " ", ln.strip().lower())
        if not key:
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(ln.strip())

    return "\n".join(cleaned).strip()


def _estimate_resume_years(resume_doc: Dict[str, Any]) -> int:
    years = _infer_years_experience(create_resume_text(resume_doc))
    if years:
        return years
    exp = resume_doc.get("experience", [])
    collected: List[int] = []
    if isinstance(exp, list):
        for item in exp:
            if isinstance(item, dict):
                y = str(item.get("years", ""))
                for m in re.findall(r"(\d{1,2})", y):
                    if m.isdigit():
                        collected.append(int(m))
    return max(collected) if collected else 0


def _match_strength_label(sim: float) -> str:
    if sim >= 0.75:
        return "Strong"
    if sim >= 0.55:
        return "Moderate"
    return "Weak"


def _build_explain_template(resume_doc: Dict[str, Any], job_doc: Dict[str, Any], sim: float) -> str:
    resume_skills = {
        s.strip().lower()
        for s in resume_doc.get("skills", [])
        if isinstance(s, str) and s.strip()
    }
    required_skills = [
        s.strip()
        for s in job_doc.get("required_skills", [])
        if isinstance(s, str) and s.strip()
    ]

    key_matching = [s for s in required_skills if s.lower() in resume_skills]
    gaps = [s for s in required_skills if s.lower() not in resume_skills]

    resume_years = _estimate_resume_years(resume_doc)
    min_exp = int(job_doc.get("min_experience", 0) or 0)
    if min_exp == 0:
        exp_fit = f"Resume indicates ~{resume_years} years experience. Job has no strict minimum."
    elif resume_years >= min_exp:
        exp_fit = f"Meets experience requirement ({resume_years}y vs required {min_exp}y)."
    else:
        exp_fit = f"Below preferred experience ({resume_years}y vs required {min_exp}y)."

    recommendation = (
        "Strong fit for interview shortlist."
        if not gaps and resume_years >= min_exp
        else "Potential fit if resume is updated to close the top skill gaps."
    )

    return (
        f"Match Strength: {_match_strength_label(sim)}\n"
        f"Match Score: {sim*100:.1f}%\n"
        f"Key Matching Skills: {', '.join(key_matching[:8]) if key_matching else 'None identified'}\n"
        f"Skill Gaps: {', '.join(gaps[:8]) if gaps else 'None identified'}\n"
        f"Experience Fit: {exp_fit}\n"
        f"Recommendation: {recommendation}"
    )


def _build_resume_improvement_template(resume_doc: Dict[str, Any], results: List[list]) -> str:
    jobs = _load_lookup("jobs")
    top_jobs: List[Dict[str, Any]] = []
    for row in results[:3]:
        jid = row[1]
        if jid in jobs:
            top_jobs.append(jobs[jid])

    if not top_jobs:
        return "No matching jobs found to generate resume improvements."

    top_job = top_jobs[0]
    top_title = top_job.get("title", "target role")
    required = [s for s in top_job.get("required_skills", []) if isinstance(s, str)]

    current_skills = [s for s in resume_doc.get("skills", []) if isinstance(s, str)]
    current_skill_set = {s.lower() for s in current_skills}
    missing = [s for s in required if s.lower() not in current_skill_set]
    suggested_skills = _unique_keep_order(current_skills + missing[:6])

    years = _estimate_resume_years(resume_doc)
    headline_title = resume_doc.get("title") or top_title

    before_summary = str(resume_doc.get("summary", "")).strip() or "(No summary provided)"
    after_summary = (
        f"{headline_title} with {max(years, 1)}+ years building production systems. "
        f"Skilled in {', '.join(suggested_skills[:6])}. "
        f"Delivered measurable outcomes in reliability, performance, and business impact."
    )

    before_skills = ", ".join(current_skills[:12]) if current_skills else "(No skills listed)"
    after_skills = ", ".join(suggested_skills[:14]) if suggested_skills else "(Add core role skills)"

    exp = resume_doc.get("experience", [])
    first_exp_desc = ""
    if isinstance(exp, list) and exp:
        first = exp[0]
        if isinstance(first, dict):
            first_exp_desc = str(first.get("description", "")).strip()
    before_bullet = first_exp_desc or "Worked on backend and APIs."
    after_bullet = (
        f"Built and optimized API services for {top_title}, reducing response time by 35% "
        f"and improving reliability to 99.9% uptime."
    )

    role_targets = []
    for i, j in enumerate(top_jobs[:3], 1):
        role_targets.append(f"{i}. {j.get('title', 'Unknown')} @ {j.get('company', 'Unknown')}")

    return (
        "ATS Rewrite Plan\n"
        f"Target Roles:\n" + "\n".join(role_targets) + "\n\n"
        "1) Professional Summary\n"
        f"Before: {before_summary}\n"
        f"After: {after_summary}\n"
        "Why: Aligns headline and impact language to top-matching role requirements.\n\n"
        "2) Skills Section\n"
        f"Before: {before_skills}\n"
        f"After: {after_skills}\n"
        "Why: Increases ATS keyword coverage for required skills.\n\n"
        "3) Experience Bullet Rewrite\n"
        f"Before: {before_bullet}\n"
        f"After: {after_bullet}\n"
        "Why: Uses quantified outcomes recruiters scan for quickly.\n\n"
        "4) Missing Skill Priorities\n"
        f"Add first: {', '.join(missing[:6]) if missing else 'No critical missing skills from top role.'}"
    )


def _build_ats_summary_rewrite(resume_doc: Dict[str, Any], results: List[list]) -> str:
    jobs = _load_lookup("jobs")
    top_job = None
    if results:
        top_job = jobs.get(results[0][1])

    current_summary = str(resume_doc.get("summary", "")).strip()
    if not current_summary:
        current_summary = create_resume_text(resume_doc)[:320].strip() or "(No summary found)"

    title = str(resume_doc.get("title", "")).strip() or (top_job.get("title") if top_job else "Software Engineer")
    years = max(1, _estimate_resume_years(resume_doc))

    resume_skills = [s for s in resume_doc.get("skills", []) if isinstance(s, str) and s.strip()]
    top_required = [s for s in (top_job.get("required_skills", []) if top_job else []) if isinstance(s, str) and s.strip()]
    merged = _unique_keep_order(resume_skills + top_required)
    skill_line = ", ".join(merged[:6]) if merged else "Python, APIs, problem solving"

    target_role = top_job.get("title", title) if top_job else title
    rewritten = (
        f"{title} with {years}+ years building production-ready solutions in {skill_line}. "
        f"Delivered measurable impact through scalable systems, reliable execution, and cross-functional collaboration. "
        f"Targeting {target_role} roles with strong alignment to required technical skills and ATS keywords."
    )

    return (
        "ATS Summary Rewrite\n"
        f"Before: {current_summary[:550]}\n"
        f"After: {rewritten}\n"
        "Why: Short, keyword-rich summary aligned to top matching role and ATS parsing."
    )


def _extract_ids(message: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract resume/job IDs from free text.

    Supports both numeric IDs (resume_001, job_003) and uploaded IDs like
    resume_mmj_resume_v1_1772282940.
    """
    lower = message.lower()

    known_resume_ids = list(_load_lookup("resumes").keys())
    known_job_ids = list(_load_lookup("jobs").keys())

    def normalize_candidate(cand: str, kind: str) -> str:
        c = cand.replace("-", "_").strip("_ ")
        if kind == "resume" and "_job_" in c:
            c = c.split("_job_", 1)[0]
        if kind == "job" and "_resume_" in c:
            c = c.split("_resume_", 1)[0]
        return c

    resume_candidates = [normalize_candidate(c, "resume") for c in re.findall(r"resume(?:[_-][a-z0-9]+)+", lower)]
    job_candidates = [normalize_candidate(c, "job") for c in re.findall(r"job(?:[_-][a-z0-9]+)+", lower)]

    resume_id: Optional[str] = None
    job_id: Optional[str] = None

    lower_resume_map = {rid.lower(): rid for rid in known_resume_ids}
    lower_job_map = {jid.lower(): jid for jid in known_job_ids}

    for cand in resume_candidates:
        if cand in lower_resume_map:
            resume_id = lower_resume_map[cand]
            break
    if resume_id is None:
        for cand in resume_candidates:
            for known_lower, known_real in sorted(lower_resume_map.items(), key=lambda x: len(x[0]), reverse=True):
                if cand.startswith(known_lower):
                    resume_id = known_real
                    break
            if resume_id:
                break

    for cand in job_candidates:
        if cand in lower_job_map:
            job_id = lower_job_map[cand]
            break
    if job_id is None:
        for cand in job_candidates:
            for known_lower, known_real in sorted(lower_job_map.items(), key=lambda x: len(x[0]), reverse=True):
                if cand.startswith(known_lower):
                    job_id = known_real
                    break
            if job_id:
                break

    if resume_id is None and resume_candidates:
        resume_id = resume_candidates[0]
    if job_id is None and job_candidates:
        job_id = job_candidates[0]

    return resume_id, job_id


def _set_last_ids(resume_id: Optional[str] = None, job_id: Optional[str] = None) -> None:
    if resume_id:
        SESSION_CONTEXT["last_resume_id"] = resume_id
    if job_id:
        SESSION_CONTEXT["last_job_id"] = job_id


def _resolve_contextual_ids(
    lower_message: str,
    resume_id: Optional[str],
    job_id: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    resume_refs = ["this resume", "that resume", "current resume", "uploaded resume"]
    job_refs = ["this job", "that job", "this role", "that role", "current job"]

    if resume_id is None and any(ref in lower_message for ref in resume_refs):
        resume_id = SESSION_CONTEXT.get("last_resume_id")
    if job_id is None and any(ref in lower_message for ref in job_refs):
        job_id = SESSION_CONTEXT.get("last_job_id")

    if ("explain" in lower_message or "why" in lower_message) and "this match" in lower_message:
        resume_id = resume_id or SESSION_CONTEXT.get("last_resume_id")
        job_id = job_id or SESSION_CONTEXT.get("last_job_id")

    return resume_id, job_id


def _is_resume_follow_up_query(lower_message: str) -> bool:
    follow_markers = [
        "summary",
        "skills",
        "bullet",
        "ats",
        "rewrite",
        "improve",
        "tailor",
        "optimize",
        "experience section",
    ]
    blocked_markers = ["find jobs", "find candidates", "explain match", "analyze job"]
    if any(marker in lower_message for marker in blocked_markers):
        return False
    return any(marker in lower_message for marker in follow_markers)


def _extract_word_limit(lower_message: str) -> Optional[int]:
    m = re.search(r"\b(\d{2,4})\s*words?\b", lower_message)
    if not m:
        return None
    n = int(m.group(1))
    if n < 30:
        return 30
    if n > 600:
        return 600
    return n


def _truncate_to_words(text: str, word_limit: Optional[int]) -> str:
    if not word_limit:
        return text.strip()
    words = text.split()
    if len(words) <= word_limit:
        return text.strip()
    return " ".join(words[:word_limit]).strip() + "..."


def _build_experience_section_rewrite(
    resume_doc: Dict[str, Any],
    results: List[list],
    word_limit: Optional[int] = None,
) -> str:
    jobs = _load_lookup("jobs")
    target_job = jobs.get(results[0][1], {}) if results else {}
    target_role = target_job.get("title", "target role")

    raw_experience = ""
    exp = resume_doc.get("experience", [])
    if isinstance(exp, list) and exp:
        parts: List[str] = []
        for item in exp[:4]:
            if isinstance(item, dict):
                role = str(item.get("title", "")).strip()
                org = str(item.get("company", "")).strip()
                desc = str(item.get("description", "")).strip()
                line = " - ".join([x for x in [role, org, desc] if x])
                if line:
                    parts.append(line)
        raw_experience = " ".join(parts)

    if not raw_experience:
        raw_experience = create_resume_text(resume_doc)

    skills = [s for s in resume_doc.get("skills", []) if isinstance(s, str)]
    skills_line = ", ".join(skills[:6]) if skills else "Python, APIs, cloud"
    rewritten = (
        f"Built and deployed production-oriented projects with focus on {skills_line}. "
        f"Delivered practical outcomes through backend development, model integration, and reliable deployment workflows. "
        f"Contributed to end-to-end solutions including API design, experimentation, and performance tuning. "
        f"Experience aligns with {target_role} expectations by combining implementation speed, hands-on debugging, and measurable delivery."
    )
    rewritten = _truncate_to_words(rewritten, word_limit)
    limit_note = f" ({word_limit} words max)" if word_limit else ""

    return (
        f"Experience Section Rewrite{limit_note}\n"
        f"Before: {raw_experience[:700]}\n"
        f"After: {rewritten}\n"
        "Why: Keeps experience concise, impact-focused, and aligned to target-role keywords."
    )


def _is_resume_general_question(lower_message: str) -> bool:
    question_markers = [
        "what", "which", "how", "can i", "should i", "do i", "am i", "for my resume", "based on my resume",
        "for this resume", "my profile", "my resume",
    ]
    blocked = ["find jobs", "find candidates", "explain match", "analyze job"]
    if any(b in lower_message for b in blocked):
        return False
    return any(q in lower_message for q in question_markers)


def _build_resume_general_answer(resume_doc: Dict[str, Any], results: List[list]) -> str:
    jobs = _load_lookup("jobs")
    top = []
    for row in results[:3]:
        jid = row[1]
        jdoc = jobs.get(jid, {})
        top.append((jdoc.get("title", jid), row[0], jdoc))

    top_lines = [f"{i+1}. {title} ({score*100:.1f}%)" for i, (title, score, _) in enumerate(top)]
    req = []
    for _, _, jdoc in top[:2]:
        req.extend([s for s in jdoc.get("required_skills", []) if isinstance(s, str)])
    req = _unique_keep_order(req)
    current = {s.lower() for s in resume_doc.get("skills", []) if isinstance(s, str)}
    gaps = [s for s in req if s.lower() not in current][:6]

    return (
        "Resume-based answer:\n"
        + ("Top matching roles:\n" + "\n".join(top_lines) + "\n" if top_lines else "No strong role matches yet.\n")
        + f"Main skill gaps: {', '.join(gaps) if gaps else 'No major gaps from top roles.'}\n"
        + "Ask next: 'rewrite summary for ATS', 'in 200 words rewrite experience section', or 'give better skills section for this resume'."
    )


def _is_job_count_query(lower_message: str) -> bool:
    if "how many" not in lower_message and "count" not in lower_message and "number of" not in lower_message:
        return False
    job_terms = ["job", "jobs", "position", "positions", "opening", "openings", "roles"]
    return any(term in lower_message for term in job_terms)


def _ingest_single_document(doc: Dict[str, Any], doc_type: Literal["resume", "job"]) -> str:
    doc_id = doc.get("id")
    if not doc_id:
        raise HTTPException(status_code=400, detail="Document must contain an 'id' field")

    if doc_type == "resume":
        is_valid, errors = validate_resume(doc)
        if not is_valid:
            raise HTTPException(status_code=400, detail="; ".join(errors[:3]))

        _ensure_index(RESUME_INDEX)
        existing = get_existing_ids(client, RESUME_INDEX)
        if doc_id in existing:
            return f"Skipped resume '{doc_id}' (already ingested)."

        text = create_resume_text(doc)
        vector = embed_text(text)
        client.insert_vectors(
            RESUME_INDEX,
            [
                {
                    "id": doc_id,
                    "vector": vector,
                    "filter": build_resume_filter(doc),
                    "meta": json.dumps({
                        "name": doc.get("name", ""),
                        "title": doc.get("title", ""),
                    }),
                }
            ],
        )
        existing.add(doc_id)
        save_ingested_ids(RESUME_INDEX, existing)

        out_file = DATA_DIR / "resumes" / f"{_slug(doc_id)}_{int(time.time())}.json"
        out_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return f"Ingested resume '{doc_id}' successfully."

    is_valid, errors = validate_job(doc)
    if not is_valid:
        raise HTTPException(status_code=400, detail="; ".join(errors[:3]))

    _ensure_index(JOB_INDEX)
    existing = get_existing_ids(client, JOB_INDEX)
    if doc_id in existing:
        return f"Skipped job '{doc_id}' (already ingested)."

    text = create_job_text(doc)
    vector = embed_text(text)
    client.insert_vectors(
        JOB_INDEX,
        [
            {
                "id": doc_id,
                "vector": vector,
                "filter": build_job_filter(doc),
                "meta": json.dumps({
                    "title": doc.get("title", ""),
                    "company": doc.get("company", ""),
                }),
            }
        ],
    )
    existing.add(doc_id)
    save_ingested_ids(JOB_INDEX, existing)

    out_file = DATA_DIR / "jobs" / f"{_slug(doc_id)}_{int(time.time())}.json"
    out_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return f"Ingested job '{doc_id}' successfully."


def _agent_reply(message: str, model: Optional[str]) -> Tuple[str, str]:
    msg = message.strip()
    lower = msg.lower()
    resume_id, job_id = _extract_ids(msg)
    resume_id, job_id = _resolve_contextual_ids(lower, resume_id, job_id)

    if ("explain" in lower or "why" in lower) and re.search(r"resume(?:[_-][a-z0-9]+)+job(?:[_-][a-z0-9]+)+", lower):
        return "Use format: explain match resume_<id> job_<id> (add a space between ids).", "explain"

    if _is_job_count_query(lower):
        jobs = _load_lookup("jobs")
        total = len(jobs)
        sample_titles = [
            f"- {doc.get('title', jid)} @ {doc.get('company', 'Unknown')}"
            for jid, doc in list(jobs.items())[:5]
        ]
        reply = f"There are {total} job positions currently available in the local index."
        if sample_titles:
            reply += "\n\nSample openings:\n" + "\n".join(sample_titles)
        return reply, "job_count"

    if "find jobs" in lower and resume_id:
        resumes = _load_lookup("resumes")
        rdoc = resumes.get(resume_id)
        if not rdoc:
            return f"Resume '{resume_id}' not found in data/resumes.", "find_jobs"
        results = find_jobs_for_resume(create_resume_text(rdoc), k=20)
        results = _rerank_jobs_for_resume(results, rdoc)
        _set_last_ids(resume_id=resume_id)
        return f"Top jobs for {rdoc.get('name', resume_id)}:\n\n{_format_job_results(results)}", "find_jobs"

    if "find candidates" in lower and job_id:
        jobs = _load_lookup("jobs")
        jdoc = jobs.get(job_id)
        if not jdoc:
            return f"Job '{job_id}' not found in data/jobs.", "find_candidates"
        results = find_candidates_for_job(create_job_text(jdoc), k=20)
        results = _rerank_candidates_for_job(results, jdoc)
        _set_last_ids(job_id=job_id)
        return f"Top candidates for {jdoc.get('title', job_id)}:\n\n{_format_candidate_results(results)}", "find_candidates"

    if ("explain" in lower or "why" in lower) and resume_id and job_id:
        resumes = _load_lookup("resumes")
        jobs = _load_lookup("jobs")
        if resume_id not in resumes or job_id not in jobs:
            return "Provide valid ids, e.g. 'Explain match resume_001 job_001'.", "explain"

        _set_last_ids(resume_id=resume_id, job_id=job_id)

        sim = 0.0
        for r in find_jobs_for_resume(create_resume_text(resumes[resume_id]), k=20):
            if r[1] == job_id:
                sim = r[0]
                break

        base = _build_explain_template(resumes[resume_id], jobs[job_id], sim)
        selected_model = _select_model(model)
        if not selected_model:
            return base + "\n\nAI Note: Ollama unavailable, showing deterministic explanation.", "explain"

        response = explain_match(resume_id, job_id, sim, model=selected_model)
        if response.startswith("[RAG unavailable"):
            return base + "\n\nAI Note: Ollama unavailable, showing deterministic explanation.", "explain"
        cleaned = _clean_generated_text(response)
        ai_line = ""
        for line in cleaned.splitlines():
            if line and not line.lower().startswith(("match strength", "key matching", "skill gaps", "experience fit", "recommendation")):
                ai_line = line
                break
        if ai_line:
            return base + "\nAI Note: " + ai_line, "explain"
        return base, "explain"

    if ("explain" in lower or "why" in lower) and ("resume" in lower or "job" in lower):
        return "Use format: explain match resume_<id> job_<id> (with a space between ids).", "explain"

    if ("explain" in lower or "why" in lower) and "match" in lower:
        return "No recent match context found. First run: explain match resume_<id> job_<id>.", "explain"

    if "improve" in lower and "resume" in lower and resume_id:
        resumes = _load_lookup("resumes")
        rdoc = resumes.get(resume_id)
        if not rdoc:
            return f"Resume '{resume_id}' not found.", "improve_resume"

        results = find_jobs_for_resume(create_resume_text(rdoc), k=20)
        results = _rerank_jobs_for_resume(results, rdoc)
        _set_last_ids(resume_id=resume_id)
        template = _build_resume_improvement_template(rdoc, results)
        return _clean_generated_text(template), "improve_resume"

    if (("rewrite" in lower and "summary" in lower) or ("ats" in lower and "summary" in lower)):
        target_resume_id = resume_id or SESSION_CONTEXT.get("last_resume_id")
        if not target_resume_id:
            return "No recent resume context found. Use: improve resume <resume_id> once, then 'rewrite summary for ATS'.", "improve_resume"
        resumes = _load_lookup("resumes")
        rdoc = resumes.get(target_resume_id)
        if not rdoc:
            return f"Resume '{target_resume_id}' not found.", "improve_resume"

        results = find_jobs_for_resume(create_resume_text(rdoc), k=20)
        results = _rerank_jobs_for_resume(results, rdoc)
        _set_last_ids(resume_id=target_resume_id)
        template = _build_ats_summary_rewrite(rdoc, results)
        return _clean_generated_text(template), "improve_resume"

    if ("rewrite" in lower or "write" in lower) and "experience" in lower:
        target_resume_id = resume_id or SESSION_CONTEXT.get("last_resume_id")
        if not target_resume_id:
            return "No recent resume context found. Use: improve resume <resume_id> once, then ask experience rewrite.", "improve_resume"
        resumes = _load_lookup("resumes")
        rdoc = resumes.get(target_resume_id)
        if not rdoc:
            return f"Resume '{target_resume_id}' not found.", "improve_resume"
        results = find_jobs_for_resume(create_resume_text(rdoc), k=20)
        results = _rerank_jobs_for_resume(results, rdoc)
        _set_last_ids(resume_id=target_resume_id)
        word_limit = _extract_word_limit(lower)
        template = _build_experience_section_rewrite(rdoc, results, word_limit)
        return _clean_generated_text(template), "improve_resume"

    if "improve" in lower and "resume" in lower:
        return "No recent resume context found. Use: improve resume <resume_id> once, then 'improve this resume'.", "improve_resume"

    if _is_resume_follow_up_query(lower):
        target_resume_id = resume_id or SESSION_CONTEXT.get("last_resume_id")
        if target_resume_id:
            resumes = _load_lookup("resumes")
            rdoc = resumes.get(target_resume_id)
            if rdoc:
                results = find_jobs_for_resume(create_resume_text(rdoc), k=20)
                results = _rerank_jobs_for_resume(results, rdoc)
                _set_last_ids(resume_id=target_resume_id)

                if "summary" in lower:
                    template = _build_ats_summary_rewrite(rdoc, results)
                    return _clean_generated_text(template), "improve_resume"

                template = _build_resume_improvement_template(rdoc, results)
                return _clean_generated_text(template), "improve_resume"

    if _is_resume_general_question(lower):
        target_resume_id = resume_id or SESSION_CONTEXT.get("last_resume_id")
        if target_resume_id:
            resumes = _load_lookup("resumes")
            rdoc = resumes.get(target_resume_id)
            if rdoc:
                results = find_jobs_for_resume(create_resume_text(rdoc), k=20)
                results = _rerank_jobs_for_resume(results, rdoc)
                _set_last_ids(resume_id=target_resume_id)
                return _build_resume_general_answer(rdoc, results), "resume_qa"

    if ("analyze job" in lower or ("analyze" in lower and "job" in lower)) and job_id:
        jobs = _load_lookup("jobs")
        jdoc = jobs.get(job_id)
        if not jdoc:
            return f"Job '{job_id}' not found.", "analyze_job"

        _set_last_ids(job_id=job_id)

        selected_model = _select_model(model)
        if not selected_model:
            return (
                "RAG is unavailable (Ollama/model not ready). Run: 'ollama serve' and pull a model.",
                "analyze_job",
            )

        results = find_candidates_for_job(create_job_text(jdoc), k=20)
        results = _rerank_candidates_for_job(results, jdoc)
        response = analyze_job_posting(job_id, results, model=selected_model)
        if response.startswith("[RAG unavailable"):
            return (
                response
                + "\n\nFallback (vector-only): Top candidates:\n"
                + _format_candidate_results(results, top_k=3),
                "analyze_job",
            )
        return _clean_generated_text(response), "analyze_job"

    if "analyze" in lower and resume_id:
        resumes = _load_lookup("resumes")
        rdoc = resumes.get(resume_id)
        if not rdoc:
            return f"Resume '{resume_id}' not found.", "analyze_resume"

        results = find_jobs_for_resume(create_resume_text(rdoc), k=20)
        results = _rerank_jobs_for_resume(results, rdoc)
        _set_last_ids(resume_id=resume_id)
        return (
            "Resume-centric analysis:\n\n"
            f"Top job fits:\n{_format_job_results(results, top_k=3)}\n\n"
            "Suggested requirement focus for target jobs:\n"
            "- Must-have: Python, SQL, core domain skills from top matches\n"
            "- Nice-to-have: cloud/devops stack, role-specific tooling\n"
            "- Resume fix: add quantified project outcomes and stronger role title alignment",
            "analyze_resume",
        )

    if "analyze" in lower and "job" in lower:
        return "No recent job context found. Use: analyze job <job_id> once, then 'analyze this job'.", "analyze_job"

    # Default semantic query path
    try:
        jobs = find_jobs_for_resume(msg, k=3)
        return (
            "I interpreted your message as a semantic job search.\n\n"
            f"Top job matches:\n{_format_job_results(jobs, top_k=3)}\n\n"
            "Tip: You can also say 'find candidates for job_003' or 'explain match resume_001 job_001'.",
            "semantic_search",
        )
    except Exception as e:
        return f"Search failed: {e}", "semantic_search"


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def api_status():
    ende_ok = True
    indexes: List[str] = []
    try:
        res = client.list_indexes()
        if isinstance(res, dict):
            if "indexes" in res and isinstance(res["indexes"], list):
                for item in res["indexes"]:
                    if isinstance(item, dict) and "name" in item:
                        indexes.append(str(item["name"]))
                    else:
                        indexes.append(str(item))
            else:
                indexes = list(res.keys())
        elif isinstance(res, list):
            indexes = [str(x) for x in res]
    except Exception:
        ende_ok = False
    return {
        "endee": ende_ok,
        "ollama": check_ollama(),
        "indexes": indexes,
    }


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    doc_type: Literal["auto", "resume", "job"] = Query(default="auto"),
):
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    raw = await file.read()

    docs: List[Dict[str, Any]] = []

    if ext == ".json":
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        parsed = payload if isinstance(payload, list) else [payload]
        if not parsed:
            raise HTTPException(status_code=400, detail="JSON file is empty")
        docs = parsed

    elif ext in (".pdf", ".docx"):
        temp_dir = DATA_DIR / ".uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{_slug(Path(filename).stem)}_{int(time.time())}{ext}"
        temp_path.write_bytes(raw)

        try:
            if ext == ".pdf":
                text = parse_pdf(temp_path)
            else:
                text = parse_docx(temp_path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

        if not text:
            raise HTTPException(status_code=400, detail=f"Could not extract text from {ext} file")

        inferred_from_name = _guess_type_from_filename(filename)
        if doc_type == "auto":
            effective_type = inferred_from_name or "resume"
        else:
            effective_type = doc_type

        docs = [_build_doc_from_unstructured_text(text, effective_type, filename)]

    else:
        raise HTTPException(status_code=400, detail="Supported formats: .json, .pdf, .docx")

    messages: List[str] = []
    uploaded_resume_ids: List[str] = []
    uploaded_job_ids: List[str] = []
    for doc in docs:
        if not isinstance(doc, dict):
            messages.append("Skipped non-object JSON entry")
            continue

        doc = _unwrap_document(doc)

        inferred = _infer_doc_type(doc)
        effective: Optional[Literal["resume", "job"]]
        if doc_type == "auto":
            effective = inferred
        else:
            effective = doc_type

        # For generated docs from PDF/DOCX, explicit fallback
        if effective is None and ext in (".pdf", ".docx"):
            effective = _guess_type_from_filename(filename) or "resume"

        if effective not in ("resume", "job"):
            messages.append("Skipped document: could not infer type (resume/job). " + _upload_help_text())
            continue

        doc = _ensure_id(doc, effective)

        try:
            msg = _ingest_single_document(doc, effective)
            messages.append(msg)
            if effective == "resume":
                uploaded_resume_ids.append(doc["id"])
            else:
                uploaded_job_ids.append(doc["id"])
        except HTTPException as e:
            messages.append(f"Error: {e.detail}. {_upload_help_text()}")
        except Exception as e:
            messages.append(f"Error: {e}")

    if uploaded_resume_ids:
        _set_last_ids(resume_id=uploaded_resume_ids[-1])
        messages.append(
            "Use these for chat commands: "
            + ", ".join(f"find jobs for {rid}" for rid in uploaded_resume_ids[:3])
        )
        messages.append(
            "For improvement: "
            + ", ".join(f"improve resume {rid}" for rid in uploaded_resume_ids[:3])
        )

    if uploaded_job_ids:
        _set_last_ids(job_id=uploaded_job_ids[-1])
        messages.append(
            "Use these for chat commands: "
            + ", ".join(f"find candidates for {jid}" for jid in uploaded_job_ids[:3])
        )

    return {
        "message": "\n".join(messages),
        "uploaded_resume_ids": uploaded_resume_ids,
        "uploaded_job_ids": uploaded_job_ids,
    }


@app.get("/api/models")
def api_models():
    if not check_ollama():
        return {"ollama": False, "models": [], "recommended": None}

    models = [m for m in list_models() if "tinyllama" not in m.lower()]
    recommended = _select_model(None) if models else None
    return {
        "ollama": True,
        "models": models,
        "recommended": recommended,
    }


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    reply, intent = _agent_reply(req.message, req.model)
    return ChatResponse(reply=reply, intent=intent)
