"""
rag.py - Retrieval-Augmented Generation for Match Explanations

Uses Endee vector search to retrieve relevant matches, then sends
the results to a local Ollama LLM to generate human-readable
explanations of why a resume and job are a good (or poor) match.

Requirements:
  - Ollama running locally (https://ollama.com)
  - A model pulled, e.g.: ollama pull llama3.2
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

DATA_DIR = Path(__file__).parent.parent / "data"

# ─── Ollama client ──────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def check_ollama() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> List[str]:
    """List available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def generate(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    """
    Generate text using Ollama.

    Args:
        prompt: The full prompt to send
        model: Ollama model name (default: llama3.2)
        stream: Whether to stream (not used here, we collect full response)

    Returns:
        Generated text string
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 320,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json().get("response", "")
        else:
            error_text = r.text or ""
            error_lower = error_text.lower()
            if r.status_code == 500 and "requires more system memory" in error_lower:
                return (
                    "[RAG unavailable: insufficient RAM for selected model.]\n"
                    f"Model: {model}\n"
                    "Try a smaller model, e.g.:\n"
                    "- ollama pull tinyllama\n"
                    "- ollama pull qwen2.5:0.5b\n"
                    "- ollama pull llama3.2:1b\n"
                    "Then choose that model in the RAG menu."
                )
            return f"[Ollama error {r.status_code}]: {error_text[:200]}"
    except requests.exceptions.ConnectionError:
        return "[Error] Cannot connect to Ollama. Make sure it's running: ollama serve"
    except requests.exceptions.Timeout:
        return "[Error] Ollama request timed out. Try a smaller model."
    except Exception as e:
        return f"[Error] {e}"


# ─── Data helpers ───────────────────────────────────────────────────

def _load_lookup(subdir: str) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    folder = DATA_DIR / subdir
    if folder.exists():
        for fp in sorted(folder.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
                doc_id = doc.get("id", "")
                if doc_id:
                    lookup[doc_id] = doc
    return lookup


def _resume_summary(doc: dict) -> str:
    """Build a concise text summary of a resume for the LLM prompt."""
    parts = [
        f"Name: {doc.get('name', 'N/A')}",
        f"Title: {doc.get('title', 'N/A')}",
        f"Location: {doc.get('location', 'N/A')}",
        f"Years of experience: {doc.get('years_experience', 'N/A')}",
        f"Skills: {', '.join(doc.get('skills', []))}",
        f"Summary: {doc.get('summary', 'N/A')}",
    ]
    for exp in doc.get("experience", []):
        parts.append(f"  - {exp.get('title', '')} at {exp.get('company', '')} ({exp.get('years', '')}): {exp.get('description', '')}")
    for edu in doc.get("education", []):
        parts.append(f"  - {edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('school', '')} ({edu.get('year', '')})")
    return "\n".join(parts)


def _job_summary(doc: dict) -> str:
    """Build a concise text summary of a job for the LLM prompt."""
    parts = [
        f"Title: {doc.get('title', 'N/A')}",
        f"Company: {doc.get('company', 'N/A')}",
        f"Location: {doc.get('location', 'N/A')}",
        f"Min experience: {doc.get('min_experience', 'N/A')} years",
        f"Remote: {'Yes' if doc.get('remote_friendly') else 'No'}",
        f"Salary: {doc.get('salary_range', 'N/A')}",
        f"Description: {doc.get('description', 'N/A')}",
        f"Required skills: {', '.join(doc.get('required_skills', []))}",
        f"Preferred skills: {', '.join(doc.get('preferred_skills', []))}",
    ]
    for resp in doc.get("responsibilities", []):
        parts.append(f"  - {resp}")
    return "\n".join(parts)


# ─── RAG prompts ────────────────────────────────────────────────────

def build_match_explanation_prompt(
    resume_doc: dict,
    job_doc: dict,
    similarity: float,
) -> str:
    """
    Build a prompt that asks the LLM to explain why a resume
    and job are (or aren't) a good match.
    """
    return f"""You are an expert AI recruiter.
Use ONLY the provided resume and job content.
If a detail is missing, write: Not found.
Do not repeat the full resume/job text.

RESUME:
{_resume_summary(resume_doc)}

JOB POSTING:
{_job_summary(job_doc)}

SIMILARITY SCORE: {similarity*100:.1f}%

Return exactly this structure:
- Match Strength: Strong/Moderate/Weak + 1 reason
- Key Matching Skills: 3-6 bullets
- Skill Gaps: 3-6 bullets
- Experience Fit: 2 bullets
- Recommendation: 1 short paragraph

Keep it under 220 words. Be specific and concise."""


def build_resume_improvement_prompt(resume_doc: dict, top_jobs: list) -> str:
    """Build a prompt for resume improvement suggestions based on target jobs."""
    jobs_lookup = _load_lookup("jobs")
    job_summaries = []
    for result in top_jobs[:3]:
        jid = result[1]
        jdoc = jobs_lookup.get(jid, {})
        if jdoc:
            job_summaries.append(_job_summary(jdoc))

    return f"""You are a career coach.
Use ONLY the provided resume and top jobs.
If evidence is missing, write: Not found.

RESUME:
{_resume_summary(resume_doc)}

TOP MATCHING JOBS:
{chr(10).join(f'--- Job {i+1} ---{chr(10)}{s}' for i, s in enumerate(job_summaries))}

Return exactly 5 numbered suggestions.
Each suggestion must include:
1) What to change
2) Why it matters for top jobs
3) Example rewrite (one line)

Keep it under 230 words."""


def build_job_posting_analysis_prompt(job_doc: dict, top_candidates: list) -> str:
    """Build a prompt analyzing a job posting based on available candidates."""
    resumes_lookup = _load_lookup("resumes")
    candidate_summaries = []
    for result in top_candidates[:3]:
        rid = result[1]
        rdoc = resumes_lookup.get(rid, {})
        if rdoc:
            candidate_summaries.append(_resume_summary(rdoc))

    return f"""You are a hiring consultant.
Use ONLY the provided job posting and candidate summaries.
If evidence is missing, write: Not found.

JOB POSTING:
{_job_summary(job_doc)}

TOP AVAILABLE CANDIDATES:
{chr(10).join(f'--- Candidate {i+1} ---{chr(10)}{s}' for i, s in enumerate(candidate_summaries))}

Return exactly this structure:
- Candidate Pool Fit: 2-3 bullets
- Best Candidate: name/id + 2 reasons
- Requirement Fixes: 3 bullets (must-have vs nice-to-have)

Keep it under 220 words."""


# ─── High-level RAG functions ──────────────────────────────────────

def explain_match(
    resume_id: str,
    job_id: str,
    similarity: float,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate an AI explanation for a resume-job match.

    Args:
        resume_id: ID of the resume
        job_id: ID of the job
        similarity: Similarity score (0-1)
        model: Ollama model to use

    Returns:
        Generated explanation text
    """
    resumes = _load_lookup("resumes")
    jobs = _load_lookup("jobs")

    resume_doc = resumes.get(resume_id)
    job_doc = jobs.get(job_id)

    if not resume_doc:
        return f"Resume '{resume_id}' not found in data."
    if not job_doc:
        return f"Job '{job_id}' not found in data."

    prompt = build_match_explanation_prompt(resume_doc, job_doc, similarity)
    return generate(prompt, model=model)


def suggest_resume_improvements(
    resume_id: str,
    search_results: list,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate AI suggestions for resume improvement."""
    resumes = _load_lookup("resumes")
    resume_doc = resumes.get(resume_id)
    if not resume_doc:
        return f"Resume '{resume_id}' not found."

    prompt = build_resume_improvement_prompt(resume_doc, search_results)
    return generate(prompt, model=model)


def analyze_job_posting(
    job_id: str,
    search_results: list,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate AI analysis of a job posting vs available candidates."""
    jobs = _load_lookup("jobs")
    job_doc = jobs.get(job_id)
    if not job_doc:
        return f"Job '{job_id}' not found."

    prompt = build_job_posting_analysis_prompt(job_doc, search_results)
    return generate(prompt, model=model)


# ─── TUI integration (called from tui.py menu option 6) ────────────

def rag_menu(console):
    """Interactive RAG sub-menu for the TUI."""
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.rule import Rule
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box

    console.print()
    console.print(Rule("[bold magenta]RAG Analysis (Ollama)[/bold magenta]"))

    # Check Ollama
    if not check_ollama():
        console.print(
            Panel(
                "[bold red]Ollama is not running![/bold red]\n\n"
                "Install Ollama: [cyan]https://ollama.com[/cyan]\n"
                "Start it: [cyan]ollama serve[/cyan]\n"
                "Pull a model: [cyan]ollama pull llama3.2[/cyan]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return

    # Show available models
    models = list_models()
    if not models:
        console.print("[bold yellow]No models found. Pull one first: ollama pull llama3.2[/bold yellow]")
        return

    # Strip ':latest' suffix for cleaner display but keep a lookup
    model_display = [m for m in models]
    console.print(f"[dim]Available models:[/dim] [cyan]{', '.join(model_display)}[/cyan]")

    model = Prompt.ask(
        "[dim]Model to use[/dim]",
        default=models[0] if models else DEFAULT_MODEL,
    )

    # Validate: if user typed a name not in models, try partial match
    if model not in models:
        matched = [m for m in models if model in m or m.startswith(model)]
        if matched:
            model = matched[0]
            console.print(f"[dim]Resolved to model:[/dim] [cyan]{model}[/cyan]")
        else:
            console.print(f"[bold yellow]Model '{model}' not found. Using {models[0]}[/bold yellow]")
            model = models[0]

    console.print()
    console.print("  [magenta]1[/magenta]  Explain a resume-job match")
    console.print("  [magenta]2[/magenta]  Resume improvement suggestions")
    console.print("  [magenta]3[/magenta]  Job posting analysis")
    console.print("  [magenta]4[/magenta]  Back to main menu")

    choice = Prompt.ask("\n[bold]Choose[/bold]", choices=["1", "2", "3", "4"])

    if choice == "4":
        return

    # Lazy import search functions
    from match import find_jobs_for_resume, find_candidates_for_job
    from ingest_resumes import create_resume_text
    from ingest_jobs import create_job_text

    if choice == "1":
        # Explain match: pick resume, pick job
        console.print(Rule("Select Resume"))
        resumes = _load_lookup("resumes")
        items_r = sorted(resumes.items())
        for i, (rid, r) in enumerate(items_r, 1):
            console.print(f"  [magenta]{i}[/magenta]. {r.get('name', rid)}")
        rc = IntPrompt.ask("Resume #", choices=[str(i) for i in range(1, len(items_r)+1)])
        resume_doc = items_r[rc-1][1]

        console.print(Rule("Select Job"))
        jobs = _load_lookup("jobs")
        items_j = sorted(jobs.items())
        for i, (jid, j) in enumerate(items_j, 1):
            console.print(f"  [magenta]{i}[/magenta]. {j.get('title', jid)} @ {j.get('company', '')}")
        jc = IntPrompt.ask("Job #", choices=[str(i) for i in range(1, len(items_j)+1)])
        job_doc = items_j[jc-1][1]

        # Get similarity by searching
        resume_text = create_resume_text(resume_doc)
        results = find_jobs_for_resume(resume_text, k=15)
        sim = 0.0
        for r in results:
            if r[1] == job_doc["id"]:
                sim = r[0]
                break

        with Progress(SpinnerColumn("dots", style="magenta"), TextColumn("[dim]Generating explanation...[/dim]"), transient=True, console=console) as prog:
            prog.add_task("gen", total=None)
            explanation = explain_match(resume_doc["id"], job_doc["id"], sim, model=model)

        console.print(Panel(
            explanation,
            title=f"[bold]Match: {resume_doc.get('name','')} ↔ {job_doc.get('title','')}[/bold]",
            border_style="magenta",
            box=box.ROUNDED,
        ))

    elif choice == "2":
        # Resume improvement
        console.print(Rule("Select Resume"))
        resumes = _load_lookup("resumes")
        items_r = sorted(resumes.items())
        for i, (rid, r) in enumerate(items_r, 1):
            console.print(f"  [magenta]{i}[/magenta]. {r.get('name', rid)}")
        rc = IntPrompt.ask("Resume #", choices=[str(i) for i in range(1, len(items_r)+1)])
        resume_doc = items_r[rc-1][1]

        resume_text = create_resume_text(resume_doc)
        results = find_jobs_for_resume(resume_text, k=5)

        with Progress(SpinnerColumn("dots", style="magenta"), TextColumn("[dim]Generating suggestions...[/dim]"), transient=True, console=console) as prog:
            prog.add_task("gen", total=None)
            suggestions = suggest_resume_improvements(resume_doc["id"], results, model=model)

        console.print(Panel(
            suggestions,
            title=f"[bold]Improvement suggestions for {resume_doc.get('name','')}[/bold]",
            border_style="magenta",
            box=box.ROUNDED,
        ))

    elif choice == "3":
        # Job posting analysis
        console.print(Rule("Select Job"))
        jobs = _load_lookup("jobs")
        items_j = sorted(jobs.items())
        for i, (jid, j) in enumerate(items_j, 1):
            console.print(f"  [magenta]{i}[/magenta]. {j.get('title', jid)} @ {j.get('company', '')}")
        jc = IntPrompt.ask("Job #", choices=[str(i) for i in range(1, len(items_j)+1)])
        job_doc = items_j[jc-1][1]

        job_text = create_job_text(job_doc)
        results = find_candidates_for_job(job_text, k=5)

        with Progress(SpinnerColumn("dots", style="magenta"), TextColumn("[dim]Analyzing job posting...[/dim]"), transient=True, console=console) as prog:
            prog.add_task("gen", total=None)
            analysis = analyze_job_posting(job_doc["id"], results, model=model)

        console.print(Panel(
            analysis,
            title=f"[bold]Analysis: {job_doc.get('title','')} @ {job_doc.get('company','')}[/bold]",
            border_style="magenta",
            box=box.ROUNDED,
        ))


# ─── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not check_ollama():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        print("Install from: https://ollama.com")
        sys.exit(1)

    models = list_models()
    print(f"Available Ollama models: {models}")

    # Quick test
    if models:
        print(f"\nTesting with model: {models[0]}")
        result = generate("Say hello in one sentence.", model=models[0])
        print(f"Response: {result}")
    else:
        print("No models found. Pull one: ollama pull llama3.2")
