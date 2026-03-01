"""
pipeline.py - Enhanced Ingestion Pipeline

A unified, improved ingestion pipeline that replaces the separate
ingest_resumes.py and ingest_jobs.py scripts. Features:

  - Schema validation before ingestion
  - Duplicate detection (skip IDs already in Endee)
  - Better text templates for higher-quality embeddings
  - Support for PDF/DOCX parsing (optional dependencies)
  - Progress reporting
  - Dry-run mode for testing
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from embedder import embed_text, embed_texts, get_embedding_dimension
from endee_client import EndeeClient
from schema import validate_resume, validate_job, validate_batch

# ─── Configuration ──────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
RESUME_INDEX = "resumes"
JOB_INDEX = "jobs"


# ─── Enhanced text templates ────────────────────────────────────────

def build_resume_embedding_text(resume: Dict[str, Any]) -> str:
    """
    Build a rich, structured text representation of a resume optimised
    for semantic embedding quality.

    Uses a template approach that weights the most important fields
    and provides clear semantic context to the embedding model.
    """
    parts: List[str] = []

    # Title and level (high weight - appears first)
    title = resume.get("title", "")
    if title:
        parts.append(f"Professional Title: {title}")

    # Summary (most important semantic content)
    summary = resume.get("summary", "")
    if summary:
        parts.append(f"Summary: {summary}")

    # Core skills (very important for matching)
    skills = resume.get("skills", [])
    if skills:
        parts.append(f"Technical Skills: {', '.join(skills)}")

    # Experience details
    for exp in resume.get("experience", []):
        exp_text = f"Role: {exp.get('title', '')} at {exp.get('company', '')}"
        if exp.get("description"):
            exp_text += f". {exp['description']}"
        parts.append(exp_text)

    # Education
    for edu in resume.get("education", []):
        edu_text = f"Education: {edu.get('degree', '')} in {edu.get('field', '')}"
        if edu.get("school"):
            edu_text += f" from {edu['school']}"
        parts.append(edu_text)

    # Certifications
    for cert in resume.get("certifications", []):
        if isinstance(cert, str):
            parts.append(f"Certification: {cert}")
        elif isinstance(cert, dict):
            parts.append(f"Certification: {cert.get('name', '')}")

    # Years of experience as context
    yoe = resume.get("years_experience")
    if yoe is not None:
        parts.append(f"Total experience: {yoe} years")

    return " | ".join(parts)


def build_job_embedding_text(job: Dict[str, Any]) -> str:
    """
    Build a rich, structured text representation of a job posting
    optimised for semantic embedding quality.
    """
    parts: List[str] = []

    # Title (most important)
    title = job.get("title", "")
    if title:
        parts.append(f"Job Title: {title}")

    # Description
    desc = job.get("description", "")
    if desc:
        parts.append(f"Description: {desc}")

    # Required skills
    req = job.get("required_skills", [])
    if req:
        parts.append(f"Required Skills: {', '.join(req)}")

    # Preferred skills
    pref = job.get("preferred_skills", [])
    if pref:
        parts.append(f"Preferred Skills: {', '.join(pref)}")

    # Responsibilities
    resps = job.get("responsibilities", [])
    if resps:
        parts.append(f"Responsibilities: {'; '.join(resps)}")

    # Experience level
    min_exp = job.get("min_experience")
    if min_exp is not None:
        parts.append(f"Minimum experience: {min_exp} years")

    return " | ".join(parts)


# ─── Document parsing (optional dependencies) ──────────────────────

def parse_pdf(file_path: Path) -> Optional[str]:
    """
    Parse a PDF file and extract text.
    Requires: pip install PyPDF2
    """
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip() if text.strip() else None
    except ImportError:
        print("  [WARN] PyPDF2 not installed. Run: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"  [WARN] Could not parse PDF {file_path.name}: {e}")
        return None


def parse_docx(file_path: Path) -> Optional[str]:
    """
    Parse a DOCX file and extract text.
    Requires: pip install python-docx
    """
    try:
        import docx
        doc = docx.Document(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text.strip() if text.strip() else None
    except ImportError:
        print("  [WARN] python-docx not installed. Run: pip install python-docx")
        return None
    except Exception as e:
        print(f"  [WARN] Could not parse DOCX {file_path.name}: {e}")
        return None


# ─── Loader ─────────────────────────────────────────────────────────

def load_json_documents(directory: Path) -> List[Dict[str, Any]]:
    """Load all JSON files from a directory."""
    docs = []
    if not directory.exists():
        return docs
    for fp in sorted(directory.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
                doc["_source_file"] = fp.name
                docs.append(doc)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Invalid JSON in {fp.name}: {e}")
        except Exception as e:
            print(f"  [ERROR] Cannot read {fp.name}: {e}")
    return docs


# ─── Filter builder ────────────────────────────────────────────────

def build_resume_filter(resume: Dict[str, Any]) -> str:
    """Build filter JSON for a resume."""
    obj = {}
    if "years_experience" in resume:
        obj["years_experience"] = resume["years_experience"]
    if "location" in resume:
        obj["location"] = resume["location"]
    if "is_open_to_work" in resume:
        obj["is_open_to_work"] = 1 if resume["is_open_to_work"] else 0
    return json.dumps(obj)


def build_job_filter(job: Dict[str, Any]) -> str:
    """Build filter JSON for a job."""
    obj = {}
    if "min_experience" in job:
        obj["min_experience"] = job["min_experience"]
    if "location" in job:
        obj["location"] = job["location"]
    if "remote_friendly" in job:
        obj["remote_friendly"] = 1 if job["remote_friendly"] else 0
    return json.dumps(obj)


# ─── Deduplication ──────────────────────────────────────────────────

def get_existing_ids(client: EndeeClient, index_name: str) -> set:
    """
    Try to find which IDs already exist in an index.
    This is a best-effort check — if the index doesn't exist yet
    or the API doesn't support listing, returns empty set.
    """
    # Endee doesn't have a "list all IDs" endpoint, so we track
    # locally via a metadata file.
    meta_file = DATA_DIR / f".{index_name}_ingested_ids.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_ingested_ids(index_name: str, ids: set):
    """Save the set of ingested IDs for dedup tracking."""
    meta_file = DATA_DIR / f".{index_name}_ingested_ids.json"
    with open(meta_file, "w") as f:
        json.dump(sorted(ids), f)


# ─── Main pipeline ─────────────────────────────────────────────────

def ingest_pipeline(
    doc_type: str = "all",
    dry_run: bool = False,
    skip_validation: bool = False,
    force: bool = False,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Run the enhanced ingestion pipeline.

    Args:
        doc_type: "resumes", "jobs", or "all"
        dry_run: If True, validate and report but don't insert
        skip_validation: Skip schema validation
        force: Re-ingest even if IDs already exist
        batch_size: Number of vectors to insert per batch

    Returns:
        Summary dict with counts and errors
    """
    summary = {
        "resumes": {"loaded": 0, "valid": 0, "skipped": 0, "ingested": 0, "errors": []},
        "jobs": {"loaded": 0, "valid": 0, "skipped": 0, "ingested": 0, "errors": []},
    }

    client = EndeeClient()

    # ── Process resumes ────
    if doc_type in ("resumes", "all"):
        print("\n" + "=" * 60)
        print("  RESUME INGESTION PIPELINE")
        print("=" * 60)

        docs = load_json_documents(DATA_DIR / "resumes")
        summary["resumes"]["loaded"] = len(docs)
        print(f"\n  Loaded {len(docs)} resume files")

        # Validate
        if not skip_validation:
            print("  Validating schemas...")
            valid_count, invalid_count, errors = validate_batch(docs, "resume")
            summary["resumes"]["valid"] = valid_count
            summary["resumes"]["errors"].extend(errors)
            if errors:
                for err in errors:
                    print(f"    WARN: {err}")
            print(f"  Validation: {valid_count} valid, {invalid_count} invalid")
        else:
            summary["resumes"]["valid"] = len(docs)

        if dry_run:
            print("  [DRY RUN] Skipping actual ingestion")
        else:
            # Create index
            try:
                client.create_index(
                    index_name=RESUME_INDEX,
                    dimension=get_embedding_dimension(),
                    space_type="cosine",
                    m=16,
                    ef_con=200,
                    precision="float32",
                )
                print(f"  Created index '{RESUME_INDEX}'")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"  Index '{RESUME_INDEX}' already exists")
                else:
                    print(f"  ERROR creating index: {e}")
                    summary["resumes"]["errors"].append(str(e))

            # Dedup
            existing = get_existing_ids(client, RESUME_INDEX) if not force else set()
            all_ids = set(existing)

            # Embed and insert
            vectors = []
            for doc in docs:
                doc_id = doc.get("id", "")
                if doc_id in existing:
                    print(f"    Skip (exists): {doc_id}")
                    summary["resumes"]["skipped"] += 1
                    continue

                text = build_resume_embedding_text(doc)
                emb = embed_text(text)

                vectors.append({
                    "id": doc_id,
                    "vector": emb,
                    "filter": build_resume_filter(doc),
                    "meta": json.dumps({
                        "name": doc.get("name", ""),
                        "title": doc.get("title", ""),
                        "email": doc.get("email", ""),
                    }),
                })
                all_ids.add(doc_id)
                print(f"    Embedded: {doc.get('name', doc_id)}")

                # Batch insert
                if len(vectors) >= batch_size:
                    client.insert_vectors(RESUME_INDEX, vectors)
                    summary["resumes"]["ingested"] += len(vectors)
                    vectors = []

            if vectors:
                client.insert_vectors(RESUME_INDEX, vectors)
                summary["resumes"]["ingested"] += len(vectors)

            save_ingested_ids(RESUME_INDEX, all_ids)
            print(f"\n  Ingested {summary['resumes']['ingested']} resumes")

    # ── Process jobs ────
    if doc_type in ("jobs", "all"):
        print("\n" + "=" * 60)
        print("  JOB INGESTION PIPELINE")
        print("=" * 60)

        docs = load_json_documents(DATA_DIR / "jobs")
        summary["jobs"]["loaded"] = len(docs)
        print(f"\n  Loaded {len(docs)} job files")

        if not skip_validation:
            print("  Validating schemas...")
            valid_count, invalid_count, errors = validate_batch(docs, "job")
            summary["jobs"]["valid"] = valid_count
            summary["jobs"]["errors"].extend(errors)
            if errors:
                for err in errors:
                    print(f"    WARN: {err}")
            print(f"  Validation: {valid_count} valid, {invalid_count} invalid")
        else:
            summary["jobs"]["valid"] = len(docs)

        if dry_run:
            print("  [DRY RUN] Skipping actual ingestion")
        else:
            try:
                client.create_index(
                    index_name=JOB_INDEX,
                    dimension=get_embedding_dimension(),
                    space_type="cosine",
                    m=16,
                    ef_con=200,
                    precision="float32",
                )
                print(f"  Created index '{JOB_INDEX}'")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"  Index '{JOB_INDEX}' already exists")
                else:
                    print(f"  ERROR creating index: {e}")
                    summary["jobs"]["errors"].append(str(e))

            existing = get_existing_ids(client, JOB_INDEX) if not force else set()
            all_ids = set(existing)

            vectors = []
            for doc in docs:
                doc_id = doc.get("id", "")
                if doc_id in existing:
                    print(f"    Skip (exists): {doc_id}")
                    summary["jobs"]["skipped"] += 1
                    continue

                text = build_job_embedding_text(doc)
                emb = embed_text(text)

                vectors.append({
                    "id": doc_id,
                    "vector": emb,
                    "filter": build_job_filter(doc),
                    "meta": json.dumps({
                        "title": doc.get("title", ""),
                        "company": doc.get("company", ""),
                        "location": doc.get("location", ""),
                    }),
                })
                all_ids.add(doc_id)
                print(f"    Embedded: {doc.get('title', doc_id)} @ {doc.get('company', '')}")

                if len(vectors) >= batch_size:
                    client.insert_vectors(JOB_INDEX, vectors)
                    summary["jobs"]["ingested"] += len(vectors)
                    vectors = []

            if vectors:
                client.insert_vectors(JOB_INDEX, vectors)
                summary["jobs"]["ingested"] += len(vectors)

            save_ingested_ids(JOB_INDEX, all_ids)
            print(f"\n  Ingested {summary['jobs']['ingested']} jobs")

    # ── Summary ────
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for dtype in ["resumes", "jobs"]:
        s = summary[dtype]
        if s["loaded"] > 0:
            print(f"\n  {dtype.upper()}:")
            print(f"    Loaded:   {s['loaded']}")
            print(f"    Valid:    {s['valid']}")
            print(f"    Skipped:  {s['skipped']}")
            print(f"    Ingested: {s['ingested']}")
            if s["errors"]:
                print(f"    Errors:   {len(s['errors'])}")

    return summary


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced ingestion pipeline for Resume-Job Matcher"
    )
    parser.add_argument(
        "--type",
        choices=["resumes", "jobs", "all"],
        default="all",
        help="What to ingest (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only, don't insert into Endee",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if IDs already exist",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip schema validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of vectors per insert batch (default: 50)",
    )

    args = parser.parse_args()

    ingest_pipeline(
        doc_type=args.type,
        dry_run=args.dry_run,
        force=args.force,
        skip_validation=args.skip_validation,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
