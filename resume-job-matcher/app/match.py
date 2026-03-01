"""
match.py - Match Jobs to Resumes using Semantic Search

This script provides functionality to:
1. Find best jobs for a given resume
2. Find best candidates for a given job
3. Interactive matching with filters

It uses the embeddings stored in Endee to find semantically similar matches.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from embedder import embed_text
from endee_client import EndeeClient


# Initialize Endee client
client = EndeeClient()

# Paths to sample data (for looking up display info by ID)
DATA_DIR = Path(__file__).parent.parent / "data"


def _load_lookup(subdir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all JSON files from a data subdirectory into a dict keyed by ID.
    Used to look up display info (name, title, etc.) after search.
    """
    lookup = {}
    folder = DATA_DIR / subdir
    if folder.exists():
        for fp in folder.glob("*.json"):
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
                doc_id = doc.get("id", "")
                if doc_id:
                    lookup[doc_id] = doc
    return lookup


def find_jobs_for_resume(
    resume_text: str,
    k: int = 5,
    min_experience: Optional[int] = None,
    location: Optional[str] = None,
    remote_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Find the best matching jobs for a given resume text.
    
    Args:
        resume_text: Text describing the candidate's skills and experience
        k: Number of jobs to return
        min_experience: Filter jobs that require <= this years of experience
        location: Filter jobs by location
        remote_only: Only show remote-friendly jobs
    
    Returns:
        List of matching jobs with similarity scores
    """
    # Generate embedding for the resume text
    query_vector = embed_text(resume_text)
    
    # Build filter if needed
    filter_conditions = []
    
    if min_experience is not None:
        # Jobs requiring at most this much experience (numeric range)
        filter_conditions.append({"min_experience": {"$lte": min_experience}})
    
    if location:
        # Category exact-match filter
        filter_conditions.append({"location": location})
    
    if remote_only:
        # Boolean stored as category 1/0
        filter_conditions.append({"remote_friendly": 1})
    
    # Convert filter to JSON string if we have conditions
    filter_json = json.dumps(filter_conditions) if filter_conditions else None
    
    # Search — fall back to unfiltered search if filter causes an error
    try:
        results = client.search(
            index_name="jobs",
            query_vector=query_vector,
            k=k,
            filter_json=filter_json
        )
    except Exception as e:
        if filter_json:
            # Retry without filter and warn
            import sys
            print(f"[Warning] Filter search failed ({e}), retrying without filters...", file=sys.stderr)
            results = client.search(
                index_name="jobs",
                query_vector=query_vector,
                k=k,
            )
        else:
            raise
    
    return results


def find_candidates_for_job(
    job_text: str,
    k: int = 5,
    min_experience: Optional[int] = None,
    location: Optional[str] = None,
    open_to_work_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Find the best matching candidates for a given job description.
    
    Args:
        job_text: Text describing the job requirements and responsibilities
        k: Number of candidates to return
        min_experience: Filter candidates with at least this many years experience
        location: Filter candidates by location
        open_to_work_only: Only show candidates open to new opportunities
    
    Returns:
        List of matching candidates with similarity scores
    """
    # Generate embedding for the job text
    query_vector = embed_text(job_text)
    
    # Build filter if needed
    filter_conditions = []
    
    if min_experience is not None:
        # Candidates with at least this much experience (numeric range)
        filter_conditions.append({"years_experience": {"$gte": min_experience}})
    
    if location:
        # Category exact-match filter
        filter_conditions.append({"location": location})
    
    if open_to_work_only:
        # Boolean stored as category 1/0
        filter_conditions.append({"is_open_to_work": 1})
    
    # Convert filter to JSON string if we have conditions
    filter_json = json.dumps(filter_conditions) if filter_conditions else None
    
    # Search — fall back to unfiltered search if filter causes an error
    try:
        results = client.search(
            index_name="resumes",
            query_vector=query_vector,
            k=k,
            filter_json=filter_json
        )
    except Exception as e:
        if filter_json:
            # Retry without filter and warn
            import sys
            print(f"[Warning] Filter search failed ({e}), retrying without filters...", file=sys.stderr)
            results = client.search(
                index_name="resumes",
                query_vector=query_vector,
                k=k,
            )
        else:
            raise
    
    return results


def format_results(results: list, result_type: str = "matches") -> str:
    """
    Format search results for display.
    
    Endee returns results as a list of lists via msgpack.
    Each result is: [similarity, id, meta_bytes, filter_str, norm, vector_list]
    
    We look up the original JSON data files using the ID to display
    human-readable info (name, title, company, etc.).
    
    Args:
        results: Search results from Endee (list of result tuples)
        result_type: Type of results ("jobs" or "candidates")
    
    Returns:
        Formatted string for display
    """
    output = []
    
    if not results:
        return "No matches found."
    
    # Load lookup tables from JSON files for display purposes
    if result_type == "jobs":
        lookup = _load_lookup("jobs")
    elif result_type == "candidates":
        lookup = _load_lookup("resumes")
    else:
        lookup = {**_load_lookup("jobs"), **_load_lookup("resumes")}
    
    for i, result in enumerate(results):
        # Unpack: [similarity, id, meta_bytes, filter_str, norm, vector_list]
        similarity = result[0]  # Cosine similarity (higher = more similar)
        id_ = result[1]
        
        similarity_pct = similarity * 100
        
        line = f"\n{i+1}. ID: {id_}"
        line += f"\n   Similarity: {similarity_pct:.1f}%"
        
        # Look up display info from original data files
        doc = lookup.get(id_, {})
        if doc:
            if result_type == "jobs":
                if doc.get("title"):
                    line += f"\n   Title: {doc['title']}"
                if doc.get("company"):
                    line += f"\n   Company: {doc['company']}"
                if doc.get("location"):
                    line += f"\n   Location: {doc['location']}"
                if doc.get("salary_range"):
                    line += f"\n   Salary: {doc['salary_range']}"
            elif result_type == "candidates":
                if doc.get("name"):
                    line += f"\n   Name: {doc['name']}"
                if doc.get("title"):
                    line += f"\n   Title: {doc['title']}"
                if doc.get("location"):
                    line += f"\n   Location: {doc['location']}"
                if doc.get("years_experience") is not None:
                    line += f"\n   Experience: {doc['years_experience']} years"
            else:
                # Generic: show common fields
                for key in ["name", "title", "company", "location"]:
                    if doc.get(key):
                        line += f"\n   {key.title()}: {doc[key]}"
        
        output.append(line)
    
    return "\n".join(output)


def interactive_matcher():
    """
    Interactive command-line interface for matching.
    """
    print("=" * 60)
    print("Resume-Job Matcher - Interactive Mode")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("  1. Find jobs for a resume (paste resume text)")
        print("  2. Find candidates for a job (paste job description)")
        print("  3. Quick match with sample data")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\nPaste your resume text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            resume_text = "\n".join(lines)
            if resume_text:
                print("\nApply filters? (y/n): ", end="")
                apply_filters = input().strip().lower() == "y"
                
                location = None
                remote_only = False
                
                if apply_filters:
                    location = input("Location filter (or press Enter to skip): ").strip() or None
                    remote_only = input("Remote only? (y/n): ").strip().lower() == "y"
                
                print("\nSearching for matching jobs...")
                results = find_jobs_for_resume(
                    resume_text,
                    k=5,
                    location=location,
                    remote_only=remote_only
                )
                print("\n" + "-" * 40)
                print("TOP MATCHING JOBS:")
                print("-" * 40)
                print(format_results(results, "jobs"))
        
        elif choice == "2":
            print("\nPaste your job description (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            job_text = "\n".join(lines)
            if job_text:
                print("\nApply filters? (y/n): ", end="")
                apply_filters = input().strip().lower() == "y"
                
                min_exp = None
                open_only = False
                
                if apply_filters:
                    exp_input = input("Minimum years experience (or press Enter to skip): ").strip()
                    min_exp = int(exp_input) if exp_input else None
                    open_only = input("Open to work only? (y/n): ").strip().lower() == "y"
                
                print("\nSearching for matching candidates...")
                results = find_candidates_for_job(
                    job_text,
                    k=5,
                    min_experience=min_exp,
                    open_to_work_only=open_only
                )
                print("\n" + "-" * 40)
                print("TOP MATCHING CANDIDATES:")
                print("-" * 40)
                print(format_results(results, "candidates"))
        
        elif choice == "3":
            # Quick demo with sample text
            print("\n" + "=" * 40)
            print("DEMO: Finding jobs for a Python developer")
            print("=" * 40)
            
            sample_resume = """
            Senior Python developer with 5 years of experience in machine learning
            and data engineering. Skilled in TensorFlow, PyTorch, pandas, and SQL.
            Built production ML pipelines serving millions of users.
            """
            
            print(f"\nSample Resume:\n{sample_resume}")
            print("\nSearching...")
            
            results = find_jobs_for_resume(sample_resume.strip(), k=3)
            print("\n" + "-" * 40)
            print("MATCHING JOBS:")
            print("-" * 40)
            print(format_results(results, "jobs"))
            
            print("\n" + "=" * 40)
            print("DEMO: Finding candidates for a ML Engineer role")
            print("=" * 40)
            
            sample_job = """
            Machine Learning Engineer needed. Must have experience with Python,
            deep learning frameworks (TensorFlow/PyTorch), and deploying ML models
            to production. Experience with NLP is a plus.
            """
            
            print(f"\nSample Job:\n{sample_job}")
            print("\nSearching...")
            
            results = find_candidates_for_job(sample_job.strip(), k=3)
            print("\n" + "-" * 40)
            print("MATCHING CANDIDATES:")
            print("-" * 40)
            print(format_results(results, "candidates"))
        
        elif choice == "4":
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid option. Please try again.")


def demo_match():
    """
    Run a simple demo matching without interactive mode.
    Good for testing the system.
    """
    print("=" * 60)
    print("Resume-Job Matcher - Demo")
    print("=" * 60)
    
    # Demo 1: Find jobs for a Python developer
    print("\n[Demo 1] Finding jobs for a Python ML developer...")
    resume_text = """
    Experienced Python developer specializing in machine learning and AI.
    5 years building production ML systems. Expert in TensorFlow, PyTorch,
    scikit-learn. Strong background in NLP and computer vision.
    """
    
    try:
        results = find_jobs_for_resume(resume_text.strip(), k=3)
        print("\nTop matching jobs:")
        print(format_results(results, "jobs"))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've run ingest_jobs.py first!")
    
    # Demo 2: Find candidates for a job
    print("\n" + "-" * 60)
    print("\n[Demo 2] Finding candidates for a Data Engineering role...")
    job_text = """
    Data Engineer position. Looking for someone with Python, SQL,
    and experience building data pipelines. Knowledge of Spark,
    Airflow, and cloud platforms (AWS/GCP) is required.
    """
    
    try:
        results = find_candidates_for_job(job_text.strip(), k=3)
        print("\nTop matching candidates:")
        print(format_results(results, "candidates"))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've run ingest_resumes.py first!")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run demo mode
        demo_match()
    else:
        # Run interactive mode
        interactive_matcher()
