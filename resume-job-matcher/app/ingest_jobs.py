"""
ingest_jobs.py - Ingest Job Data into Endee

This script reads job posting JSON files from the data/jobs directory,
generates embeddings for each job, and stores them in Endee vector database.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from embedder import embed_text, get_embedding_dimension
from endee_client import EndeeClient


# Configuration
JOBS_DIR = Path(__file__).parent.parent / "data" / "jobs"
INDEX_NAME = "jobs"


def load_jobs() -> List[Dict[str, Any]]:
    """
    Load all job JSON files from the data/jobs directory.
    
    Returns:
        List of job dictionaries
    """
    jobs = []
    
    # Check if directory exists
    if not JOBS_DIR.exists():
        print(f"Error: Jobs directory not found at {JOBS_DIR}")
        return jobs
    
    # Load each JSON file
    for file_path in JOBS_DIR.glob("*.json"):
        print(f"Loading: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            job = json.load(f)
            jobs.append(job)
    
    return jobs


def create_job_text(job: Dict[str, Any]) -> str:
    """
    Create a searchable text representation of a job posting.
    This text will be converted to an embedding.
    
    Args:
        job: Job dictionary with title, description, requirements, etc.
    
    Returns:
        Concatenated text suitable for embedding
    """
    parts = []
    
    # Add job title (very important)
    if "title" in job:
        parts.append(job["title"])
    
    # Add job description
    if "description" in job:
        parts.append(job["description"])
    
    # Add required skills
    if "required_skills" in job:
        skills_text = "Required skills: " + ", ".join(job["required_skills"])
        parts.append(skills_text)
    
    # Add preferred skills
    if "preferred_skills" in job:
        preferred_text = "Preferred skills: " + ", ".join(job["preferred_skills"])
        parts.append(preferred_text)
    
    # Add responsibilities
    if "responsibilities" in job:
        for resp in job["responsibilities"]:
            parts.append(resp)
    
    return " ".join(parts)


def create_filter_data(job: Dict[str, Any]) -> str:
    """
    Create filter metadata as a JSON string.
    This allows filtering search results by criteria.
    
    Args:
        job: Job dictionary
    
    Returns:
        JSON string with filterable fields
    """
    filter_obj = {}
    
    # Minimum experience required (numeric filter)
    if "min_experience" in job:
        filter_obj["min_experience"] = job["min_experience"]
    
    # Location (category filter)
    if "location" in job:
        filter_obj["location"] = job["location"]
    
    # Remote friendly (boolean filter)
    if "remote_friendly" in job:
        filter_obj["remote_friendly"] = 1 if job["remote_friendly"] else 0
    
    return json.dumps(filter_obj)


def ingest_jobs():
    """
    Main function to ingest all jobs into Endee.
    """
    print("=" * 60)
    print("Job Posting Ingestion Pipeline")
    print("=" * 60)
    
    # Step 1: Load jobs from JSON files
    print("\n[Step 1] Loading job postings...")
    jobs = load_jobs()
    if not jobs:
        print("No jobs found. Please add JSON files to data/jobs/")
        return
    print(f"Loaded {len(jobs)} job postings")
    
    # Step 2: Connect to Endee
    print("\n[Step 2] Connecting to Endee...")
    client = EndeeClient()
    
    # Step 3: Create index (if it doesn't exist)
    print(f"\n[Step 3] Creating index '{INDEX_NAME}'...")
    try:
        response = client.create_index(
            index_name=INDEX_NAME,
            dimension=get_embedding_dimension(),  # 384 for MiniLM
            space_type="cosine",  # Cosine similarity for text
            m=16,  # HNSW connections
            ef_con=200,  # Construction ef
            precision="float32"  # Full precision for accuracy
        )
        print(f"Index created: {response}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Index '{INDEX_NAME}' already exists, continuing...")
        else:
            raise e
    
    # Step 4: Generate embeddings and prepare vectors
    print("\n[Step 4] Generating embeddings...")
    vectors = []
    
    for job in jobs:
        # Create text representation for embedding
        text = create_job_text(job)
        
        # Generate embedding
        embedding = embed_text(text)
        
        # Create vector object for Endee
        vector_obj = {
            "id": job.get("id", job.get("title", "unknown")),
            "vector": embedding,
            "filter": create_filter_data(job),
            "meta": json.dumps({
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", "")
            })
        }
        vectors.append(vector_obj)
        print(f"  Embedded: {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}")
    
    # Step 5: Insert vectors into Endee
    print(f"\n[Step 5] Inserting {len(vectors)} vectors into Endee...")
    response = client.insert_vectors(INDEX_NAME, vectors)
    print(f"Insertion complete!")
    
    # Step 6: Verify insertion
    print("\n[Step 6] Verifying...")
    try:
        info = client.get_index_info(INDEX_NAME)
        print(f"Index info: {info}")
    except Exception as e:
        print(f"Could not get index info: {e}")
    
    print("\n" + "=" * 60)
    print("Job ingestion completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    ingest_jobs()
