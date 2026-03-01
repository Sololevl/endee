"""
ingest_resumes.py - Ingest Resume Data into Endee

This script reads resume JSON files from the data/resumes directory,
generates embeddings for each resume, and stores them in Endee vector database.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from embedder import embed_text, get_embedding_dimension
from endee_client import EndeeClient


# Configuration
RESUMES_DIR = Path(__file__).parent.parent / "data" / "resumes"
INDEX_NAME = "resumes"


def load_resumes() -> List[Dict[str, Any]]:
    """
    Load all resume JSON files from the data/resumes directory.
    
    Returns:
        List of resume dictionaries
    """
    resumes = []
    
    # Check if directory exists
    if not RESUMES_DIR.exists():
        print(f"Error: Resumes directory not found at {RESUMES_DIR}")
        return resumes
    
    # Load each JSON file
    for file_path in RESUMES_DIR.glob("*.json"):
        print(f"Loading: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            resume = json.load(f)
            resumes.append(resume)
    
    return resumes


def create_resume_text(resume: Dict[str, Any]) -> str:
    """
    Create a searchable text representation of a resume.
    This text will be converted to an embedding.
    
    Args:
        resume: Resume dictionary with name, summary, skills, experience, etc.
    
    Returns:
        Concatenated text suitable for embedding
    """
    # Combine relevant fields into a single text
    parts = []
    
    # Add summary (most important)
    if "summary" in resume:
        parts.append(resume["summary"])
    
    # Add skills as comma-separated list
    if "skills" in resume:
        skills_text = "Skills: " + ", ".join(resume["skills"])
        parts.append(skills_text)
    
    # Add job titles from experience
    if "experience" in resume:
        for exp in resume["experience"]:
            if "title" in exp:
                parts.append(exp["title"])
            if "description" in exp:
                parts.append(exp["description"])
    
    # Add education
    if "education" in resume:
        for edu in resume["education"]:
            if "degree" in edu:
                parts.append(edu["degree"])
            if "field" in edu:
                parts.append(edu["field"])
    
    return " ".join(parts)


def create_filter_data(resume: Dict[str, Any]) -> str:
    """
    Create filter metadata as a JSON string.
    This allows filtering search results by criteria like experience or location.
    
    Args:
        resume: Resume dictionary
    
    Returns:
        JSON string with filterable fields
    """
    filter_obj = {}
    
    # Experience in years (numeric filter)
    if "years_experience" in resume:
        filter_obj["years_experience"] = resume["years_experience"]
    
    # Location (category filter)
    if "location" in resume:
        filter_obj["location"] = resume["location"]
    
    # Is open to work (boolean filter)
    if "is_open_to_work" in resume:
        filter_obj["is_open_to_work"] = 1 if resume["is_open_to_work"] else 0
    
    return json.dumps(filter_obj)


def ingest_resumes():
    """
    Main function to ingest all resumes into Endee.
    """
    print("=" * 60)
    print("Resume Ingestion Pipeline")
    print("=" * 60)
    
    # Step 1: Load resumes from JSON files
    print("\n[Step 1] Loading resumes...")
    resumes = load_resumes()
    if not resumes:
        print("No resumes found. Please add JSON files to data/resumes/")
        return
    print(f"Loaded {len(resumes)} resumes")
    
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
    
    for resume in resumes:
        # Create text representation for embedding
        text = create_resume_text(resume)
        
        # Generate embedding
        embedding = embed_text(text)
        
        # Create vector object for Endee
        vector_obj = {
            "id": resume.get("id", resume.get("name", "unknown")),
            "vector": embedding,
            "filter": create_filter_data(resume),
            "meta": json.dumps({
                "name": resume.get("name", ""),
                "title": resume.get("title", ""),
                "email": resume.get("email", "")
            })
        }
        vectors.append(vector_obj)
        print(f"  Embedded: {resume.get('name', 'Unknown')}")
    
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
    print("Resume ingestion completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    ingest_resumes()
