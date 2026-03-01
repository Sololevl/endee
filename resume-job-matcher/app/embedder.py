"""
embedder.py - Text to Vector Embeddings

This module converts text (resumes and job descriptions) into numerical
vector embeddings using sentence-transformers. These embeddings capture
the semantic meaning of the text, allowing us to find similar documents.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union


# Initialize the embedding model
# 'all-MiniLM-L6-v2' is a good balance of speed and quality
# It produces 384-dimensional vectors
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Load model once and reuse (cached globally for efficiency)
_model = None


def get_model() -> SentenceTransformer:
    """
    Get or initialize the sentence transformer model.
    Uses lazy loading to avoid loading the model until needed.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully!")
    return _model


def embed_text(text: str) -> List[float]:
    """
    Convert a single text string into a vector embedding.
    
    Args:
        text: The text to embed (e.g., a resume summary or job description)
    
    Returns:
        A list of floats representing the embedding vector
    """
    model = get_model()
    # encode() returns a numpy array, we convert to list for JSON serialization
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert multiple text strings into vector embeddings.
    More efficient than calling embed_text multiple times.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    model = get_model()
    # Batch encoding is more efficient for multiple texts
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """
    Return the dimension of embedding vectors produced by the model.
    This is needed when creating the Endee index.
    """
    return EMBEDDING_DIMENSION


# Example usage and testing
if __name__ == "__main__":
    # Test the embedding functionality
    test_texts = [
        "Senior Python developer with 5 years of machine learning experience",
        "Looking for a Python ML engineer with deep learning skills",
        "Marketing manager with social media expertise"
    ]
    
    print(f"Embedding dimension: {get_embedding_dimension()}")
    print("\nGenerating embeddings for test texts...")
    
    embeddings = embed_texts(test_texts)
    
    for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
        print(f"\n[{i+1}] '{text[:50]}...'")
        print(f"    Vector (first 5 dims): {emb[:5]}")
        print(f"    Vector length: {len(emb)}")
