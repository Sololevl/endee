"""
endee_client.py - Endee REST API Wrapper

This module provides a simple Python client for interacting with Endee
vector database via its REST API. It handles index creation, vector
insertion, and similarity search operations.
"""

import requests
import msgpack
from typing import List, Dict, Any, Optional


class EndeeClient:
    """
    A simple client for Endee vector database REST API.
    
    Endee is a high-performance vector database that stores embeddings
    and allows fast similarity search using HNSW algorithm.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8080",
        auth_token: Optional[str] = None
    ):
        """
        Initialize the Endee client.
        
        Args:
            base_url: The base URL of the Endee server (default: localhost:8080)
            auth_token: Optional authentication token (if Endee is running in auth mode)
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        
        # Add auth token if provided
        if auth_token:
            self.headers["Authorization"] = auth_token
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise exceptions for errors.
        """
        if response.status_code >= 400:
            try:
                error_msg = response.json().get("error", response.text)
            except:
                error_msg = response.text
            raise Exception(f"Endee API error ({response.status_code}): {error_msg}")
        
        # Check if response is msgpack (search results)
        if response.headers.get("Content-Type") == "application/msgpack":
            return msgpack.unpackb(response.content, raw=False)
        
        # Try JSON, otherwise return text
        try:
            return response.json()
        except:
            return {"message": response.text}
    
    def create_index(
        self,
        index_name: str,
        dimension: int,
        space_type: str = "cosine",
        m: int = 16,
        ef_con: int = 200,
        precision: str = "float32"
    ) -> Dict[str, Any]:
        """
        Create a new vector index in Endee.
        
        Args:
            index_name: Name for the index (e.g., "resumes" or "jobs")
            dimension: Dimension of vectors to store (must match embedding dimension)
            space_type: Distance metric - "cosine", "l2", or "ip" (inner product)
            m: HNSW parameter - number of connections per node (higher = better recall)
            ef_con: HNSW parameter - construction ef (higher = better quality, slower build)
            precision: Vector precision - "float32", "float16", "int16", "int8"
        
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/api/v1/index/create"
        payload = {
            "index_name": index_name,
            "dim": dimension,
            "space_type": space_type,
            "M": m,
            "ef_con": ef_con,
            "precision": precision
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        return self._handle_response(response)
    
    def insert_vectors(
        self,
        index_name: str,
        vectors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Insert vectors into an index.
        
        Args:
            index_name: Name of the target index
            vectors: List of vector objects, each containing:
                - id: Unique identifier for the vector
                - vector: List of floats (the embedding)
                - filter: Optional JSON string with filterable metadata
                - meta: Optional metadata string
        
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/vector/insert"
        response = requests.post(url, json=vectors, headers=self.headers)
        return self._handle_response(response)
    
    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 10,
        ef: int = 0,
        filter_json: Optional[str] = None,
        include_vectors: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar vectors in an index.
        
        Args:
            index_name: Name of the index to search
            query_vector: The query embedding vector
            k: Number of results to return
            ef: Search ef parameter (higher = better recall, slower search)
            filter_json: Optional JSON string with filter conditions
            include_vectors: Whether to include vectors in results
        
        Returns:
            Search results with IDs, distances, and optionally metadata
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/search"
        payload = {
            "vector": query_vector,
            "k": k,
            "include_vectors": include_vectors
        }
        
        if ef > 0:
            payload["ef"] = ef
        
        if filter_json:
            payload["filter"] = filter_json
        
        response = requests.post(url, json=payload, headers=self.headers)
        return self._handle_response(response)
    
    def delete_index(self, index_name: str) -> Dict[str, Any]:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
        
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/delete"
        response = requests.delete(url, headers=self.headers)
        return self._handle_response(response)
    
    def list_indexes(self) -> Dict[str, Any]:
        """
        List all indexes.
        
        Returns:
            List of index names and their info
        """
        url = f"{self.base_url}/api/v1/index/list"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """
        Get information about an index.
        
        Args:
            index_name: Name of the index
        
        Returns:
            Index information including vector count, dimension, etc.
        """
        url = f"{self.base_url}/api/v1/index/{index_name}/info"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)


# Example usage
if __name__ == "__main__":
    # Create client
    client = EndeeClient()
    
    print("Testing Endee connection...")
    try:
        indexes = client.list_indexes()
        print(f"Connected! Current indexes: {indexes}")
    except Exception as e:
        print(f"Could not connect to Endee: {e}")
        print("Make sure Endee is running on http://localhost:8080")
