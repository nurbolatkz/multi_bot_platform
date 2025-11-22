import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
import os

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # For text-embedding-3-small

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class VectorSearcher:
    """Handles vector similarity search with enhanced optimization"""
    
    @staticmethod
    def get_embedding(text: str, model: str = EMBEDDING_MODEL, dimensions: int = EMBEDDING_DIMENSIONS) -> Optional[List[float]]:
        """Generate embedding for text with configurable model and dimensions"""
        try:
            response = client.embeddings.create(
                model=model,
                input=text,
                dimensions=dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    @staticmethod
    def cosine_similarity_score(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)
        return float(np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))
    
    @staticmethod
    def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate Euclidean distance between two embeddings"""
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)
        return float(np.linalg.norm(arr1 - arr2))
    
    @staticmethod
    def search(query_embedding: List[float], qa_database: List[Dict], 
               qa_embeddings: List[List[float]], top_k: int = 5, 
               similarity_metric: str = "cosine") -> List[Dict]:
        """Search for similar questions using vector similarity with configurable metric"""
        if not query_embedding or len(qa_embeddings) == 0:
            return []
        
        similarities = []
        for idx, db_embedding in enumerate(qa_embeddings):
            if similarity_metric == "cosine":
                score = VectorSearcher.cosine_similarity_score(query_embedding, db_embedding)
            elif similarity_metric == "euclidean":
                # Convert distance to similarity (smaller distance = higher similarity)
                score = 1.0 / (1.0 + VectorSearcher.euclidean_distance(query_embedding, db_embedding))
            else:
                # Default to cosine
                score = VectorSearcher.cosine_similarity_score(query_embedding, db_embedding)
            
            similarities.append({
                'index': idx,
                'score': score,
                'question': qa_database[idx]['question'],
                'answer': qa_database[idx]['answer'],
                'category': qa_database[idx].get('category', 'general'),
                'metadata': qa_database[idx].get('metadata', {})
            })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def batch_search(query_embeddings: List[List[float]], qa_database: List[Dict], 
                     qa_embeddings: List[List[float]], top_k: int = 5) -> List[List[Dict]]:
        """Batch search for multiple query embeddings"""
        results = []
        for query_embedding in query_embeddings:
            results.append(VectorSearcher.search(query_embedding, qa_database, qa_embeddings, top_k))
        return results