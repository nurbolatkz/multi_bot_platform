import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
import os

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class VectorSearcher:
    """Handles vector similarity search"""
    
    @staticmethod
    def get_embedding(text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
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
        return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    
    @staticmethod
    def search(query_embedding: List[float], qa_database: List[Dict], 
               qa_embeddings: List[List[float]], top_k: int = 5) -> List[Dict]:
        """Search for similar questions using vector similarity"""
        if not query_embedding or len(qa_embeddings) == 0:
            return []
        
        similarities = []
        for idx, db_embedding in enumerate(qa_embeddings):
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