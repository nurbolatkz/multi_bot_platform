from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional
import numpy as np

class KeywordSearcher:
    """Handles keyword-based search using TF-IDF with enhanced optimization"""
    
    @staticmethod
    def search(query: str, qa_database: List[Dict], 
               tfidf_vectorizer: TfidfVectorizer, tfidf_matrix, 
               top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Search using TF-IDF keyword matching with minimum similarity threshold"""
        if tfidf_vectorizer is None or tfidf_matrix is None:
            return []
        
        try:
            query_vec = tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Apply minimum similarity threshold
            valid_indices = np.where(similarities >= min_similarity)[0]
            if len(valid_indices) == 0:
                return []
            
            # Get top k from valid indices
            valid_similarities = similarities[valid_indices]
            top_indices = valid_indices[np.argsort(valid_similarities)[-top_k:][::-1]]
            
            results = []
            for idx in top_indices:
                results.append({
                    'index': int(idx),
                    'score': float(similarities[idx]),
                    'question': qa_database[idx]['question'],
                    'answer': qa_database[idx]['answer'],
                    'category': qa_database[idx].get('category', 'general'),
                    'metadata': qa_database[idx].get('metadata', {})
                })
            
            return results
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
    
    @staticmethod
    def advanced_search(query: str, qa_database: List[Dict], 
                       tfidf_vectorizer: TfidfVectorizer, tfidf_matrix,
                       top_k: int = 5, min_similarity: float = 0.0,
                       boost_category: Optional[str] = None, 
                       category_boost_factor: float = 1.5) -> List[Dict]:
        """Advanced search with category boosting"""
        if tfidf_vectorizer is None or tfidf_matrix is None:
            return []
        
        try:
            query_vec = tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Apply minimum similarity threshold
            valid_indices = np.where(similarities >= min_similarity)[0]
            if len(valid_indices) == 0:
                return []
            
            # Apply category boosting if specified
            if boost_category:
                for idx in valid_indices:
                    if qa_database[idx].get('category') == boost_category:
                        similarities[idx] *= category_boost_factor
            
            # Get top k from valid indices
            valid_similarities = similarities[valid_indices]
            top_indices = valid_indices[np.argsort(valid_similarities)[-top_k:][::-1]]
            
            results = []
            for idx in top_indices:
                results.append({
                    'index': int(idx),
                    'score': float(similarities[idx]),
                    'question': qa_database[idx]['question'],
                    'answer': qa_database[idx]['answer'],
                    'category': qa_database[idx].get('category', 'general'),
                    'metadata': qa_database[idx].get('metadata', {}),
                    'boosted': qa_database[idx].get('category') == boost_category if boost_category else False
                })
            
            return results
        except Exception as e:
            print(f"Advanced keyword search error: {e}")
            return []