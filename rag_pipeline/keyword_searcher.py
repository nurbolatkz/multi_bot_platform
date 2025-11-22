from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

class KeywordSearcher:
    """Handles keyword-based search using TF-IDF"""
    
    @staticmethod
    def search(query: str, qa_database: List[Dict], 
               tfidf_vectorizer, tfidf_matrix, top_k: int = 5) -> List[Dict]:
        """Search using TF-IDF keyword matching"""
        if tfidf_vectorizer is None or tfidf_matrix is None:
            return []
        
        try:
            query_vec = tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            
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