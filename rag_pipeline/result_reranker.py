from typing import List, Dict

class ResultReranker:
    """Reranks results from multiple search methods"""
    
    @staticmethod
    def combine_and_rerank(vector_results: List[Dict], keyword_results: List[Dict], 
                          vector_weight: float = 0.7) -> List[Dict]:
        """Combine and rerank results from both search methods"""
        keyword_weight = 1.0 - vector_weight
        combined = {}
        
        # Add vector search results
        for result in vector_results:
            idx = result['index']
            combined[idx] = {
                'index': idx,
                'question': result['question'],
                'answer': result['answer'],
                'category': result['category'],
                'metadata': result.get('metadata', {}),
                'vector_score': result['score'],
                'keyword_score': 0.0
            }
        
        # Add keyword search results
        for result in keyword_results:
            idx = result['index']
            if idx in combined:
                combined[idx]['keyword_score'] = result['score']
            else:
                combined[idx] = {
                    'index': idx,
                    'question': result['question'],
                    'answer': result['answer'],
                    'category': result['category'],
                    'metadata': result.get('metadata', {}),
                    'vector_score': 0.0,
                    'keyword_score': result['score']
                }
        
        # Calculate combined score
        for item in combined.values():
            item['final_score'] = (item['vector_score'] * vector_weight) + \
                                 (item['keyword_score'] * keyword_weight)
        
        # Sort by final score
        reranked = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
        
        return reranked[:5]