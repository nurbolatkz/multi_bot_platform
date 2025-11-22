from typing import List, Dict
import numpy as np

class ResultReranker:
    """Reranks results from multiple search methods with configurable weighting"""
    
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
    
    @staticmethod
    def reciprocal_rank_fusion(vector_results: List[Dict], keyword_results: List[Dict], 
                              k: int = 60) -> List[Dict]:
        """Combine results using reciprocal rank fusion"""
        fused_scores = {}
        
        # Process vector search results
        for rank, result in enumerate(vector_results):
            idx = result['index']
            if idx not in fused_scores:
                fused_scores[idx] = {
                    'result': result,
                    'vector_rank': rank + 1,
                    'keyword_rank': float('inf'),
                    'vector_score': result['vector_score'] if 'vector_score' in result else result['score']
                }
            else:
                fused_scores[idx]['vector_rank'] = rank + 1
                fused_scores[idx]['vector_score'] = result['vector_score'] if 'vector_score' in result else result['score']
        
        # Process keyword search results
        for rank, result in enumerate(keyword_results):
            idx = result['index']
            if idx not in fused_scores:
                fused_scores[idx] = {
                    'result': result,
                    'vector_rank': float('inf'),
                    'keyword_rank': rank + 1,
                    'keyword_score': result['keyword_score'] if 'keyword_score' in result else result['score']
                }
            else:
                fused_scores[idx]['keyword_rank'] = rank + 1
                fused_scores[idx]['keyword_score'] = result['keyword_score'] if 'keyword_score' in result else result['score']
        
        # Calculate reciprocal rank fusion scores
        for item in fused_scores.values():
            vector_rrf = 1.0 / (k + item['vector_rank']) if item['vector_rank'] != float('inf') else 0
            keyword_rrf = 1.0 / (k + item['keyword_rank']) if item['keyword_rank'] != float('inf') else 0
            item['final_score'] = vector_rrf + keyword_rrf
        
        # Sort by final score and prepare results
        sorted_items = sorted(fused_scores.values(), key=lambda x: x['final_score'], reverse=True)
        reranked = []
        
        for item in sorted_items[:5]:  # Top 5 results
            result = item['result'].copy()
            result['final_score'] = item['final_score']
            result['vector_rank'] = item['vector_rank']
            result['keyword_rank'] = item['keyword_rank']
            reranked.append(result)
        
        return reranked
    
    @staticmethod
    def weighted_reciprocal_rank_fusion(vector_results: List[Dict], keyword_results: List[Dict], 
                                       vector_weight: float = 0.7, k: int = 60) -> List[Dict]:
        """Combine results using weighted reciprocal rank fusion"""
        fused_scores = {}
        
        # Process vector search results
        for rank, result in enumerate(vector_results):
            idx = result['index']
            if idx not in fused_scores:
                fused_scores[idx] = {
                    'result': result,
                    'vector_rank': rank + 1,
                    'keyword_rank': float('inf'),
                    'vector_score': result['vector_score'] if 'vector_score' in result else result['score']
                }
            else:
                fused_scores[idx]['vector_rank'] = rank + 1
                fused_scores[idx]['vector_score'] = result['vector_score'] if 'vector_score' in result else result['score']
        
        # Process keyword search results
        for rank, result in enumerate(keyword_results):
            idx = result['index']
            if idx not in fused_scores:
                fused_scores[idx] = {
                    'result': result,
                    'vector_rank': float('inf'),
                    'keyword_rank': rank + 1,
                    'keyword_score': result['keyword_score'] if 'keyword_score' in result else result['score']
                }
            else:
                fused_scores[idx]['keyword_rank'] = rank + 1
                fused_scores[idx]['keyword_score'] = result['keyword_score'] if 'keyword_score' in result else result['score']
        
        # Calculate weighted reciprocal rank fusion scores
        for item in fused_scores.values():
            vector_rrf = 1.0 / (k + item['vector_rank']) if item['vector_rank'] != float('inf') else 0
            keyword_rrf = 1.0 / (k + item['keyword_rank']) if item['keyword_rank'] != float('inf') else 0
            item['final_score'] = (vector_rrf * vector_weight) + (keyword_rrf * (1.0 - vector_weight))
        
        # Sort by final score and prepare results
        sorted_items = sorted(fused_scores.values(), key=lambda x: x['final_score'], reverse=True)
        reranked = []
        
        for item in sorted_items[:5]:  # Top 5 results
            result = item['result'].copy()
            result['final_score'] = item['final_score']
            result['vector_rank'] = item['vector_rank']
            result['keyword_rank'] = item['keyword_rank']
            reranked.append(result)
        
        return reranked