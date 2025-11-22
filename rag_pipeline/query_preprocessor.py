import re
from typing import Dict, Optional

class QueryPreprocessor:
    """Handles query cleaning and normalization"""
    
    @staticmethod
    def clean(query: str, custom_abbreviations: Optional[Dict[str, str]] = None) -> str:
        """Clean and normalize the query with optional custom abbreviations"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Default abbreviations
        abbreviations = {
            'uni': 'university',
            'dept': 'department',
            'info': 'information',
            'admin': 'administration',
            'msg': 'message',
            'pls': 'please',
            'thx': 'thanks'
        }
        
        # Merge with custom abbreviations if provided
        if custom_abbreviations:
            abbreviations.update(custom_abbreviations)
        
        # Replace abbreviations
        words = query.split()
        words = [abbreviations.get(word, word) for word in words]
        query = ' '.join(words)
        
        # Remove special characters but keep spaces and question marks
        query = re.sub(r'[^\w\s?]', ' ', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query