import re
from typing import Dict, Optional, List, Tuple
import langdetect

class QueryPreprocessor:
    """Handles query cleaning, normalization, and language detection"""
    
    # Extended abbreviation dictionaries by category
    ABBREVIATIONS = {
        'general': {
            'uni': 'university',
            'dept': 'department',
            'info': 'information',
            'admin': 'administration',
            'msg': 'message',
            'pls': 'please',
            'thx': 'thanks',
            'fyi': 'for your information',
            'asap': 'as soon as possible',
            'faq': 'frequently asked questions',
            'idk': 'i do not know',
            'tbh': 'to be honest',
            'omg': 'oh my god',
            'lol': 'laugh out loud',
            'brb': 'be right back',
            'ttyl': 'talk to you later'
        },
        'academic': {
            'prof': 'professor',
            'assoc': 'associate',
            'asst': 'assistant',
            'phd': 'doctor of philosophy',
            'bsc': 'bachelor of science',
            'msc': 'master of science',
            'eng': 'engineering',
            'comp': 'computer',
            'sci': 'science',
            'math': 'mathematics',
            'phys': 'physics',
            'chem': 'chemistry',
            'bio': 'biology'
        },
        'business': {
            'corp': 'corporation',
            'inc': 'incorporated',
            'llc': 'limited liability company',
            'ceo': 'chief executive officer',
            'cto': 'chief technology officer',
            'cfo': 'chief financial officer',
            'hr': 'human resources',
            'it': 'information technology',
            'qa': 'quality assurance',
            'kpi': 'key performance indicator',
            'roi': 'return on investment',
            'crm': 'customer relationship management'
        }
    }
    
    # Language-specific preprocessing rules
    LANGUAGE_RULES = {
        'en': {
            'contractions': {
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would",
                "'m": " am"
            }
        },
        'es': {
            'contractions': {
                "don't": "do not"
            }
        }
    }
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the language of the text"""
        try:
            return langdetect.detect(text)
        except:
            return 'en'  # Default to English
    
    @staticmethod
    def expand_contractions(text: str, language: str = 'en') -> str:
        """Expand contractions based on language"""
        if language in QueryPreprocessor.LANGUAGE_RULES:
            contractions = QueryPreprocessor.LANGUAGE_RULES[language].get('contractions', {})
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
        return text
    
    @staticmethod
    def clean(query: str, custom_abbreviations: Optional[Dict[str, str]] = None, 
              language: str = 'auto', bot_category: str = 'general') -> Tuple[str, str]:
        """Clean and normalize the query with language detection and custom rules"""
        original_query = query
        
        # Detect language if set to auto
        if language == 'auto':
            detected_language = QueryPreprocessor.detect_language(query)
        else:
            detected_language = language
        
        # Convert to lowercase
        query = query.lower().strip()
        
        # Expand contractions based on language
        query = QueryPreprocessor.expand_contractions(query, detected_language)
        
        # Build abbreviation dictionary
        abbreviations = QueryPreprocessor.ABBREVIATIONS.get('general', {}).copy()
        
        # Add category-specific abbreviations
        if bot_category in QueryPreprocessor.ABBREVIATIONS:
            abbreviations.update(QueryPreprocessor.ABBREVIATIONS[bot_category])
        
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
        
        return query, detected_language
    
    @staticmethod
    def add_custom_abbreviations(bot_id: str, abbreviations: Dict[str, str]):
        """Add custom abbreviations for a specific bot"""
        # This would typically store in a database or config file
        # For now, we'll just demonstrate the concept
        if not hasattr(QueryPreprocessor, '_custom_abbreviations'):
            QueryPreprocessor._custom_abbreviations = {}
        
        if bot_id not in QueryPreprocessor._custom_abbreviations:
            QueryPreprocessor._custom_abbreviations[bot_id] = {}
        
        QueryPreprocessor._custom_abbreviations[bot_id].update(abbreviations)
    
    @staticmethod
    def get_custom_abbreviations(bot_id: str) -> Dict[str, str]:
        """Get custom abbreviations for a specific bot"""
        if hasattr(QueryPreprocessor, '_custom_abbreviations') and bot_id in QueryPreprocessor._custom_abbreviations:
            return QueryPreprocessor._custom_abbreviations[bot_id]
        return {}