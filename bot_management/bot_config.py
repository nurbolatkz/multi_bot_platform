from datetime import datetime
from typing import List, Dict, Optional, Any
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3

class BotConfig:
    """Configuration for each bot with complete data isolation"""
    
    def __init__(self, bot_id: str, bot_name: str, system_prompt: str, 
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 escalation_message: Optional[str] = None,
                 language: str = "en"):
        # Core bot identification
        self.bot_id = bot_id
        self.bot_name = bot_name
        
        # Configuration parameters
        self.system_prompt = system_prompt
        self.confidence_threshold = confidence_threshold
        self.language = language
        self.escalation_message = escalation_message or (
            "I don't have enough information to answer this accurately. "
            "Your question has been forwarded to our support team who will respond within 24 hours."
        )
        
        # RAG pipeline configuration
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.chat_model = DEFAULT_CHAT_MODEL
        self.vector_weight = DEFAULT_VECTOR_WEIGHT
        self.keyword_weight = DEFAULT_KEYWORD_WEIGHT
        
        # Data storage - isolated per bot
        self.qa_database: List[Dict[str, Any]] = []
        self.qa_embeddings: List[List[float]] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        # Custom configurations
        self.custom_parameters: Dict[str, Any] = {}
        
        # Metadata
        self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        self.query_count = 0
        self.escalation_count = 0
        self.successful_queries = 0
        
        # Performance tracking
        self.avg_response_time = 0.0
        self.total_response_time = 0.0
    
    def update_config(self, **kwargs):
        """Update bot configuration parameters"""
        allowed_fields = {
            'system_prompt', 'confidence_threshold', 'language', 'escalation_message',
            'embedding_model', 'chat_model', 'vector_weight', 'keyword_weight'
        }
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                setattr(self, key, value)
            else:
                self.custom_parameters[key] = value
        
        self.last_updated = datetime.now().isoformat()
    
    def add_qa_pair(self, question: str, answer: str, category: str = "general", metadata: Optional[Dict] = None):
        """Add a single QA pair to the bot's database"""
        qa_pair = {
            'question': question,
            'answer': answer,
            'category': category,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        self.qa_database.append(qa_pair)
    
    def add_qa_pairs(self, qa_pairs: List[Dict[str, Any]]):
        """Add multiple QA pairs to the bot's database"""
        for pair in qa_pairs:
            self.add_qa_pair(
                pair.get('question', ''),
                pair.get('answer', ''),
                pair.get('category', 'general'),
                pair.get('metadata')
            )
    
    def clear_qa_database(self):
        """Clear the bot's QA database and associated embeddings"""
        self.qa_database = []
        self.qa_embeddings = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.query_count = 0
        self.escalation_count = 0
        self.successful_queries = 0
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the bot's configuration"""
        return {
            'bot_id': self.bot_id,
            'bot_name': self.bot_name,
            'language': self.language,
            'confidence_threshold': self.confidence_threshold,
            'embedding_model': self.embedding_model,
            'chat_model': self.chat_model,
            'vector_weight': self.vector_weight,
            'keyword_weight': self.keyword_weight,
            'qa_database_size': len(self.qa_database),
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }
    
    def update_performance_metrics(self, response_time: float, successful: bool = True):
        """Update performance tracking metrics"""
        self.query_count += 1
        if successful:
            self.successful_queries += 1
            
        # Update average response time
        self.total_response_time += response_time
        self.avg_response_time = self.total_response_time / self.query_count