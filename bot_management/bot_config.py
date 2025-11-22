from datetime import datetime
from typing import List, Dict, Optional
import os

# Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

class BotConfig:
    """Configuration for each bot"""
    
    def __init__(self, bot_id: str, bot_name: str, system_prompt: str, 
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 escalation_message: Optional[str] = None,
                 language: str = "en"):
        self.bot_id = bot_id
        self.bot_name = bot_name
        self.system_prompt = system_prompt
        self.confidence_threshold = confidence_threshold
        self.language = language
        self.escalation_message = escalation_message or (
            "I don't have enough information to answer this accurately. "
            "Your question has been forwarded to our support team who will respond within 24 hours."
        )
        self.qa_database = []
        self.qa_embeddings = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.created_at = datetime.now().isoformat()
        self.query_count = 0
        self.escalation_count = 0