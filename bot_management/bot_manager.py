from typing import Dict, List, Optional
from bot_management.bot_config import BotConfig
from datetime import datetime

class BotManager:
    """Manages multiple bot instances"""
    
    def __init__(self):
        self.bots: Dict[str, BotConfig] = {}
        self.creation_timestamps: Dict[str, str] = {}
    
    def create_bot(self, bot_id: str, bot_name: str, system_prompt: str, 
                   confidence_threshold: float = 0.75,
                   escalation_message: Optional[str] = None,
                   language: str = "en",
                   category: str = "general") -> dict:
        """Create a new bot instance"""
        if bot_id in self.bots:
            return {
                "success": False,
                "error": f"Bot with ID '{bot_id}' already exists"
            }
        
        try:
            self.bots[bot_id] = BotConfig(
                bot_id=bot_id,
                bot_name=bot_name,
                system_prompt=system_prompt,
                confidence_threshold=confidence_threshold,
                escalation_message=escalation_message,
                language=language,
                category=category
            )
            self.creation_timestamps[bot_id] = datetime.now().isoformat()
            
            return {
                "success": True,
                "bot_id": bot_id,
                "message": f"Bot '{bot_name}' created successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create bot: {str(e)}"
            }
    
    def get_bot(self, bot_id: str) -> Optional[BotConfig]:
        """Get bot by ID"""
        return self.bots.get(bot_id)
    
    def update_bot(self, bot_id: str, **kwargs) -> dict:
        """Update bot configuration"""
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "success": False,
                "error": f"Bot with ID '{bot_id}' not found"
            }
        
        try:
            # Update allowed fields
            if 'bot_name' in kwargs:
                bot.bot_name = kwargs['bot_name']
            if 'system_prompt' in kwargs:
                bot.system_prompt = kwargs['system_prompt']
            if 'confidence_threshold' in kwargs:
                threshold = kwargs['confidence_threshold']
                if 0 <= threshold <= 1:
                    bot.confidence_threshold = threshold
                else:
                    return {
                        "success": False,
                        "error": "Confidence threshold must be between 0 and 1"
                    }
            if 'escalation_message' in kwargs:
                bot.escalation_message = kwargs['escalation_message']
            if 'language' in kwargs:
                bot.language = kwargs['language']
            if 'category' in kwargs:
                bot.category = kwargs['category']
            
            return {
                "success": True,
                "message": f"Bot '{bot_id}' updated successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to update bot: {str(e)}"
            }
    
    def delete_bot(self, bot_id: str) -> dict:
        """Delete a bot instance"""
        if bot_id not in self.bots:
            return {
                "success": False,
                "error": f"Bot with ID '{bot_id}' not found"
            }
        
        try:
            bot_name = self.bots[bot_id].bot_name
            del self.bots[bot_id]
            if bot_id in self.creation_timestamps:
                del self.creation_timestamps[bot_id]
            
            return {
                "success": True,
                "message": f"Bot '{bot_name}' deleted successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete bot: {str(e)}"
            }
    
    def list_bots(self) -> List[Dict]:
        """List all bots with detailed information"""
        return [
            {
                'id': bot_id,
                'name': config.bot_name,
                'category': config.category,
                'qa_count': len(config.qa_database),
                'created_at': self.creation_timestamps.get(bot_id, "Unknown"),
                'query_count': config.query_count,
                'escalation_count': config.escalation_count
            } 
            for bot_id, config in self.bots.items()
        ]
    
    def bot_exists(self, bot_id: str) -> bool:
        """Check if bot exists"""
        return bot_id in self.bots
    
    def get_bot_stats(self, bot_id: str) -> dict:
        """Get detailed statistics for a bot"""
        bot = self.get_bot(bot_id)
        if not bot:
            return {
                "success": False,
                "error": f"Bot with ID '{bot_id}' not found"
            }
        
        escalation_rate = 0
        if bot.query_count > 0:
            escalation_rate = (bot.escalation_count / bot.query_count) * 100
        
        return {
            "success": True,
            "bot_id": bot_id,
            "bot_name": bot.bot_name,
            "category": bot.category,
            "created_at": self.creation_timestamps.get(bot_id, "Unknown"),
            "total_queries": bot.query_count,
            "total_escalations": bot.escalation_count,
            "escalation_rate": f"{escalation_rate:.2f}%",
            "qa_database_size": len(bot.qa_database),
            "confidence_threshold": bot.confidence_threshold,
            "language": bot.language
        }
    
    def get_all_bot_stats(self) -> List[Dict]:
        """Get statistics for all bots"""
        return [self.get_bot_stats(bot_id) for bot_id in self.bots.keys()]