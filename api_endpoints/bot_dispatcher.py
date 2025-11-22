from flask import jsonify
from typing import Dict, Any, Optional
from bot_management.bot_manager import BotManager
from bot_management.bot_config import BotConfig

class BotDispatcher:
    """Centralized bot dispatcher for routing requests to appropriate bots"""
    
    def __init__(self, bot_manager: BotManager):
        self.bot_manager = bot_manager
    
    def get_bot_config(self, bot_id: str) -> Optional[BotConfig]:
        """Get bot configuration by ID"""
        return self.bot_manager.get_bot(bot_id)
    
    def validate_bot_exists(self, bot_id: str) -> Dict[str, Any]:
        """Validate that a bot exists"""
        if not bot_id:
            return {
                "valid": False,
                "error": jsonify({'error': 'Bot ID is required'}), 
                "status_code": 400
            }
        
        if not self.bot_manager.bot_exists(bot_id):
            return {
                "valid": False,
                "error": jsonify({'error': f'Bot {bot_id} not found'}), 
                "status_code": 404
            }
        
        return {"valid": True}
    
    def validate_bot_ready(self, bot_id: str) -> Dict[str, Any]:
        """Validate that a bot is ready (has loaded QA data)"""
        validation = self.validate_bot_exists(bot_id)
        if not validation["valid"]:
            return validation
        
        bot_config = self.get_bot_config(bot_id)
        if not bot_config:
            return {
                "valid": False,
                "error": jsonify({'error': f'Bot {bot_id} not found'}), 
                "status_code": 404
            }
        
        # Check if bot has loaded QA data
        if len(bot_config.qa_database) == 0:
            return {
                "valid": False,
                "error": jsonify({'error': f'Bot {bot_id} has no QA data loaded'}), 
                "status_code": 400
            }
        
        return {"valid": True}
    
    def dispatch_to_bot(self, bot_id: str, action: str, **kwargs) -> Dict[str, Any]:
        """Dispatch action to specific bot"""
        # Validate bot exists
        validation = self.validate_bot_exists(bot_id)
        if not validation["valid"]:
            return validation
        
        # Get bot config
        bot_config = self.get_bot_config(bot_id)
        if not bot_config:
            return {
                "valid": False,
                "error": jsonify({'error': f'Bot {bot_id} not found'}), 
                "status_code": 404
            }
        
        # Dispatch based on action
        if action == "query":
            return self._handle_query(bot_config, **kwargs)
        elif action == "load_qa":
            return self._handle_load_qa(bot_config, **kwargs)
        elif action == "get_stats":
            return self._handle_get_stats(bot_config)
        else:
            return {
                "valid": False,
                "error": jsonify({'error': f'Unknown action: {action}'}), 
                "status_code": 400
            }
    
    def _handle_query(self, bot_config: BotConfig, **kwargs) -> Dict[str, Any]:
        """Handle query action for bot"""
        # This would contain the query processing logic
        return {
            "valid": True,
            "bot_config": bot_config
        }
    
    def _handle_load_qa(self, bot_config: BotConfig, **kwargs) -> Dict[str, Any]:
        """Handle load QA action for bot"""
        # This would contain the QA loading logic
        return {
            "valid": True,
            "bot_config": bot_config
        }
    
    def _handle_get_stats(self, bot_config: BotConfig) -> Dict[str, Any]:
        """Handle get stats action for bot"""
        # This would contain the stats retrieval logic
        return {
            "valid": True,
            "bot_config": bot_config
        }