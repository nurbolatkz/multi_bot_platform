from functools import wraps
from flask import jsonify, request
from typing import Callable, Any
from bot_management.bot_manager import BotManager

def validate_bot_exists(bot_manager: BotManager):
    """Middleware to validate bot exists before processing request"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            bot_id = kwargs.get('bot_id')
            if not bot_id:
                return jsonify({'error': 'Bot ID is required'}), 400
            
            if not bot_manager.bot_exists(bot_id):
                return jsonify({'error': f'Bot {bot_id} not found'}), 404
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_bot_ready(bot_manager: BotManager):
    """Middleware to validate bot is ready (has loaded QA data) before processing request"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            bot_id = kwargs.get('bot_id')
            if not bot_id:
                return jsonify({'error': 'Bot ID is required'}), 400
            
            bot_config = bot_manager.get_bot(bot_id)
            if not bot_config:
                return jsonify({'error': f'Bot {bot_id} not found'}), 404
            
            # Check if bot has loaded QA data
            if len(bot_config.qa_database) == 0:
                return jsonify({'error': f'Bot {bot_id} has no QA data loaded'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_json_data():
    """Middleware to ensure request contains JSON data"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.json:
                return jsonify({'error': 'JSON data required'}), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator