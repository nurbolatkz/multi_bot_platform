import time
import threading
from datetime import datetime
from typing import Dict, Any
from bot_management.bot_manager import BotManager

class HealthChecker:
    """Handles periodic health checks for all bots"""
    
    def __init__(self, bot_manager: BotManager, check_interval: int = 240):  # 4 minutes default
        self.bot_manager = bot_manager
        self.check_interval = check_interval
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.last_check_time: Dict[str, str] = {}
        self.is_running = False
        self.thread = None
    
    def start_health_checks(self):
        """Start background health checking"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self.thread.start()
            print("Health checker started")
    
    def stop_health_checks(self):
        """Stop background health checking"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Health checker stopped")
    
    def _health_check_loop(self):
        """Main health check loop"""
        while self.is_running:
            self._perform_health_checks()
            time.sleep(self.check_interval)
    
    def _perform_health_checks(self):
        """Perform health checks on all bots"""
        print(f"Performing health checks at {datetime.now().isoformat()}")
        
        for bot_id, bot_config in self.bot_manager.bots.items():
            try:
                # Simple health check - verify bot has QA data loaded
                is_healthy = len(bot_config.qa_database) > 0
                status = "healthy" if is_healthy else "degraded"
                
                self.health_status[bot_id] = {
                    "status": status,
                    "last_check": datetime.now().isoformat(),
                    "qa_count": len(bot_config.qa_database),
                    "query_count": bot_config.query_count,
                    "escalation_count": bot_config.escalation_count
                }
                
                self.last_check_time[bot_id] = datetime.now().isoformat()
                print(f"Bot {bot_id}: {status}")
                
            except Exception as e:
                self.health_status[bot_id] = {
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "error": str(e)
                }
                print(f"Bot {bot_id}: unhealthy - {e}")
    
    def get_bot_health(self, bot_id: str) -> Dict[str, Any]:
        """Get health status for a specific bot"""
        return self.health_status.get(bot_id, {
            "status": "unknown",
            "last_check": None
        })
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all bots"""
        return self.health_status
    
    def get_last_check_time(self, bot_id: str) -> str:
        """Get last check time for a specific bot"""
        return self.last_check_time.get(bot_id, "never")