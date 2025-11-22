from datetime import datetime

class ModeratorQueue:
    """Handles low-confidence questions"""
    
    @staticmethod
    def add_to_queue(bot_id: str, phone_number: str, question: str, 
                     confidence: float = 0.0) -> bool:
        """Add question to moderator queue"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Save to bot-specific file
            filename = f'moderator_queue_{bot_id}.txt'
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp}|{phone_number}|{confidence:.3f}|{question}\n")
            
            print(f"[{bot_id}] Added to moderator queue: {phone_number} - {question}")
            return True
        except Exception as e:
            print(f"Error adding to moderator queue: {e}")
            return False