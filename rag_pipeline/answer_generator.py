from openai import OpenAI
from typing import List, Dict, Optional
import os

# Configuration
CHAT_MODEL = "gpt-4o-mini"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class AnswerGenerator:
    """Generates final answer using ChatGPT"""
    
    @staticmethod
    def generate(question: str, context_results: List[Dict], 
                 system_prompt: str, language: str = "en") -> str:
        """Generate answer using GPT with retrieved context"""
        
        # Format context
        context = ""
        for i, result in enumerate(context_results[:3], 1):
            context += f"Context {i}:\nQ: {result['question']}\nA: {result['answer']}\n\n"
        
        user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

Please provide a helpful answer based on the context above. Respond in {language}."""

        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            return content if content is not None else "Sorry, I couldn't generate a response. Please try again."
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error. Please try again or contact support."