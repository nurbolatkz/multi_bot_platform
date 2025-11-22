from openai import OpenAI
from typing import List, Dict, Optional, Tuple
import os

# Configuration
CHAT_MODEL = "gpt-4o-mini"
CONFIDENCE_MODEL = "gpt-4o-mini"  # Model for confidence scoring

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class AnswerGenerator:
    """Generates final answer using ChatGPT with enhanced capabilities"""
    
    @staticmethod
    def generate(question: str, context_results: List[Dict], 
                 system_prompt: str, language: str = "en",
                 temperature: float = 0.3, max_tokens: int = 500,
                 chat_model: str = CHAT_MODEL) -> Tuple[str, Dict]:
        """Generate answer using GPT with retrieved context and enhanced features"""
        
        # Format context with relevance scores and categories
        context = ""
        context_metadata = []
        for i, result in enumerate(context_results[:5], 1):  # Increase context to 5
            context += f"Context {i} (Score: {result.get('final_score', result.get('score', 0)):.3f}):\n"
            context += f"Q: {result['question']}\nA: {result['answer']}\n"
            if 'category' in result:
                context += f"Category: {result['category']}\n"
            context += "\n"
            context_metadata.append({
                'index': i,
                'score': result.get('final_score', result.get('score', 0)),
                'category': result.get('category', 'general')
            })
        
        user_prompt = f"""CONTEXT:
{context}
QUESTION:
{question}

Please provide a helpful answer based on the context above. Respond in {language}.
Instructions:
1. Use ONLY information from the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Be concise but thorough
4. Use the same language as the question ({language})
5. Format your response clearly with bullet points or numbered lists when appropriate"""

        try:
            response = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            answer = content if content is not None else "Sorry, I couldn't generate a response. Please try again."
            
            # Generate metadata about the response
            response_metadata = {
                'model': chat_model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'context_used': len(context_results[:5]),
                'context_metadata': context_metadata
            }
            
            return answer, response_metadata
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            error_response = "Sorry, I encountered an error. Please try again or contact support."
            return error_response, {'error': str(e)}
    
    @staticmethod
    def generate_with_style(question: str, context_results: List[Dict], 
                           system_prompt: str, language: str = "en",
                           response_style: str = "concise") -> Tuple[str, Dict]:
        """Generate answer with specific style (concise, detailed, friendly, professional)"""
        
        style_prompts = {
            "concise": "Be brief and to the point. Use bullet points for lists.",
            "detailed": "Provide comprehensive information with examples where relevant.",
            "friendly": "Use a warm, conversational tone. Be helpful and encouraging.",
            "professional": "Use formal language and professional terminology."
        }
        
        style_instruction = style_prompts.get(response_style, style_prompts["concise"])
        
        # Format context
        context = ""
        for i, result in enumerate(context_results[:5], 1):
            context += f"Context {i} (Score: {result.get('final_score', result.get('score', 0)):.3f}):\n"
            context += f"Q: {result['question']}\nA: {result['answer']}\n\n"
        
        user_prompt = f"""CONTEXT:
{context}
QUESTION:
{question}

Please provide a helpful answer based on the context above. Respond in {language}.
Style: {style_instruction}"""

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
            answer = content if content is not None else "Sorry, I couldn't generate a response. Please try again."
            
            response_metadata = {
                'style': response_style,
                'model': CHAT_MODEL,
                'context_used': len(context_results[:5])
            }
            
            return answer, response_metadata
            
        except Exception as e:
            print(f"Error generating styled answer: {e}")
            error_response = "Sorry, I encountered an error. Please try again or contact support."
            return error_response, {'error': str(e)}
    
    @staticmethod
    def calculate_confidence(question: str, context_results: List[Dict], 
                           generated_answer: str) -> Tuple[float, str]:
        """Calculate confidence score for the generated answer"""
        
        if not context_results:
            return 0.0, "No context provided"
        
        # Get the top context score
        top_score = context_results[0].get('final_score', context_results[0].get('score', 0))
        
        # If score is very low, confidence is low
        if top_score < 0.3:
            return 0.3, "Low context relevance"
        
        # For higher scores, we can be more confident
        # Clamp between 0.3 and 0.95 (never 100% confidence)
        confidence = min(0.95, max(0.3, top_score))
        
        return confidence, "Based on context relevance score"
    
    @staticmethod
    def generate_with_confidence(question: str, context_results: List[Dict], 
                                system_prompt: str, language: str = "en") -> Tuple[str, float, str, Dict]:
        """Generate answer and calculate confidence score"""
        
        # Generate the answer
        answer, metadata = AnswerGenerator.generate(question, context_results, system_prompt, language)
        
        # Calculate confidence
        confidence, confidence_reason = AnswerGenerator.calculate_confidence(question, context_results, answer)
        
        return answer, confidence, confidence_reason, metadata
    
    @staticmethod
    def generate_conditional_answer(question: str, context_results: List[Dict], 
                                   system_prompt: str, language: str = "en") -> Tuple[str, bool, Dict]:
        """Generate answer and indicate if context was sufficient"""
        
        # Check if we have good context
        if not context_results:
            insufficient_response = "I don't have enough information to answer your question accurately. Could you please provide more details or rephrase your question?"
            return insufficient_response, False, {'reason': 'No context available'}
        
        # Check the top score
        top_score = context_results[0].get('final_score', context_results[0].get('score', 0))
        if top_score < 0.4:  # Threshold for sufficient context
            insufficient_response = "I couldn't find relevant information to answer your question. Would you like me to help with something else?"
            return insufficient_response, False, {'reason': 'Low context relevance', 'top_score': top_score}
        
        # Generate answer with good context
        answer, metadata = AnswerGenerator.generate(question, context_results, system_prompt, language)
        metadata['top_score'] = top_score
        metadata['context_count'] = len(context_results)
        
        return answer, True, metadata