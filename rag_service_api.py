from flask import Flask, request, jsonify
from openai import OpenAI
import os
import re
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Multi-bot storage: {bot_id: {config, qa_database, embeddings, etc.}}
bots = {}

class BotConfig:
    """Configuration for each bot"""
    
    def __init__(self, bot_id: str, bot_name: str, system_prompt: str, 
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 escalation_message: str = None,
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

class QueryPreprocessor:
    """Handles query cleaning and normalization"""
    
    @staticmethod
    def clean(query: str, custom_abbreviations: Dict[str, str] = None) -> str:
        """Clean and normalize the query with optional custom abbreviations"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Default abbreviations
        abbreviations = {
            'uni': 'university',
            'dept': 'department',
            'info': 'information',
            'admin': 'administration',
            'msg': 'message',
            'pls': 'please',
            'thx': 'thanks'
        }
        
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
        
        return query

class VectorSearcher:
    """Handles vector similarity search"""
    
    @staticmethod
    def get_embedding(text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    @staticmethod
    def cosine_similarity_score(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)
        return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    
    @staticmethod
    def search(query_embedding: List[float], qa_database: List[Dict], 
               qa_embeddings: List[List[float]], top_k: int = 5) -> List[Dict]:
        """Search for similar questions using vector similarity"""
        if not query_embedding or len(qa_embeddings) == 0:
            return []
        
        similarities = []
        for idx, db_embedding in enumerate(qa_embeddings):
            score = VectorSearcher.cosine_similarity_score(query_embedding, db_embedding)
            similarities.append({
                'index': idx,
                'score': score,
                'question': qa_database[idx]['question'],
                'answer': qa_database[idx]['answer'],
                'category': qa_database[idx].get('category', 'general'),
                'metadata': qa_database[idx].get('metadata', {})
            })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]

class KeywordSearcher:
    """Handles keyword-based search using TF-IDF"""
    
    @staticmethod
    def search(query: str, qa_database: List[Dict], 
               tfidf_vectorizer, tfidf_matrix, top_k: int = 5) -> List[Dict]:
        """Search using TF-IDF keyword matching"""
        if tfidf_vectorizer is None or tfidf_matrix is None:
            return []
        
        try:
            query_vec = tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'index': int(idx),
                    'score': float(similarities[idx]),
                    'question': qa_database[idx]['question'],
                    'answer': qa_database[idx]['answer'],
                    'category': qa_database[idx].get('category', 'general'),
                    'metadata': qa_database[idx].get('metadata', {})
                })
            
            return results
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

class ResultReranker:
    """Reranks results from multiple search methods"""
    
    @staticmethod
    def combine_and_rerank(vector_results: List[Dict], keyword_results: List[Dict], 
                          vector_weight: float = 0.7) -> List[Dict]:
        """Combine and rerank results from both search methods"""
        keyword_weight = 1.0 - vector_weight
        combined = {}
        
        # Add vector search results
        for result in vector_results:
            idx = result['index']
            combined[idx] = {
                'index': idx,
                'question': result['question'],
                'answer': result['answer'],
                'category': result['category'],
                'metadata': result.get('metadata', {}),
                'vector_score': result['score'],
                'keyword_score': 0.0
            }
        
        # Add keyword search results
        for result in keyword_results:
            idx = result['index']
            if idx in combined:
                combined[idx]['keyword_score'] = result['score']
            else:
                combined[idx] = {
                    'index': idx,
                    'question': result['question'],
                    'answer': result['answer'],
                    'category': result['category'],
                    'metadata': result.get('metadata', {}),
                    'vector_score': 0.0,
                    'keyword_score': result['score']
                }
        
        # Calculate combined score
        for item in combined.values():
            item['final_score'] = (item['vector_score'] * vector_weight) + \
                                 (item['keyword_score'] * keyword_weight)
        
        # Sort by final score
        reranked = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
        
        return reranked[:5]

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
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error. Please try again or contact support."

# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'total_bots': len(bots),
        'bots': [{'id': bot_id, 'name': config.bot_name, 'qa_count': len(config.qa_database)} 
                 for bot_id, config in bots.items()]
    })

@app.route('/bot/create', methods=['POST'])
def create_bot():
    """Create a new bot instance"""
    try:
        data = request.json
        bot_id = data.get('bot_id')
        bot_name = data.get('bot_name')
        system_prompt = data.get('system_prompt')
        confidence_threshold = data.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        escalation_message = data.get('escalation_message')
        language = data.get('language', 'en')
        
        if not bot_id or not bot_name:
            return jsonify({'error': 'bot_id and bot_name are required'}), 400
        
        if bot_id in bots:
            return jsonify({'error': f'Bot {bot_id} already exists'}), 400
        
        # Default system prompt if not provided
        if not system_prompt:
            system_prompt = f"""You are a helpful assistant for {bot_name}.

Your role is to answer questions accurately using ONLY the information provided in the CONTEXT.

INSTRUCTIONS:
1. Answer using ONLY information from the CONTEXT
2. Be friendly, concise, and clear
3. Use bullet points for multi-step processes
4. Do not make up information
5. Keep responses under 250 words
6. Use a warm, supportive tone
7. Respond in the same language as the question"""
        
        # Create bot config
        bots[bot_id] = BotConfig(
            bot_id=bot_id,
            bot_name=bot_name,
            system_prompt=system_prompt,
            confidence_threshold=confidence_threshold,
            escalation_message=escalation_message,
            language=language
        )
        
        return jsonify({
            'status': 'success',
            'bot_id': bot_id,
            'bot_name': bot_name,
            'message': f'Bot {bot_name} created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bot/<bot_id>/load_qa', methods=['POST'])
def load_qa_database(bot_id):
    """Load Q&A database for specific bot"""
    try:
        if bot_id not in bots:
            return jsonify({'error': f'Bot {bot_id} not found'}), 404
        
        bot_config = bots[bot_id]
        data = request.json
        qa_data = data.get('qa_pairs', [])
        
        if not qa_data:
            return jsonify({'error': 'No Q&A pairs provided'}), 400
        
        # Load Q&A database
        bot_config.qa_database = qa_data
        
        # Generate embeddings for all questions
        print(f"[{bot_id}] Generating embeddings...")
        bot_config.qa_embeddings = []
        questions = [item['question'] for item in bot_config.qa_database]
        
        for question in questions:
            embedding = VectorSearcher.get_embedding(question)
            bot_config.qa_embeddings.append(embedding)
        
        # Create TF-IDF matrix for keyword search
        print(f"[{bot_id}] Creating TF-IDF matrix...")
        bot_config.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        bot_config.tfidf_matrix = bot_config.tfidf_vectorizer.fit_transform(questions)
        
        return jsonify({
            'status': 'success',
            'bot_id': bot_id,
            'loaded': len(bot_config.qa_database),
            'message': f'Q&A database loaded successfully for {bot_config.bot_name}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bot/<bot_id>/query', methods=['POST'])
def process_query(bot_id):
    """Process query for specific bot"""
    try:
        if bot_id not in bots:
            return jsonify({'error': f'Bot {bot_id} not found'}), 404
        
        bot_config = bots[bot_id]
        data = request.json
        phone_number = data.get('phone_number')
        question = data.get('question')
        custom_abbreviations = data.get('abbreviations', {})
        
        if not phone_number or not question:
            return jsonify({'error': 'Missing phone_number or question'}), 400
        
        bot_config.query_count += 1
        
        # Step 1: Preprocess query
        cleaned_query = QueryPreprocessor.clean(question, custom_abbreviations)
        print(f"[{bot_id}] Original: {question}")
        print(f"[{bot_id}] Cleaned: {cleaned_query}")
        
        # Step 2: Generate embedding
        query_embedding = VectorSearcher.get_embedding(cleaned_query)
        if not query_embedding:
            return jsonify({'error': 'Failed to generate embedding'}), 500
        
        # Step 3: Vector Search
        vector_results = VectorSearcher.search(
            query_embedding, 
            bot_config.qa_database, 
            bot_config.qa_embeddings, 
            top_k=5
        )
        
        # Step 4: Keyword Search
        keyword_results = KeywordSearcher.search(
            cleaned_query,
            bot_config.qa_database,
            bot_config.tfidf_vectorizer,
            bot_config.tfidf_matrix,
            top_k=5
        )
        
        # Step 5: Rerank Results
        reranked_results = ResultReranker.combine_and_rerank(vector_results, keyword_results)
        
        if not reranked_results:
            # No results found - escalate
            ModeratorQueue.add_to_queue(bot_id, phone_number, question, 0.0)
            bot_config.escalation_count += 1
            return jsonify({
                'bot_id': bot_id,
                'phone_number': phone_number,
                'answer': bot_config.escalation_message,
                'confidence': 0.0,
                'escalated': True
            })
        
        # Step 6: Check Confidence Score
        top_score = reranked_results[0]['final_score']
        print(f"[{bot_id}] Top confidence score: {top_score}")
        
        if top_score < bot_config.confidence_threshold:
            # Low confidence - escalate to moderator
            ModeratorQueue.add_to_queue(bot_id, phone_number, question, top_score)
            bot_config.escalation_count += 1
            return jsonify({
                'bot_id': bot_id,
                'phone_number': phone_number,
                'answer': bot_config.escalation_message,
                'confidence': top_score,
                'escalated': True,
                'matched_question': reranked_results[0]['question']
            })
        
        # Step 7: High confidence - Generate answer
        answer = AnswerGenerator.generate(
            question, 
            reranked_results,
            bot_config.system_prompt,
            bot_config.language
        )
        
        return jsonify({
            'bot_id': bot_id,
            'phone_number': phone_number,
            'answer': answer,
            'confidence': top_score,
            'escalated': False,
            'matched_question': reranked_results[0]['question'],
            'category': reranked_results[0]['category']
        })
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/bot/<bot_id>/config', methods=['GET', 'PUT'])
def bot_config_endpoint(bot_id):
    """Get or update bot configuration"""
    if bot_id not in bots:
        return jsonify({'error': f'Bot {bot_id} not found'}), 404
    
    bot_config = bots[bot_id]
    
    if request.method == 'GET':
        return jsonify({
            'bot_id': bot_config.bot_id,
            'bot_name': bot_config.bot_name,
            'confidence_threshold': bot_config.confidence_threshold,
            'language': bot_config.language,
            'qa_count': len(bot_config.qa_database),
            'query_count': bot_config.query_count,
            'escalation_count': bot_config.escalation_count,
            'created_at': bot_config.created_at
        })
    
    elif request.method == 'PUT':
        data = request.json
        
        if 'confidence_threshold' in data:
            threshold = data['confidence_threshold']
            if 0 <= threshold <= 1:
                bot_config.confidence_threshold = threshold
        
        if 'system_prompt' in data:
            bot_config.system_prompt = data['system_prompt']
        
        if 'escalation_message' in data:
            bot_config.escalation_message = data['escalation_message']
        
        if 'language' in data:
            bot_config.language = data['language']
        
        return jsonify({
            'status': 'success',
            'message': f'Bot {bot_id} updated successfully'
        })

@app.route('/bot/<bot_id>/delete', methods=['DELETE'])
def delete_bot(bot_id):
    """Delete a bot instance"""
    if bot_id not in bots:
        return jsonify({'error': f'Bot {bot_id} not found'}), 404
    
    del bots[bot_id]
    return jsonify({
        'status': 'success',
        'message': f'Bot {bot_id} deleted successfully'
    })

@app.route('/bot/<bot_id>/stats', methods=['GET'])
def bot_stats(bot_id):
    """Get bot statistics"""
    if bot_id not in bots:
        return jsonify({'error': f'Bot {bot_id} not found'}), 404
    
    bot_config = bots[bot_id]
    
    escalation_rate = 0
    if bot_config.query_count > 0:
        escalation_rate = (bot_config.escalation_count / bot_config.query_count) * 100
    
    return jsonify({
        'bot_id': bot_id,
        'bot_name': bot_config.bot_name,
        'total_queries': bot_config.query_count,
        'total_escalations': bot_config.escalation_count,
        'escalation_rate': f"{escalation_rate:.2f}%",
        'qa_database_size': len(bot_config.qa_database),
        'confidence_threshold': bot_config.confidence_threshold
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)