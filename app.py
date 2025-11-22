from flask import Flask, request, jsonify
from bot_management.bot_manager import BotManager
from bot_management.bot_config import DEFAULT_CONFIDENCE_THRESHOLD
from rag_pipeline.query_preprocessor import QueryPreprocessor
from rag_pipeline.vector_searcher import VectorSearcher
from rag_pipeline.keyword_searcher import KeywordSearcher
from rag_pipeline.result_reranker import ResultReranker
from rag_pipeline.answer_generator import AnswerGenerator
from utils.moderator_queue import ModeratorQueue
from utils.health_checker import HealthChecker
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time
from typing import Dict, Any, Optional, List

app = Flask(__name__)

# Initialize bot manager and health checker
bot_manager = BotManager()
health_checker = HealthChecker(bot_manager)

# Start health checks
health_checker.start_health_checks()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'total_bots': len(bot_manager.bots),
        'bots': bot_manager.list_bots()
    })

@app.route('/bot/create', methods=['POST'])
def create_bot():
    """Create a new bot instance"""
    try:
        data: Dict[str, Any] = request.json or {}
        bot_id = data.get('bot_id')
        bot_name = data.get('bot_name')
        system_prompt = data.get('system_prompt')
        confidence_threshold = data.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        escalation_message = data.get('escalation_message')
        language = data.get('language', 'en')
        
        if not bot_id or not bot_name:
            return jsonify({'error': 'bot_id and bot_name are required'}), 400
        
        # Provide default system prompt if not provided
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
        
        result = bot_manager.create_bot(bot_id, bot_name, system_prompt, confidence_threshold, 
                                       escalation_message, language)
        
        if not result["success"]:
            return jsonify({'error': result["error"]}), 400
        
        return jsonify(result), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bot/<bot_id>/load_qa', methods=['POST'])
def load_qa_database(bot_id):
    """Load Q&A database for specific bot"""
    try:
        bot_config = bot_manager.get_bot(bot_id)
        if not bot_config:
            return jsonify({'error': f'Bot {bot_id} not found'}), 404
        
        data: Dict[str, Any] = request.json or {}
        qa_data = data.get('qa_pairs', [])
        
        if not qa_data:
            return jsonify({'error': 'No Q&A pairs provided'}), 400
        
        # Clear existing database
        bot_config.clear_qa_database()
        
        # Add QA pairs
        bot_config.add_qa_pairs(qa_data)
        
        # Generate embeddings for all questions
        print(f"[{bot_id}] Generating embeddings...")
        questions = [item['question'] for item in bot_config.qa_database]
        
        # Generate embeddings with error handling
        embeddings: List[List[float]] = []
        for question in questions:
            embedding = VectorSearcher.get_embedding(question)
            if embedding is not None:
                embeddings.append(embedding)
        
        bot_config.qa_embeddings = embeddings
        
        # Create TF-IDF matrix for keyword search
        print(f"[{bot_id}] Creating TF-IDF matrix...")
        if questions:  # Only create TF-IDF if we have questions
            bot_config.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            if bot_config.tfidf_vectorizer is not None:
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
    start_time = time.time()
    bot_config = None
    
    try:
        bot_config = bot_manager.get_bot(bot_id)
        if not bot_config:
            return jsonify({'error': f'Bot {bot_id} not found'}), 404
        
        data: Dict[str, Any] = request.json or {}
        phone_number = data.get('phone_number')
        question = data.get('question')
        custom_abbreviations = data.get('abbreviations', {})
        
        if not phone_number or not question:
            return jsonify({'error': 'Missing phone_number or question'}), 400
        
        # Step 1: Preprocess query
        cleaned_query = QueryPreprocessor.clean(question, custom_abbreviations)
        print(f"[{bot_id}] Original: {question}")
        print(f"[{bot_id}] Cleaned: {cleaned_query}")
        
        # Step 2: Generate embedding
        query_embedding = VectorSearcher.get_embedding(cleaned_query)
        if not query_embedding:
            bot_config.update_performance_metrics(time.time() - start_time, False)
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
        reranked_results = ResultReranker.combine_and_rerank(
            vector_results, 
            keyword_results,
            bot_config.vector_weight
        )
        
        if not reranked_results:
            # No results found - escalate
            ModeratorQueue.add_to_queue(bot_id, phone_number, question, 0.0)
            bot_config.escalation_count += 1
            bot_config.update_performance_metrics(time.time() - start_time, False)
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
            bot_config.update_performance_metrics(time.time() - start_time, False)
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
        
        # Update performance metrics
        bot_config.update_performance_metrics(time.time() - start_time, True)
        
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
        if bot_config:
            bot_config.update_performance_metrics(time.time() - start_time, False)
        return jsonify({'error': str(e)}), 500

@app.route('/bot/<bot_id>/config', methods=['GET', 'PUT'])
def bot_config_endpoint(bot_id):
    """Get or update bot configuration"""
    if request.method == 'GET':
        bot_config = bot_manager.get_bot(bot_id)
        if not bot_config:
            return jsonify({'error': f'Bot {bot_id} not found'}), 404
        
        return jsonify(bot_config.get_config_summary())
    
    elif request.method == 'PUT':
        data: Dict[str, Any] = request.json or {}
        
        result = bot_manager.update_bot(bot_id, **data)
        
        if not result["success"]:
            return jsonify({'error': result["error"]}), 400
        
        return jsonify(result)

@app.route('/bot/<bot_id>/delete', methods=['DELETE'])
def delete_bot(bot_id):
    """Delete a bot instance"""
    result = bot_manager.delete_bot(bot_id)
    
    if not result["success"]:
        return jsonify({'error': result["error"]}), 404
    
    return jsonify(result)

@app.route('/bot/<bot_id>/stats', methods=['GET'])
def bot_stats(bot_id):
    """Get bot statistics"""
    result = bot_manager.get_bot_stats(bot_id)
    
    if not result["success"]:
        return jsonify({'error': result["error"]}), 404
    
    # Remove the success key from the response
    result.pop("success", None)
    return jsonify(result)

@app.route('/bot/<bot_id>/health', methods=['GET'])
def bot_health(bot_id):
    """Get health status for a specific bot"""
    if not bot_manager.bot_exists(bot_id):
        return jsonify({'error': f'Bot {bot_id} not found'}), 404
    
    health_status = health_checker.get_bot_health(bot_id)
    health_status['last_check_time'] = health_checker.get_last_check_time(bot_id)
    
    return jsonify(health_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)