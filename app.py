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
from api_endpoints.middleware import validate_bot_exists, validate_bot_ready
from api_endpoints.bot_dispatcher import BotDispatcher
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time
from typing import Dict, Any, Optional, List

app = Flask(__name__)

# Initialize bot manager and health checker
bot_manager = BotManager()
health_checker = HealthChecker(bot_manager)
bot_dispatcher = BotDispatcher(bot_manager)

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
        bot_category = data.get('category', 'general')
        vector_weight = data.get('vector_weight', 0.7)
        keyword_weight = data.get('keyword_weight', 0.3)
        embedding_model = data.get('embedding_model', 'text-embedding-3-small')
        chat_model = data.get('chat_model', 'gpt-4o-mini')
        response_style = data.get('response_style', 'concise')
        
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
                                       escalation_message, language, bot_category)
        
        # Update bot with additional configuration
        if result["success"]:
            bot_config = bot_manager.get_bot(bot_id)
            if bot_config:
                bot_config.update_config(
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    embedding_model=embedding_model,
                    chat_model=chat_model,
                    response_style=response_style
                )
        
        # Add custom abbreviations if provided
        custom_abbreviations = data.get('custom_abbreviations', {})
        if custom_abbreviations:
            QueryPreprocessor.add_custom_abbreviations(bot_id, custom_abbreviations)
        
        if not result["success"]:
            return jsonify({'error': result["error"]}), 400
        
        return jsonify(result), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bot/<bot_id>/load_qa', methods=['POST'])
@validate_bot_exists(bot_manager)
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
            embedding = VectorSearcher.get_embedding(question, bot_config.embedding_model)
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
@validate_bot_ready(bot_manager)
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
        language_preference = data.get('language', 'auto')
        search_method = data.get('search_method', 'hybrid')  # hybrid, vector, keyword, rrf
        min_similarity = data.get('min_similarity', 0.0)
        boost_category = data.get('boost_category')
        response_style = data.get('response_style', getattr(bot_config, 'response_style', 'concise'))
        temperature = data.get('temperature', 0.3)
        max_tokens = data.get('max_tokens', 500)
        
        if not phone_number or not question:
            return jsonify({'error': 'Missing phone_number or question'}), 400
        
        # Step 1: Preprocess query with enhanced processing
        # Get bot-specific custom abbreviations
        bot_custom_abbreviations = QueryPreprocessor.get_custom_abbreviations(bot_id)
        # Merge with request-specific abbreviations
        if custom_abbreviations:
            bot_custom_abbreviations.update(custom_abbreviations)
        
        # Determine bot category for appropriate abbreviation expansion
        bot_category = getattr(bot_config, 'category', 'general')
        
        cleaned_query, detected_language = QueryPreprocessor.clean(
            question, 
            bot_custom_abbreviations, 
            language_preference,
            bot_category
        )
        
        print(f"[{bot_id}] Original: {question}")
        print(f"[{bot_id}] Cleaned: {cleaned_query}")
        print(f"[{bot_id}] Detected language: {detected_language}")
        
        # Step 2: Generate embedding
        query_embedding = VectorSearcher.get_embedding(cleaned_query, bot_config.embedding_model)
        if not query_embedding:
            bot_config.update_performance_metrics(time.time() - start_time, False)
            return jsonify({'error': 'Failed to generate embedding'}), 500
        
        # Step 3 & 4: Search based on method
        vector_results = []
        keyword_results = []
        
        if search_method in ['hybrid', 'vector', 'rrf']:
            # Vector Search
            vector_results = VectorSearcher.search(
                query_embedding, 
                bot_config.qa_database, 
                bot_config.qa_embeddings, 
                top_k=10,  # Get more results for better reranking
                similarity_metric="cosine"
            )
        
        if search_method in ['hybrid', 'keyword', 'rrf']:
            # Keyword Search
            if boost_category:
                keyword_results = KeywordSearcher.advanced_search(
                    cleaned_query,
                    bot_config.qa_database,
                    bot_config.tfidf_vectorizer,
                    bot_config.tfidf_matrix,
                    top_k=10,  # Get more results for better reranking
                    min_similarity=min_similarity,
                    boost_category=boost_category,
                    category_boost_factor=1.5
                )
            else:
                keyword_results = KeywordSearcher.search(
                    cleaned_query,
                    bot_config.qa_database,
                    bot_config.tfidf_vectorizer,
                    bot_config.tfidf_matrix,
                    top_k=10,  # Get more results for better reranking
                    min_similarity=min_similarity
                )
        
        # Step 5: Rerank Results based on method
        reranked_results = []
        if search_method == 'rrf':
            # Use reciprocal rank fusion
            reranked_results = ResultReranker.reciprocal_rank_fusion(vector_results, keyword_results)
        elif search_method == 'hybrid':
            # Use weighted combination
            reranked_results = ResultReranker.combine_and_rerank(
                vector_results, 
                keyword_results,
                bot_config.vector_weight
            )
        elif search_method == 'vector':
            # Use only vector results
            reranked_results = vector_results[:5]
        elif search_method == 'keyword':
            # Use only keyword results
            reranked_results = keyword_results[:5]
        else:
            # Default to weighted combination
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
                'escalated': True,
                'matched_question': reranked_results[0]['question'] if reranked_results else '',
                'search_method': search_method
            })
        
        # Step 6: Generate Answer with Enhanced Features
        answer, confidence, confidence_reason, response_metadata = AnswerGenerator.generate_with_confidence(
            question, 
            reranked_results,
            bot_config.system_prompt,
            detected_language
        )
        
        # Calculate final confidence score
        top_score = reranked_results[0]['final_score'] if 'final_score' in reranked_results[0] else reranked_results[0]['score']
        final_confidence = min(top_score, confidence)  # Use the more conservative confidence score
        
        print(f"[{bot_id}] Final confidence score: {final_confidence}")
        
        if final_confidence < bot_config.confidence_threshold:
            # Low confidence - escalate to moderator
            ModeratorQueue.add_to_queue(bot_id, phone_number, question, final_confidence)
            bot_config.escalation_count += 1
            bot_config.update_performance_metrics(time.time() - start_time, False)
            return jsonify({
                'bot_id': bot_id,
                'phone_number': phone_number,
                'answer': bot_config.escalation_message,
                'confidence': final_confidence,
                'confidence_reason': confidence_reason,
                'escalated': True,
                'matched_question': reranked_results[0]['question'],
                'search_method': search_method,
                'response_metadata': response_metadata
            })
        
        # Step 7: High confidence - Return answer
        # Update performance metrics
        bot_config.update_performance_metrics(time.time() - start_time, True)
        
        return jsonify({
            'bot_id': bot_id,
            'phone_number': phone_number,
            'answer': answer,
            'confidence': final_confidence,
            'confidence_reason': confidence_reason,
            'escalated': False,
            'matched_question': reranked_results[0]['question'],
            'category': reranked_results[0]['category'],
            'detected_language': detected_language,
            'search_method': search_method,
            'response_metadata': response_metadata
        })
        
    except Exception as e:
        print(f"Error processing query: {e}")
        if bot_config:
            bot_config.update_performance_metrics(time.time() - start_time, False)
        return jsonify({'error': str(e)}), 500

@app.route('/bot/<bot_id>/config', methods=['GET', 'PUT'])
@validate_bot_exists(bot_manager)
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
@validate_bot_exists(bot_manager)
def delete_bot(bot_id):
    """Delete a bot instance"""
    result = bot_manager.delete_bot(bot_id)
    
    if not result["success"]:
        return jsonify({'error': result["error"]}), 404
    
    return jsonify(result)

@app.route('/bot/<bot_id>/stats', methods=['GET'])
@validate_bot_exists(bot_manager)
def bot_stats(bot_id):
    """Get bot statistics"""
    result = bot_manager.get_bot_stats(bot_id)
    
    if not result["success"]:
        return jsonify({'error': result["error"]}), 404
    
    # Remove the success key from the response
    result.pop("success", None)
    return jsonify(result)

@app.route('/bot/<bot_id>/health', methods=['GET'])
@validate_bot_exists(bot_manager)
def bot_health(bot_id):
    """Get health status for a specific bot"""
    health_status = health_checker.get_bot_health(bot_id)
    health_status['last_check_time'] = health_checker.get_last_check_time(bot_id)
    
    return jsonify(health_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)