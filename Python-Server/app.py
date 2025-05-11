import os
import json
import logging
import tempfile
import atexit
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from milvus import default_server
from pymilvus import connections
from decryption_util import decrypt_api_key
from serpAPI import get_answer_box_and_top_organic_results
from gemini_api import query_gemini_api
from retriever import retrieve
from FineTunedIntentClassifierModel import load_model as load_intent_model, classify_intent
from stt_transcriber import transcribe_audio
from whisper_model import model as whisper_model

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start Milvus-Lite server at application startup
default_server.set_base_dir("./milvus_data")
default_server.start()
milvus_port = default_server.listen_port
logger.info("Milvus-Lite running at http://127.0.0.1:%d", milvus_port)
# Establish Milvus connection
connections.connect(alias="default", host="127.0.0.1", port=milvus_port)

# Register shutdown hook to stop Milvus when application exits

def shutdown_milvus():
    try:
        connections.disconnect("default")
        logger.info("Disconnected Milvus-Lite connection")
    except Exception as e:
        logger.warning("Error disconnecting Milvus: %s", e)
    try:
        default_server.stop()
        logger.info("Milvus-Lite stopped")
    except Exception as e:
        logger.warning("Error stopping Milvus-Lite: %s", e)

atexit.register(shutdown_milvus)

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global flag for search mode
GLOBAL_SEARCH_MODE = False

# Load the intent classifier model once at startup
try:
    classifier_model, classifier_tokenizer = load_intent_model("./intent_model_lora")
    logger.info("Intent classifier model loaded successfully.")
except Exception as e:
    logger.error("Failed to load intent classifier model: %s", e)
    classifier_model, classifier_tokenizer = None, None

@app.route('/api/search-toggle', methods=['POST'])
def set_search_mode():
    global GLOBAL_SEARCH_MODE
    data = request.get_json()
    if not data or 'searchMode' not in data:
        return jsonify({"error": "No search mode provided"}), 400
    GLOBAL_SEARCH_MODE = bool(data['searchMode'])
    logger.info("Global search mode set to %s", GLOBAL_SEARCH_MODE)
    return jsonify({"search_mode": GLOBAL_SEARCH_MODE}), 200

@app.route('/api/gemini', methods=['POST'])
def gemini_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided."}), 400

    query = data['text']
    skip_validation = data.get("skipApiKeyValidation", False)

    if skip_validation:
        full_prompt = query
    else:
        encrypted_api_key = request.headers.get("X-User-ApiKey") or data.get("apiKey")
        decrypted_api_key = None
        if encrypted_api_key:
            try:
                decrypted_api_key = decrypt_api_key(encrypted_api_key)
            except Exception as e:
                logger.error("Error decrypting API key: %s", e)
                return jsonify({"error": "Failed to decrypt API key."}), 500
        else:
            logger.warning("No API key provided; continuing without API key.")

        # Determine intent
        if GLOBAL_SEARCH_MODE:
            intent = "web search"
        else:
            if classifier_model is None or classifier_tokenizer is None:
                return jsonify({"error": "Intent classifier is not available."}), 500
            label = classify_intent(classifier_model, classifier_tokenizer, query)
            intent = "web search" if label == 1 else "RAG Search"
        logger.info("Query classified as: %s", intent)

        # Build prompt
        if intent == "web search" and decrypted_api_key:
            try:
                serp_data = get_answer_box_and_top_organic_results(query, decrypted_api_key)
            except Exception as e:
                logger.warning("SERP API call failed: %s", e)
                serp_data = None

            if serp_data:
                serp_context = json.dumps(serp_data)
                full_prompt = (
                    "Role: You are an AI Assistant named Lucy in Uni-Ask who helps students with any university-related questions.\n\n"
                    f"User Query: {query}\n\n"
                    f"Web Search Results: {serp_context}\n\n"
                    "Based on the above, generate a helpful and concise response with relevant web links. "
                    "If the context doesn’t provide relevant information, say you don’t know. "
                    "Ignore the context for general chit-chat or greetings."
                )
            else:
                full_prompt = query

        elif intent == "RAG Search":
            try:
                chunks = retrieve(query, top_k=20)
                context_lines = [
                    f"Source: {c['source_path']} (chunk {c['chunk_id']}): {c['chunk_text']}"
                    for c in chunks
                ]
                rag_context = "\n".join(context_lines)
                print(rag_context)

                full_prompt = (
                    "Role: You are an AI Assistant named Lucy in Uni-Ask who helps students with any university-related questions.\n\n"
                    f"User Query: {query}\n\n"
                    f"Retrieved Context:\n{rag_context}\n\n"
                    "Based on the above, generate a helpful and concise response. "
                    "If the context doesn’t have relevant information, say you don’t know. "
                    "Ignore the context for general chit-chat or greetings."
                )
            except Exception as e:
                logger.error("RAG retrieval failed: %s", e)
                full_prompt = query
        else:
            full_prompt = query

    try:
        response_text = query_gemini_api(full_prompt)
        payload = {"reply": response_text}
        if not skip_validation:
            payload["intent"] = intent
        return jsonify(payload), 200

    except Exception as e:
        logger.error("Error processing query: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio_endpoint():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        transcription = transcribe_audio(file_path, whisper_model)
        os.remove(file_path)
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error("Transcription error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
