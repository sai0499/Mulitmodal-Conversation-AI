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

# ————————————————
# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# ————————————————
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ————————————————
# Start Milvus-Lite server at startup
default_server.set_base_dir("./milvus_data")
default_server.start()
milvus_port = default_server.listen_port
logger.info("Milvus-Lite running at http://127.0.0.1:%d", milvus_port)
connections.connect(alias="default", host="127.0.0.1", port=milvus_port)

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

# ————————————————
# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# ————————————————
# Global flag for forcing web-search mode
GLOBAL_SEARCH_MODE = False

label_map_path = "./intent_model_lora/label_map.json"
with open(label_map_path, "r") as f:
    label_map = json.load(f)      # e.g. {"RAG Search":0, "web search":1, "slot filling":2, "small talk":3}
inv_label_map = {v: k for k, v in label_map.items()}

# Load the intent classifier once
try:
    classifier_model, classifier_tokenizer = load_intent_model("./intent_model_lora")
    logger.info("Intent classifier model loaded.")
except Exception as e:
    logger.error("Failed to load intent classifier: %s", e)
    classifier_model = classifier_tokenizer = None

# ————————————————
# The fixed role‐prompt that defines your LLM’s persona
LLM_ROLE_PROMPT = (
    "Role: You are an AI Assistant named Lucy in Uni-Ask who helps students at the Otto von Guriecke University(OVGU)."
    "with any university-related questions."
)

# ————————————————
@app.route('/api/search-toggle', methods=['POST'])
def set_search_mode():
    global GLOBAL_SEARCH_MODE
    data = request.get_json()
    if not data or 'searchMode' not in data:
        return jsonify({"error": "No search mode provided."}), 400
    GLOBAL_SEARCH_MODE = bool(data['searchMode'])
    logger.info("Global search mode set to %s", GLOBAL_SEARCH_MODE)
    return jsonify({"search_mode": GLOBAL_SEARCH_MODE}), 200

# ————————————————
@app.route('/api/gemini', methods=['POST'])
def gemini_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided."}), 400

    # 1) Split the incoming fields
    user_transcript = data['text']                  # this is the history + "Assistant:" tail
    raw_query       = data.get('rawQuery', user_transcript)
    skip_validation = data.get('skipApiKeyValidation', False)

    # 2) Always start the final prompt with the LLM role‐prompt
    #    + a blank line + the transcript (history + "Assistant:")
    base_prompt = f"{LLM_ROLE_PROMPT}\n\n{user_transcript}"

    # 3) If skipping classification/retrieval, use only base_prompt
    if skip_validation:
        full_prompt = base_prompt

    else:
        # 4) Decrypt user API key if provided
        enc_key = request.headers.get("X-User-ApiKey") or data.get("apiKey")
        dec_key = None
        if enc_key:
            try:
                dec_key = decrypt_api_key(enc_key)
            except Exception as e:
                logger.error("API key decryption failed: %s", e)
                return jsonify({"error": "Failed to decrypt API key."}), 500

        # 5) Classify only the fresh user query
        if GLOBAL_SEARCH_MODE:
            intent = "web search"
        else:
            if not (classifier_model and classifier_tokenizer):
                return jsonify({"error": "Intent classifier unavailable."}), 500
            label  = classify_intent(classifier_model, classifier_tokenizer, raw_query)
            intent = inv_label_map.get(label, "unknown_intent")
        logger.info("Classified intent: %s", intent)

        # 6) Attach retrieval context _after_ the base_prompt
        if intent == "web search" and dec_key:
            try:
                serp_data = get_answer_box_and_top_organic_results(raw_query, dec_key)
            except Exception as e:
                logger.warning("SERP API failed: %s", e)
                serp_data = None

            if serp_data:
                serp_context = json.dumps(serp_data)
                full_prompt = (
                    base_prompt +
                    "\n\nWeb Search Results:\n" + serp_context +
                    "\n\nBased on the conversation above and the web results, generate a helpful, concise response with relevant links."
                )
            else:
                full_prompt = base_prompt

        elif intent == "RAG Search":
            try:
                chunks = retrieve(raw_query, top_k=20)
                rag_lines = [
                    f"Source: {c['source_path']} (chunk {c['chunk_id']}): {c['chunk_text']}"
                    for c in chunks
                ]
                rag_context = "\n".join(rag_lines)
                full_prompt = (
                    base_prompt +
                    "\n\nRetrieved Context:\n" + rag_context +
                    "\n\nBased on the conversation above and the retrieved context, generate a helpful, concise response referring to  the relevant docs"
                )
            except Exception as e:
                logger.error("RAG retrieval failed: %s", e)
                full_prompt = base_prompt

        else:
            full_prompt = base_prompt

    # 7) Send the assembled prompt to Gemini
    try:
        reply = query_gemini_api(full_prompt)
        payload = {"reply": reply}
        if not skip_validation:
            payload["intent"] = intent
        return jsonify(payload), 200
    except Exception as e:
        logger.error("LLM call error: %s", e)
        return jsonify({"error": "Failed to generate response."}), 500

# ————————————————
@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio_endpoint():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    file = request.files['audio']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        text = transcribe_audio(file_path, whisper_model)
        os.remove(file_path)
        return jsonify({"transcription": text}), 200
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error("Transcription error: %s", e)
        return jsonify({"error": "Transcription failed."}), 500

# ————————————————
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
