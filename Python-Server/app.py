import os
import json
import tempfile
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from stt_transcriber import transcribe_audio
from whisper_model import model
from gemini_api import query_gemini_api
from FineTunedIntentClassifierModel import load_model, classify_intent
from serpAPI import get_answer_box_and_top_organic_results

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS (credentials aren’t needed here for our global flag approach)
CORS(app, supports_credentials=True)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variable to store the search mode flag.
GLOBAL_SEARCH_MODE = False

# Load the intent classifier model once at startup.
try:
    classifier_model, classifier_tokenizer = load_model("./intent_model_lora")
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

    # Update the global variable with the provided value.
    GLOBAL_SEARCH_MODE = bool(data['searchMode'])
    logger.info("Global search mode set to %s", GLOBAL_SEARCH_MODE)
    return jsonify({"search_mode": GLOBAL_SEARCH_MODE}), 200

@app.route('/api/gemini', methods=['POST'])
def gemini_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided."}), 400

    query = data['text']
    try:
        # Use the global variable instead of session.
        search_mode = GLOBAL_SEARCH_MODE
        logger.info("Global search mode is: %s", search_mode)

        if search_mode:
            logger.info("Search mode activated globally. Using SerpAPI for query.")
            serp_data = get_answer_box_and_top_organic_results(query)
            if serp_data is None:
                logger.warning("SerpAPI call failed; proceeding with the original query.")
                full_prompt = query
            else:
                serp_context = json.dumps(serp_data)
                full_prompt = (
                    f"User Query: {query}\n\n"
                    f"Web Search Results: {serp_context}\n\n"
                    "Based on the above, generate a helpful and concise response with relevant web links."
                )
            intent = "web search"
        else:
            if classifier_model is None or classifier_tokenizer is None:
                raise Exception("Intent classifier is not available.")

            predicted_label = classify_intent(classifier_model, classifier_tokenizer, query)
            intent = "web search" if predicted_label == 1 else "RAG Search"
            logger.info("Query classified as: %s", intent)

            if intent == "web search":
                serp_data = get_answer_box_and_top_organic_results(query)
                if serp_data is None:
                    logger.warning("SerpAPI call failed; proceeding with the original query.")
                    full_prompt = query
                else:
                    serp_context = json.dumps(serp_data)
                    full_prompt = (
                        f"User Query: {query}\n\n"
                        f"Web Search Results: {serp_context}\n\n"
                        "Based on the above, generate a helpful and concise response with relevant web links."
                    )
            else:
                full_prompt = query

        # Query the Gemini API with the composed prompt.
        response_text = query_gemini_api(full_prompt)
        return jsonify({"reply": response_text, "intent": intent}), 200

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
        transcription = transcribe_audio(file_path, model)
        os.remove(file_path)
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error("Transcription error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # In production, use a robust WSGI server (e.g., gunicorn) with debug mode off.
    app.run(host='0.0.0.0', port=5000, debug=False)
