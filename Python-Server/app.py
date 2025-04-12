from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from stt_transcriber import transcribe_audio
from whisper_model import model
from gemini_api import query_gemini_api  # Import the Gemini integration module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure a secure temporary folder (in production, use a robust configuration)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

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
        # Remove temporary file after processing.
        os.remove(file_path)
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route('/api/gemini', methods=['POST'])
def gemini_endpoint():
    """
    Endpoint to process text queries via the Gemini API.
    Expects a JSON payload with a 'text' key.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided."}), 400

    prompt = data['text']
    try:
        response_text = query_gemini_api(prompt)
        return jsonify({"reply": response_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # In production, use a robust WSGI server like gunicorn and disable debug mode.
    app.run(host='0.0.0.0', port=5000, debug=False)
