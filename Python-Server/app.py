# app.py – Production Ready Version
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import os
import tempfile
from stt_transcriber import transcribe_audio
from whisper_model import model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Use a secure temporary folder or environment-based config in production
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
        # Clean up the temporary file
        os.remove(file_path)
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # In production, disable debug mode and use a production WSGI server (e.g., gunicorn)
    app.run(host='0.0.0.0', port=5000, debug=False)
