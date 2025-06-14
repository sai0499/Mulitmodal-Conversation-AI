import os
import whisper
import torch

def transcribe_audio(audio_file_path, model):
    """
    Transcribe the audio file using the preloaded Whisper model.
    Additional audio preprocessing (conversion, normalization) can be added here.
    """
    result = model.transcribe(audio_file_path)
    transcription = result.get("text", "").strip()
    return transcription

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("audio_file", help="Path to the audio file to be transcribed.")
    parser.add_argument("--model_size", default="small", help="Size of the Whisper model to use (default: small).")
    parser.add_argument("--device", default=None, help="Device to use: 'cuda' for GPU or 'cpu' for CPU.")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"File not found: {args.audio_file}")
        sys.exit(1)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(args.model_size, device=device)
    transcription = model.transcribe(args.audio_file).get("text", "").strip()
    print(transcription)
