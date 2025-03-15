import whisper
import torchaudio
import torch
import os

model = whisper.load_model("small").to(torch.float32)

def transcribe_audio(filepath):
    result = model.transcribe(filepath)
    print(result["text"])
    return result["text"]

audio_file = input("Enter the path to your audio file (MP3/WAV): ").strip()

if audio_file and os.path.exists(audio_file):
    transcribe_audio(audio_file)
else:
    print("❌ Error: File not found!")
