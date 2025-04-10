import whisper
import sounddevice as sd
import numpy as np
import torchaudio
import torch
import os

model = whisper.load_model("small").to(torch.float32)

def record_audio(filename="recorded_audio.wav", silence_duration=3):
    
    samplerate = 22050
    buffer = []
    silent_chunks = 0
    chunk_duration = 0.2 
    chunk_samples = int(samplerate * chunk_duration)
    silence_threshold = 0.01 

    print("🎤 Start") 

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            buffer.extend(chunk.flatten().tolist())

            volume = np.sqrt(np.mean(np.square(chunk)))

            if volume < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= silence_duration / chunk_duration:
                    break
            else:
                silent_chunks = 0

    print("✅ Thank you")

    audio_data = np.array(buffer, dtype=np.float32)
    torchaudio.save(filename, torch.tensor(audio_data).unsqueeze(0), sample_rate=samplerate)
    return filename 

def transcribe_audio(filepath):
    """Transcribes audio using Whisper and directly prints the text."""
    result = model.transcribe(filepath)
    print(result["text"])
    return result["text"]

choice = input("Do you want to (1) Record audio or (2) Upload a file? (Enter 1 or 2): ").strip()

if choice == "1":
    audio_file = record_audio()  
    transcribe_audio(audio_file) 

elif choice == "2":
    audio_file = input("Enter the path to your audio file (MP3/WAV): ").strip()
    
    if audio_file and os.path.exists(audio_file):
        transcribe_audio(audio_file)
    else:
        print("❌ Error: File not found!")

else:
    print("❌ Invalid choice. Please enter 1 or 2.")
