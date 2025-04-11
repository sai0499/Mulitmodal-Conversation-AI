import whisper
import torch

def load_model(model_size="small", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_size, device=device)

# Load the model once at startup
model = load_model()
