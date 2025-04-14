import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel

def load_model(model_dir: str):
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)                                  # Load tokenizer from the saved directory
    base_model = RobertaForSequenceClassification.from_pretrained(model_dir)                 # Load the base model (this includes the original model weights)
    model = PeftModel.from_pretrained(base_model, model_dir)                                 # Wrap with the LoRA/PEFT adapter
    return model, tokenizer

def classify_intent(model, tokenizer, text: str, max_length: int = 64) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenize input text for inference
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return predicted_label

def main():
    # Directory where the model and tokenizer were saved.
    model_dir = "./intent_model_lora"
    # Load the saved model and tokenizer.
    model, tokenizer = load_model(model_dir)
    # IMPORTANT: Use the same label mapping as during training.
    inv_label_map = {
        0: "RAG Search",
        1: "web search",
    }
    print("Model loaded. Enter text to classify the intent or type 'exit' to quit.")
    while True:
        text = input("Enter your query: ").strip()
        if text.lower() == "exit":
            break
        label_id = classify_intent(model, tokenizer, text)
        intent = inv_label_map.get(label_id, "Unknown")
        print("Predicted Intent:", intent)


if __name__ == "__main__":
    main()
