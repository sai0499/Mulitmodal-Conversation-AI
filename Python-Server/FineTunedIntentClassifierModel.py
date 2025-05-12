#!/usr/bin/env python3
import json
import torch
import argparse
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification
)
from peft import PeftModel

def load_model(model_dir: str):
    """
    Load the LoRA‐fine‐tuned RoBERTa intent classifier.
    Returns: (model, tokenizer)
    """
    # 1) read your label map to infer how many labels you trained on
    with open(f"{model_dir}/label_map.json", "r") as f:
        label_map = json.load(f)               # e.g. {"RAG Search":0, "web search":1, ...}
    num_labels = len(label_map)

    # 2) tokenizer (from your saved dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)

    # 3) new config with the correct head size
    config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels)

    # 4) base model with resized head
    base_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        config=config
    )

    # 5) overlay your LoRA adapter weights
    model = PeftModel.from_pretrained(base_model, model_dir)

    # 6) move to device & eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer


def classify_intent(
    model: torch.nn.Module,
    tokenizer: RobertaTokenizer,
    text: str,
    max_length: int = 64
) -> int:
    """
    Tokenize `text`, run it through the model, and return the predicted label ID.
    """
    # 1) tokenize + pad/trim
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    # 2) send inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3) forward pass
    with torch.no_grad():
        logits = model(**inputs).logits

    # 4) return argmax
    return int(torch.argmax(logits, dim=-1).item())


def main():
    """
    Simple REPL so you can run:
      python FineTunedIntentClassifierModel.py --model_dir ./intent_model_lora
    """
    parser = argparse.ArgumentParser(
        description="CLI for your LoRA‐fine‐tuned RoBERTa intent classifier"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./intent_model_lora",
        help="Path to your fine‐tuned model directory"
    )
    args = parser.parse_args()

    # load model + tokenizer
    model, tokenizer = load_model(args.model_dir)

    # reload label_map so we can print names
    with open(f"{args.model_dir}/label_map.json", "r") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}

    print(f"Loaded model from {args.model_dir} → {len(label_map)} labels.")
    print("Type a sentence and press Enter (or 'exit' to quit):")

    while True:
        text = input("> ").strip()
        if text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        label_id = classify_intent(model, tokenizer, text)
        intent_name = inv_label_map.get(label_id, "UNKNOWN")
        print(f"→ Predicted Intent [{label_id}]: {intent_name}\n")


if __name__ == "__main__":
    main()
