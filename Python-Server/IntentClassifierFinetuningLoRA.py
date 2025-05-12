import os
import logging
import torch
import numpy as np
import pandas as pd

from transformers import (RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, set_seed)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import LoRA modules from PEFT
from peft import LoraConfig, get_peft_model, TaskType

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def load_and_prepare_data(csv_path: str):
    logger = logging.getLogger(__name__)
    try:
        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        if 'Question' not in df.columns or 'Label' not in df.columns:
            raise ValueError("CSV file must contain 'Question' and 'Label' columns.")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise

    # Map string labels to integers (sorted for consistency)
    label_map = {label: idx for idx, label in enumerate(sorted(df['Label'].unique()))}
    df['Label'] = df['Label'].map(label_map)

    # Split dataset while preserving label proportions
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Question'].tolist(), df['Label'].tolist(),
        test_size=0.2, random_state=42, stratify=df['Label']
    )

    return train_texts, train_labels, val_texts, val_labels, label_map

def tokenize_texts(tokenizer, texts, max_length=64):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def main():
    logger = setup_logging()
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Paths and hyperparameters
    dataset_path = "D:/Human Centered NLP/conversation-ai/SyntheticDatasetforIntentClassifier/Intent_Classifier_Dataset - Processed.csv"
    output_dir = "./intent_model_lora"
    num_epochs = 3
    train_batch_size = 8
    eval_batch_size = 8
    learning_rate = 2e-5
    weight_decay = 0.01
    max_length = 64

    # Load and prepare data
    logger.info("Loading dataset...")
    train_texts, train_labels, val_texts, val_labels, label_map = load_and_prepare_data(dataset_path)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize texts
    logger.info("Tokenizing dataset...")
    train_encodings = tokenize_texts(tokenizer, train_texts, max_length)
    val_encodings = tokenize_texts(tokenizer, val_texts, max_length)

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels,
    })
    val_dataset = Dataset.from_dict({
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": val_labels,
    })

    # Load pre-trained RoBERTa model for sequence classification
    logger.info("Loading pre-trained model...")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_map))
    model.to(device)

    # Set up LoRA configuration and adapt the model
    logger.info("Applying LoRA adaptation...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        r=8,                        # Rank of the low-rank decomposition
        lora_alpha=16,              # Scaling factor
        lora_dropout=0.05,           # Dropout probability for LoRA layers
        target_modules=["query", "key"],  # Target modules in the transformer to be adapted
        bias='none'
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA model summary:")
    logger.info(model.print_trainable_parameters())

    # Training arguments configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",   # Changed to "epoch" to match evaluation strategy
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to=["none"]
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Begin training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model and tokenizer
    logger.info("Saving model and tokenizer...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training completed and model saved.")

    # Define a simple inference function
    def classify_intent(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        # Convert numeric label back to the corresponding string label
        inv_label_map = {v: k for k, v in label_map.items()}
        return inv_label_map[predicted_label]

    # Interactive inference loop (can be removed or adjusted for production API usage)
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        prediction = classify_intent(user_query)
        logger.info(f"Predicted Intent: {prediction}")
        print("Predicted Intent:", prediction)

if __name__ == '__main__':
    main()