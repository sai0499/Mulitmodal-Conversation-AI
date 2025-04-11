#!/usr/bin/env python
import os
import sys
import argparse
import logging
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from packaging import version
import transformers


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_dataset(file_path):
    if not os.path.exists(file_path):
        logging.error("Dataset file not found: %s", file_path)
        sys.exit(1)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error("Failed to load dataset: %s", e)
        sys.exit(1)
    logging.info("Dataset loaded successfully. Shape: %s", df.shape)
    return df


def preprocess_dataset(df, text_column="Question", label_column="Label"):
    if text_column not in df.columns or label_column not in df.columns:
        logging.error("Dataset must contain '%s' and '%s' columns.", text_column, label_column)
        sys.exit(1)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        logging.info("'ID' column found and dropped.")

    label_map = {label: idx for idx, label in enumerate(df[label_column].unique())}
    df[label_column] = df[label_column].map(label_map)
    logging.info("Labels converted to numerical values. Label map: %s", label_map)
    return df, label_map


def tokenize_data(texts, tokenizer, max_length=64):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)


def prepare_datasets(df, tokenizer, text_column="Question", label_column="Label", test_size=0.2, random_state=42):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df[text_column], df[label_column], test_size=test_size, random_state=random_state
    )
    logging.info("Data split into train and test sets: %d train samples, %d test samples", len(train_texts),
                 len(test_texts))

    train_encodings = tokenize_data(train_texts.tolist(), tokenizer)
    test_encodings = tokenize_data(test_texts.tolist(), tokenizer)

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels.tolist(),
    })
    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels.tolist(),
    })
    return train_dataset, test_dataset


def train_model(train_dataset, test_dataset, num_labels, output_dir, learning_rate, batch_size, num_epochs,
                weight_decay=0.01):
    logging.info("Initializing model and training parameters")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    # Set additional training arguments conditionally based on transformers version.
    additional_args = {}
    # evaluation_strategy is supported from transformers 4.0.0 onwards.
    if version.parse(transformers.__version__) >= version.parse("4.0.0"):
        additional_args["evaluation_strategy"] = "epoch"
        additional_args["logging_dir"] = os.path.join(output_dir, "logs")
        additional_args["logging_steps"] = 10
    else:
        logging.warning(
            "Your transformers version (%s) does not support evaluation_strategy/logging_dir/logging_steps. Consider upgrading transformers.",
            transformers.__version__)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        **additional_args
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    logging.info("Starting training for %d epochs", num_epochs)
    trainer.train()
    logging.info("Training finished")
    return model


def save_model(model, tokenizer, output_dir):
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logging.info("Model and tokenizer saved to %s", output_dir)
    except Exception as e:
        logging.error("Failed to save model: %s", e)
        sys.exit(1)


def classify_intent(text, model, tokenizer, label_map, max_length=64):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    reverse_label_map = {v: k for k, v in label_map.items()}
    return reverse_label_map.get(predicted_label, "Unknown")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Production-ready Intent Classifier Training and Inference Script")
    parser.add_argument("--dataset", type=str, default="Intent_Classifier_Dataset - Processed.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="./intent_model",
                        help="Directory to save the trained model and tokenizer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use as test set")
    parser.add_argument("--text_column", type=str, default="Question",
                        help="Name of the question column in the CSV file")
    parser.add_argument("--label_column", type=str, default="Label", help="Name of the label column in the CSV file")
    args = parser.parse_args()

    logging.info("Loading dataset from: %s", args.dataset)
    df = load_dataset(args.dataset)
    df, label_map = preprocess_dataset(df, text_column=args.text_column, label_column=args.label_column)

    logging.info("Loading RoBERTa tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset, test_dataset = prepare_datasets(
        df,
        tokenizer,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size
    )

    model = train_model(
        train_dataset,
        test_dataset,
        num_labels=len(label_map),
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )

    save_model(model, tokenizer, args.output_dir)

    logging.info("Entering interactive inference mode. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("Enter your query (or type 'exit' to quit): ")
        except (KeyboardInterrupt, EOFError):
            logging.info("Exiting interactive mode.")
            break
        if user_input.lower() == "exit":
            break
        predicted_intent = classify_intent(user_input, model, tokenizer, label_map)
        print("Predicted Intent:", predicted_intent)


if __name__ == "__main__":
    main()
