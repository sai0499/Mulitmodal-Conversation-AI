import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

torch.cuda.is_available()
# Load dataset
df = pd.read_csv("study_queries.csv")

# Convert labels to numerical values
label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_map)

# Split into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenization function
def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=64)

# Tokenize datasets
train_encodings = tokenize_function(train_texts.tolist())
test_encodings = tokenize_function(test_texts.tolist())

# Convert to Hugging Face Dataset format
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

# Load RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_map))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./intent_model")
tokenizer.save_pretrained("./intent_model")

# Load trained model
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return list(label_map.keys())[list(label_map.values()).index(predicted_label)]

# User input for intent classification
while True:
    user_query = input("Enter your study-related query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    print("Predicted Intent:", classify_intent(user_query))
