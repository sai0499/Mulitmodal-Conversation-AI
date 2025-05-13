import os
import sys
import json
import argparse
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from FineTunedIntentClassifierModel import load_model, classify_intent

def evaluate_intent_classifier(
    dataset_path: str,
    model_dir: str,
    output_path: str
):
    model, tokenizer = load_model(model_dir)

    with open(os.path.join(model_dir, "label_map.json"), "r") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}

    df = pd.read_csv(dataset_path)

    records = []
    for _, row in df.iterrows():
        text = row["question"]
        true_label = row["answer"]
        true_id = label_map.get(true_label)

        pred_id = classify_intent(model, tokenizer, text)
        pred_label = inv_label_map.get(pred_id, "UNKNOWN")

        records.append({
            "question":        text,
            "true_label":      true_label,
            "predicted_label": pred_label,
            "correct":         (pred_id == true_id)
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_path, index=False)

    total   = len(results_df)
    correct = results_df["correct"].sum()
    accuracy = correct / total * 100
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")

    try:
        from sklearn.metrics import classification_report
        y_true = results_df["true_label"]
        y_pred = results_df["predicted_label"]
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, digits=4))
    except ImportError:
        print("\nFor a detailed report, install scikit-learn (`pip install scikit-learn`).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate intent classifier")
    parser.add_argument("--dataset",  type=str, required=True,
                        help="CSV with columns 'question' and 'answer'")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Folder containing your fine-tuned model and label_map.json")
    parser.add_argument("--output",   type=str, default="intent_evaluation.csv",
                        help="Where to save per‐example prediction results")
    args = parser.parse_args()

    evaluate_intent_classifier(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        output_path=args.output
    )
