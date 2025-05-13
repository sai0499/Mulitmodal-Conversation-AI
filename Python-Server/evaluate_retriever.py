import os
import sys
import importlib.util
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
retriever_path = os.path.join(script_dir, "retriever.py")
spec = importlib.util.spec_from_file_location("retriever", retriever_path)
retriever_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retriever_mod)
retrieve = retriever_mod.retrieve

def evaluate_retriever(
    dataset_path: str,
    output_path: str,
    top_k: int = 50,
    rerank: bool = True
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row["question"]
        answer = row["answer"].strip()
        candidates: List[Dict[str, Any]] = retrieve(question, top_k=top_k, rerank=rerank)

        found = False
        rank = None
        for idx, cand in enumerate(candidates, start=1):
            text = (cand.get("chunk_text") or cand.get("chunk_preview") or "").strip()
            if answer.lower() in text.lower():
                found = True
                rank = idx
                break

        records.append({
            "question": question,
            "answer": answer,
            "found": found,
            "rank": rank
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_path, index=False)

    total = len(results_df)
    found_count = results_df["found"].sum()
    recall_pct = found_count / total * 100
    print(f"Found answers in top {top_k}: {found_count}/{total} ({recall_pct:.2f}% recall)")

    return results_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG retriever on Q&A dataset")
    parser.add_argument("--dataset", type=str, default="dataset.csv",
                        help="CSV with 'question' and 'answer' columns")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="Where to save per-question results")
    parser.add_argument("--top_k", type=int, default=50,
                        help="How many passages to retrieve per query")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Disable the reranking step")
    args = parser.parse_args()

    evaluate_retriever(
        dataset_path=args.dataset,
        output_path=args.output,
        top_k=args.top_k,
        rerank=args.rerank
    )
