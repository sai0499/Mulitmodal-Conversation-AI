import os
import logging
from typing import List, Dict, Any

import torch
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPTokenizerFast, CLIPModel
from pymilvus import connections, Collection

# Configuration
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_collection")
MILVUS_HOST     = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT     = os.getenv("MILVUS_PORT", "19530")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
# Ensure collection is loaded only once
_collection_loaded = False


def get_milvus_collection() -> Collection:
    """
    Connects to an existing Milvus instance and returns the loaded collection.
    """
    if not connections.has_connection("default"):
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Connected to Milvus at %s:%s", MILVUS_HOST, MILVUS_PORT)

    coll = Collection(COLLECTION_NAME)
    global _collection_loaded
    if not _collection_loaded:
        coll.load()
        _collection_loaded = True
        logger.info("Loaded collection '%s'", COLLECTION_NAME)
    return coll


def embed_query(
    query: str,
    tokenizer: CLIPTokenizerFast,
    model: CLIPModel
) -> List[float]:
    """
    Embeds the input query string into a normalized vector.
    """
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vector = model.get_text_features(**inputs)
    vector = vector / vector.norm(p=2, dim=-1, keepdim=True)
    return vector[0].cpu().tolist()


def dense_search(
    coll: Collection,
    query_vec: List[float],
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Retrieves the top `limit` most similar chunks from Milvus using inner-product,
    including full chunk text if stored under 'chunk_text'.
    """
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }
    results = coll.search(
        data=[query_vec],
        anns_field="emb",
        param=search_params,
        limit=limit,
        output_fields=["source_path", "chunk_id", "chunk_preview", "chunk_text", "emb"],
    )
    hits = results[0]
    candidates = []
    for hit in hits:
        entity = hit.entity
        candidates.append({
            "score": hit.score,
            "source_path": entity.get("source_path"),
            "chunk_id": entity.get("chunk_id"),
            # preview for quick checks, full text if available
            "chunk_preview": entity.get("chunk_preview"),
            "chunk_text": entity.get("chunk_text"),
            "vector": entity.get("emb"),
        })
    return candidates


def diffusion_rerank(
    candidates: List[Dict[str, Any]],
    alpha: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Re-ranks candidates using personalized PageRank over cosine similarity graph.
    """
    n = len(candidates)
    if n <= 1:
        return candidates

    vectors = [c["vector"] for c in candidates]
    scores = [c["score"] for c in candidates]
    sim_matrix = cosine_similarity(vectors)

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=sim_matrix[i][j])

    total_score = sum(scores) or 1.0
    personalization = {i: scores[i] / total_score for i in range(n)}
    pagerank_scores = nx.pagerank(G, alpha=alpha, personalization=personalization, weight="weight")
    ordered_indices = sorted(range(n), key=lambda i: pagerank_scores[i], reverse=True)
    return [candidates[i] for i in ordered_indices]


def retrieve(
    query: str,
    top_k: int = 10,
    rerank: bool = True
) -> List[Dict[str, Any]]:
    """
    Full RAG retrieve pipeline:
     1. Embed query
     2. Dense search top 50
     3. Optionally re-rank
     4. Return top_k
    """
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    q_vec = embed_query(query, tokenizer, model)
    coll = get_milvus_collection()
    candidates = dense_search(coll, q_vec, limit=50)

    if rerank:
        candidates = diffusion_rerank(candidates)

    return candidates[:top_k]


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="RAG Retriever: Dense + Diffusion")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--top_k", type=int, default=10, help="Number of final results")
    parser.add_argument("--no_rerank", action="store_false", dest="rerank",
                        help="Skip diffusion-based re-ranking")

    args = parser.parse_args()
    results = retrieve(args.query, top_k=args.top_k, rerank=args.rerank)
    for idx, r in enumerate(results, start=1):
        print(f"{idx:2d}. [{r['score']:.4f}] {r['source_path']} (chunk {r['chunk_id']}):")
        # print full text if available, else preview
        text = r.get('chunk_text') or r.get('chunk_preview')
        print(f"    {text}\n")

    # Disconnect after CLI run
    if connections.has_connection("default"):
        connections.disconnect("default")
