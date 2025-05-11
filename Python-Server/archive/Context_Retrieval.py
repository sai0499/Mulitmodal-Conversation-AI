import os
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import networkx as nx
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# Configuration & Logging
MILVUS_HOST       = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT       = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME   = "hybrid_embeddings"
EMBED_DIM         = 512          # CLIP ViT-B/32 outputs 512-dim vectors
TOP_K             = 50
GRAPH_K           = 10
DIFFUSION_ALPHA   = 0.85
DIFFUSION_ITERS   = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HybridRetriever")


# Milvus Lite Client
class MilvusLiteClient:
    def __init__(self, host: str, port: str, collection_name: str, embed_dim: int):
        self.alias = "default"
        connections.connect(self.alias, host=host, port=port)
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        self._ensure_collection()

    def _ensure_collection(self):
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id",
                            dtype=DataType.INT64,
                            is_primary=True,
                            auto_id=True),
                FieldSchema(name="embedding",
                            dtype=DataType.FLOAT_VECTOR,
                            dim=self.embed_dim),
                FieldSchema(name="meta",
                            dtype=DataType.VARCHAR,
                            max_length=1024)
            ]
            schema = CollectionSchema(fields, description="Hybrid embeddings")
            Collection(self.collection_name, schema)
            logger.info(f"Created collection '{self.collection_name}'")
        else:
            logger.info(f"Using existing collection '{self.collection_name}'")

        coll = Collection(self.collection_name)
        if not coll.has_index():
            index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 200}
            }
            coll.create_index("embedding", index_params)
            logger.info("Created HNSW index on 'embedding'")

    def insert(self, embeddings: np.ndarray, metas: List[str]) -> None:
        coll = Collection(self.collection_name)
        coll.insert([embeddings.tolist(), metas])
        coll.flush()
        logger.info(f"Inserted {len(metas)} embeddings")

    def search(self,
               query_emb: np.ndarray,
               top_k: int = TOP_K
               ) -> List[Dict[str, Any]]:
        coll = Collection(self.collection_name)
        results = coll.search(
            data=query_emb.tolist(),
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["meta"]
        )
        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "meta": hit.entity.get("meta")
            })
        return hits


# Graph-Based Diffusion Re-Ranker
class GraphDiffusionReranker:
    def __init__(self,
                 embeddings: np.ndarray,
                 ids: List[int],
                 k: int = GRAPH_K):
        self.ids = np.array(ids)
        self.embeddings = embeddings.astype("float32")
        self.k = k
        self.graph = self._build_knn_graph()

    def _build_knn_graph(self) -> nx.Graph:
        logger.info("Building k-NN graph")
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        index.add(self.embeddings)
        _, neighbors = index.search(self.embeddings, self.k + 1)

        G = nx.Graph()
        for i, nbrs in enumerate(neighbors):
            src = int(self.ids[i])
            for j in nbrs[1:]:
                dst = int(self.ids[j])
                G.add_edge(src, dst)
        logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def rerank(self,
               initial_hits: List[Dict[str, Any]],
               alpha: float = DIFFUSION_ALPHA,
               iters: int = DIFFUSION_ITERS
               ) -> List[Tuple[int, float]]:
        seeds = [h["id"] for h in initial_hits]
        subG = self.graph.subgraph(seeds).copy()
        logger.info(f"Running diffusion on subgraph of size {subG.number_of_nodes()}")

        # personalization vector
        p0 = {n: 1.0 / len(seeds) for n in subG.nodes()}
        scores = p0.copy()

        # degree-based normalization
        deg = {n: subG.degree(n) for n in subG.nodes()}

        for _ in range(iters):
            prev = scores.copy()
            for n in subG.nodes():
                neigh_sum = sum(prev[m] / deg[m] for m in subG.neighbors(n) if deg[m] > 0)
                scores[n] = alpha * neigh_sum + (1 - alpha) * p0[n]

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Hybrid Retriever Pipeline
class HybridRetriever:
    def __init__(self,
                 milvus_host: str,
                 milvus_port: str,
                 collection_name: str,
                 embed_dim: int):
        # CLIP via Hugging Face
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Milvus Lite
        self.mclient = MilvusLiteClient(
            milvus_host, milvus_port, collection_name, embed_dim
        )
        # placeholders for offline graph
        self._all_embeddings: np.ndarray = None
        self._all_ids: List[int] = []
        self._graph_reranker: GraphDiffusionReranker = None

    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        emb = emb.cpu().numpy()
        faiss.normalize_L2(emb)
        return emb

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        emb = emb.cpu().numpy()
        faiss.normalize_L2(emb)
        return emb

    def index_dataset(self,
                      items: List[Tuple[str, str]],
                      is_image: bool = True):
        metas, embs = [], []
        for meta, content in items:
            emb = (self.encode_image(content) if is_image
                   else self.encode_text(content))
            metas.append(meta)
            embs.append(emb.squeeze(0))

        embs = np.vstack(embs)
        self.mclient.insert(embs, metas)
        # build graph reranker
        self._all_embeddings = embs
        # here we assume Milvus auto-assigned IDs in insertion order 0..N-1
        self._all_ids = list(range(len(metas)))
        self._graph_reranker = GraphDiffusionReranker(
            embeddings=embs,
            ids=self._all_ids,
            k=GRAPH_K
        )

    def search(self,
               query: str,
               is_image: bool = False,
               top_k: int = TOP_K
               ) -> List[Dict[str, Any]]:
        q_emb = (self.encode_image(query) if is_image
                 else self.encode_text(query))
        initial = self.mclient.search(q_emb, top_k=top_k)
        reranked = self._graph_reranker.rerank(initial)
        id2meta = {h["id"]: h["meta"] for h in initial}
        return [
            {"id": doc_id, "meta": id2meta[doc_id], "score": float(score)}
            for doc_id, score in reranked
            if doc_id in id2meta
        ]

if __name__ == "__main__":
    retriever = HybridRetriever(
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        embed_dim=EMBED_DIM
    )

    # Index images
    # image_items = [("img1", "/path/to/img1.jpg"),
    #                ("img2", "/path/to/img2.jpg"), ...]
    # retriever.index_dataset(image_items, is_image=True)

    # Query
    # results = retriever.search("A cat sitting on a windowsill", is_image=False, top_k=20)
    # for r in results:
    #     print(r)
