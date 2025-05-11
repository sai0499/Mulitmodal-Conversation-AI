import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from rank_bm25 import BM25Okapi
from ranx import evaluate
from typing import List, Tuple
import logging
import os
import pdfplumber
import re
from nltk.corpus import wordnet
import nltk

# Download NLTK data for query expansion
nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DenseRetrieval:
    def __init__(self, model_name: str = 'multi-qa-mpnet-base-dot-v1', index_path: str = 'faiss_index.bin'):

        self.model = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.bm25 = None
        self.index_path = index_path
        logging.info(f"Initialized DenseRetrieval with model {model_name}")

    def preprocess_text(self, text: str) -> str:

        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(?:header|footer).*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        return text.lower()

    def segment_into_paragraphs(self, text: str, min_length: int = 100, max_length: int = 1000) -> List[str]:

        paragraphs = [p.strip() for p in re.split(r'\n{2,}|\.\s*\n', text) if p.strip()]
        filtered_paragraphs = []
        for p in paragraphs:
            if len(p) < min_length or p.startswith('http'):
                continue
            while len(p) > max_length:
                split_point = p.rfind('. ', 0, max_length) + 1
                if split_point <= 0:
                    split_point = max_length
                filtered_paragraphs.append(p[:split_point].strip())
                p = p[split_point:].strip()
            if len(p) >= min_length:
                filtered_paragraphs.append(p)
        logging.info(f"Segmented text into {len(filtered_paragraphs)} paragraphs")
        return filtered_paragraphs

    def encode_documents(self, documents: List[str]) -> np.ndarray:

        embeddings = self.model.encode(documents, batch_size=32, show_progress_bar=True)
        logging.info(f"Encoded {len(documents)} documents into embeddings of shape {embeddings.shape}")
        return embeddings

    def build_index(self, documents: List[str]):
        """
        Build and save FAISS index and BM25 index for document embeddings.

        Args:
            documents: List of document texts or chunks.
        """
        self.documents = []
        for doc in documents:
            doc = self.preprocess_text(doc)
            self.documents.extend(self.segment_into_paragraphs(doc))

        embeddings = self.encode_documents(self.documents)

        nlist = min(256, len(embeddings))
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        self.index.train(embeddings)
        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path)

        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        logging.info(f"Built and saved FAISS index with {self.index.ntotal} vectors and BM25 index")

    def load_index(self):
        """
        Load FAISS index from disk.
        """
        self.index = faiss.read_index(self.index_path)
        logging.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def expand_query(self, query: str) -> str:

        words = query.split()
        expanded_words = []
        for word in words:
            expanded_words.append(word)
            synonyms = wordnet.synsets(word)
            for syn in synonyms[:2]:
                for lemma in syn.lemmas():
                    expanded_words.append(lemma.name().replace('_', ' '))
        expanded_query = ' '.join(set(expanded_words))
        logging.info(f"Expanded query: {expanded_query}")
        return expanded_query

    def retrieve(self, query: str, top_k: int = 5, dense_weight: float = 0.6) -> List[Tuple[str, float]]:

        if self.index is None or self.bm25 is None:
            raise ValueError("Index not built or loaded. Call build_index or load_index first.")

        query = self.preprocess_text(query)
        expanded_query = self.expand_query(query)

        query_embedding = self.model.encode([expanded_query])[0]
        self.index.nprobe = 20
        dense_distances, dense_indices = self.index.search(np.array([query_embedding]), top_k * 3)
        dense_scores = {idx: 1 / (1 + dist) for idx, dist in zip(dense_indices[0], dense_distances[0]) if
                        idx >= 0 and idx < len(self.documents)}

        tokenized_query = expanded_query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_max = max(bm25_scores, default=1.0) + 1e-10
        bm25_scores = {i: score / bm25_max for i, score in enumerate(bm25_scores) if score > 0}

        combined_scores = {}
        for idx in set(dense_scores.keys()).union(bm25_scores.keys()):
            dense_score = dense_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            combined_score = dense_weight * dense_score + (1 - dense_weight) * bm25_score
            combined_scores[idx] = combined_score

        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k * 3]

        pairs = [(query, self.documents[idx]) for idx in top_indices]
        if pairs:
            cross_scores = self.cross_encoder.predict(pairs)
            reranked = sorted(zip(top_indices, cross_scores), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            reranked = [(idx, combined_scores.get(idx, 0.0)) for idx in top_indices[:top_k]]

        results = []
        for idx, score in reranked:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        logging.info(f"Retrieved {len(results)} documents for query: {query}")
        return results


def load_document_from_pdf(file_path: str) -> List[str]:
    documents = []
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return documents

    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text() or ''
                if page_text.strip():
                    text += page_text + '\n'
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = ' '.join(cell or '' for cell in row if cell)
                        if row_text.strip():
                            text += row_text + '\n'

            if text.strip():
                documents.append(text)
                logging.info(f"Extracted text (first 200 chars): {text[:200]}...")
            else:
                logging.warning(f"No text extracted from {file_path}")
        logging.info(f"Loaded PDF: {file_path}")
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")

    return documents


def evaluate_retrieval(retriever, queries: dict, qrels: dict, top_k: int = 5) -> dict:
    run = {}
    for qid, query in queries.items():
        results = retriever.retrieve(query, top_k=top_k)
        run[qid] = {f"doc_{i}": score for i, (_, score) in enumerate(results)}

    metrics = evaluate(qrels, run, ['mrr@5', 'precision@5', 'recall@5'])
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics


def main():
    # Path to your PDF file
    pdf_file_path = 'C:/Users/saite/Downloads/SPO_Master_2021_04_08_Courtesy_Translation.pdf'
    # Load document from PDF
    documents = load_document_from_pdf(pdf_file_path)

    if not documents:
        print(f"No valid text found in {pdf_file_path}. Please check the PDF file.")
        return

    # Initialize retriever
    retriever = DenseRetrieval()

    # Build index
    retriever.build_index(documents)

    # Optional: Evaluate retrieval (uncomment and provide your queries and qrels)
    # queries = {
    #     'q1': 'What is self-leadership?',
    #     'q2': 'How to manage stress?'
    # }
    # qrels = {
    #     'q1': {'doc_0': 1, 'doc_1': 0},
    #     'q2': {'doc_2': 1, 'doc_3': 0}
    # }
    # evaluate_retrieval(retriever, queries, qrels)

    # Interactive query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting...")
            break
        if not query:
            print("Empty query, please try again.")
            continue

        # Retrieve documents
        results = retriever.retrieve(query, top_k=3)
        print(f"\nQuery: {query}")
        print("Top retrieved documents:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"Document: {doc}")


if __name__ == "__main__":
    main()