import logging
import re
from pathlib import Path
from typing import List, Dict

import torch
from transformers import CLIPTokenizerFast, CLIPModel

from docling.datamodel.base_models      import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend   import PyPdfiumDocumentBackend
from docling.document_converter         import DocumentConverter, PdfFormatOption

from milvus import default_server
from pymilvus import (connections, utility, Collection, CollectionSchema, FieldSchema, DataType,)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DOCS_DIR         = Path(r"C:\Users\saite\Downloads\docs")       # folder with your .pdf / .txt
MILVUS_DATA_DIR  = "./milvus_data"      # where Milvus-Lite persists data
COLLECTION_NAME  = "rag_collection"
CLIP_DIM         = 512                  # CLIP ViT-B/32 embedding size
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORDS        = 500
OVERLAP          = 50
# ─────────────────────────────────────────────────────────────────────────────


def start_milvus_lite(data_dir: str):
    default_server.set_base_dir(data_dir)
    default_server.start()
    port = default_server.listen_port
    logging.info("Milvus-Lite running at http://127.0.0.1:%d", port)
    connections.connect(alias="default", host="127.0.0.1", port=port)


def init_collection() -> Collection:
    fields = [
        FieldSchema("id",            dtype=DataType.INT64,        is_primary=True, auto_id=True),
        FieldSchema("emb",           dtype=DataType.FLOAT_VECTOR, dim=CLIP_DIM),
        FieldSchema("source_path",   dtype=DataType.VARCHAR,      max_length=1024),
        FieldSchema("chunk_id",      dtype=DataType.INT64),
        FieldSchema("chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema("chunk_preview", dtype=DataType.VARCHAR,      max_length=65535),
    ]
    schema = CollectionSchema(fields, description="RAG document chunks")
    if utility.has_collection(COLLECTION_NAME):
        logging.info("Dropping existing collection '%s'", COLLECTION_NAME)
        utility.drop_collection(COLLECTION_NAME)
    logging.info("Creating collection '%s'", COLLECTION_NAME)
    return Collection(name=COLLECTION_NAME, schema=schema)


def load_and_dump_documents(paths: List[Path]) -> Dict[str, str]:
    pdf_opts = PdfPipelineOptions(do_table_structure=True)
    pdf_opts.do_ocr = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_opts,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    docs: Dict[str, str] = {}
    for p in paths:
        try:
            if p.suffix.lower() == ".pdf":
                logging.info("Converting PDF %s", p)
                result = converter.convert(str(p))
                md = result.document.export_to_markdown()
            else:
                logging.info("Reading text file %s", p)
                md = p.read_text(encoding="utf-8")

            md_out = p.with_suffix(".md")
            md_out.write_text(md, encoding="utf-8")
            logging.info("Wrote markdown to %s", md_out)
            docs[str(p)] = md

        except Exception as e:
            logging.error("Failed to process %s: %s", p, e)
    return docs


def split_markdown_blocks(markdown: str) -> List[str]:
    """
    Emit each heading or table row as its own block,
    and group other lines into paragraph blocks.
    """
    lines = markdown.splitlines()
    blocks: List[str] = []
    buf: List[str] = []

    def flush_buf():
        nonlocal buf
        if buf:
            blocks.append("\n".join(buf))
            buf = []

    for line in lines:
        # heading => flush, then as its own block
        if re.match(r"^#{1,6}\s", line):
            flush_buf()
            blocks.append(line)
        # any table line => flush, then own block
        elif re.search(r"\|.*\|", line):
            flush_buf()
            blocks.append(line)
        else:
            buf.append(line)
    flush_buf()
    return blocks


def chunk_blocks(
    blocks: List[str],
    max_words: int = MAX_WORDS,
    overlap:   int = OVERLAP
) -> List[str]:
    chunks: List[str] = []
    curr: List[str] = []
    curr_count = 0

    for block in blocks:
        wc = len(block.split())
        # if adding this block overflows, close current chunk
        if curr and curr_count + wc > max_words:
            chunks.append(" ".join(curr))
            carry = " ".join(" ".join(curr).split()[-overlap:])
            curr, curr_count = [carry], len(carry.split())
        curr.append(block)
        curr_count += wc

    if curr:
        chunks.append(" ".join(curr))
    return chunks


def embed_text(texts: List[str], tokenizer, model, device: str):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu()


def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    # 1) Start & connect to Milvus-Lite
    start_milvus_lite(MILVUS_DATA_DIR)

    # 2) Create/Recreate collection
    coll = init_collection()

    # 3) Convert docs → markdown
    paths = list(DOCS_DIR.glob("**/*"))
    docs  = load_and_dump_documents(paths)

    # 4) Load CLIP
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    model     = CLIPModel       .from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    # 5) Chunk, embed, preview, insert
    for src, md in docs.items():
        blocks = split_markdown_blocks(md)
        chunks = chunk_blocks(blocks)
        total  = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            preview = chunk[:200] + ("…" if len(chunk) > 200 else "")
            print(f"\n--- Chunk {idx}/{total} of {src} preview ---\n{preview}\n")
            emb = embed_text([chunk], tokenizer, model, DEVICE)[0].tolist()
            print(f"Embedding vector ({len(emb)} dims):\n{emb}\n")
            coll.insert([{
                "emb":           emb,
                "source_path":   src,
                "chunk_id":      idx - 1,
                "chunk_text": chunk,
                "chunk_preview": preview,
            }])
            logging.info("Inserted chunk %d/%d for %s", idx, total, src)
    # 6) Create index & load
    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
    logging.info("Creating index on '%s.emb'", COLLECTION_NAME)
    coll.create_index(field_name="emb", index_params=index_params)
    coll.load()
    logging.info("Collection '%s' is ready.", COLLECTION_NAME)

    # 7) Shutdown
    connections.disconnect("default")
    default_server.stop()
    logging.info("Milvus-Lite stopped, data at '%s' persists.", MILVUS_DATA_DIR)


if __name__ == "__main__":
    main()
