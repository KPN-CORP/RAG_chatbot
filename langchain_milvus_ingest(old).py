"""
ingest_langchain_milvus_complete.py

- Uses LangChain Gemini embeddings (GoogleGenerativeAIEmbeddings).
- Ensures Milvus collection exists with schema including `checksum` field (pymilvus).
- Dedupes by checksum in Milvus when possible.
- Avoids double-embedding: if collection missing, embed first batch, create collection and insert with pymilvus.
- If collection exists, uses LangChainMilvus.from_documents(...) to insert (LangChain handles batching).
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import os
import logging
import time

from langchain.docstore.document import Document
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore

# LangChain Milvus wrapper (try common names)
try:
    from langchain_milvus import Milvus as LangchainMilvus
except Exception:
    try:
        from langchain_community.vectorstores.milvus import Milvus as LangchainMilvus
    except Exception:
        LangchainMilvus = None  # we'll handle later

# pymilvus
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    Collection,
    DataType,
    utility,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY environment variable")

# ---------------- helpers ----------------
def chunks_to_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for c in chunks:
        text = c.get("chunk_text") or c.get("text")
        if not text:
            raise ValueError("Chunk missing 'chunk_text' or 'text': %r" % (c,))
        metadata = {
            "chunk_id": c.get("chunk_id"),
            "checksum": c.get("checksum"),
            "source": c.get("source"),
            "page": c.get("page"),
            "element_type": c.get("element_type"),
            "table_id": c.get("table_id"),
            "row_id": c.get("row_id"),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def ensure_collection_exists_with_schema(collection_name: str, dim: int,
                                        host: str = "127.0.0.1", port: str = "19530",
                                        index_params: Optional[dict] = None) -> Collection:
    """
    Ensures the Milvus collection exists with the required schema:
    id (VARCHAR PK), embedding (FLOAT_VECTOR dim), checksum (VARCHAR), metadata (JSON).
    Returns a pymilvus Collection object.
    """
    if index_params is None:
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
    connections.connect(alias="default", host=host, port=port)
    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        vec_field = next((f for f in coll.schema.fields if f.dtype == DataType.FLOAT_VECTOR), None)
        if vec_field and vec_field.dim != dim:
            raise RuntimeError(f"Existing collection '{collection_name}' dim {vec_field.dim} != new dim {dim}")
        coll.load()
        return coll

    # create schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="checksum", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, description="RAG KB - chunks with Gemini embeddings")
    coll = Collection(name=collection_name, schema=schema)
    coll.create_index(field_name="embedding", index_params=index_params)
    coll.load()
    return coll


def milvus_query_checksums(collection_name: str, checksums: List[str], host: str = "127.0.0.1", port: str = "19530") -> Set[str]:
    found = set()
    if not checksums:
        return found
    try:
        connections.connect(alias="default", host=host, port=port)
    except Exception as e:
        log.warning("Milvus connect failed for checksum lookup: %s", e)
        return found
    if not utility.has_collection(collection_name):
        return found
    try:
        coll = Collection(collection_name)
        quoted = ", ".join([f'"{c}"' for c in checksums])
        expr = f'checksum in [{quoted}]'
        rows = coll.query(expr=expr, output_fields=["checksum"])
        for r in rows:
            cs = r.get("checksum")
            if cs:
                found.add(cs)
    except Exception as e:
        log.warning("Milvus checksum query failed (skipping): %s", e)
    return found


def insert_with_pymilvus(coll: Collection, ids: List[str], vectors: List[List[float]], checksums: List[str], metadatas: List[Dict[str, Any]]):
    coll.insert([ids, vectors, checksums, metadatas])
    coll.flush()
    coll.load()


def embed_batch_texts(texts: List[str], model: str, batch_size: int = 32) -> List[List[float]]:
    emb = GoogleGenerativeAIEmbeddings(model=model, google_api_key=API_KEY)
    vectors = [],
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vs = emb.embed_documents(batch)
        vectors.extend(vs)
    return vectors


# --------------- main orchestrator (final) ---------------
def process_and_store_chunks(
    chunks: List[Dict[str, Any]],
    collection_name: str,
    milvus_host: str = "127.0.0.1",
    milvus_port: str = "19530",
    embedding_model: str = "models/embedding-001",
    embed_batch_size: int = 32,
    insert_batch_size: int = 64,
    dedupe_with_milvus: bool = True,
) -> Tuple[int, int]:
    """
    End-to-end:
      - Convert chunks -> docs
      - If collection exists: dedupe by checksum (query Milvus) and use LangChainMilvus.from_documents to insert.
      - If collection does NOT exist:
          * Dedup not possible (first ingest)
          * Embed first batch with LangChain embeddings to learn dim
          * Create collection via pymilvus with that dim
          * Insert that first batch via pymilvus (no double embed)
          * For remaining batches: either use pymilvus insert (we show pymilvus for consistency) or use LangChain wrapper if you prefer.

    Returns (skipped, inserted)
    """
    if not chunks:
        return 0, 0

    # Convert and compact metadata
    docs = chunks_to_documents(chunks)

    # Connect to Milvus to check existence
    try:
        connections.connect(alias="default", host=milvus_host, port=milvus_port)
    except Exception as e:
        log.warning("Could not connect to Milvus: %s", e)

    coll_exists = utility.has_collection(collection_name) if utility.has_collection(collection_name) else False

    # If collection exists, we can dedupe by checksum
    to_process_docs = docs
    skipped = 0
    if dedupe_with_milvus and coll_exists:
        checksums = [d.metadata.get("checksum") for d in docs if d.metadata.get("checksum")]
        if checksums:
            found = milvus_query_checksums(collection_name, checksums, host=milvus_host, port=milvus_port)
            if found:
                new_docs = []
                for d in docs:
                    cs = d.metadata.get("checksum")
                    if cs and cs in found:
                        skipped += 1
                    else:
                        new_docs.append(d)
                to_process_docs = new_docs

    if not to_process_docs:
        log.info("No docs to process after dedupe.")
        return skipped, 0

    inserted = 0

    # CASE A: collection already exists => we can rely on LangChainMilvus.from_documents
    if coll_exists and LangchainMilvus is not None:
        log.info("Collection exists. Using LangChainMilvus.from_documents to embed+insert.")
        # use LangChain embeddings wrapper; LangChain will embed and insert
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=API_KEY)
        try:
            vectorstore = LangchainMilvus.from_documents(
                documents=to_process_docs,
                embedding=embeddings,
                collection_name=collection_name,
                connection_args={"host": milvus_host, "port": milvus_port},
            )
        except TypeError:
            # fallback signature
            vectorstore = LangchainMilvus.from_documents(
                documents=to_process_docs,
                embedding=embeddings,
                collection_name=collection_name,
                host=milvus_host,
                port=str(milvus_port),
            )
        inserted = len(to_process_docs)
        return skipped, inserted

    # CASE B: collection does not exist -> we must create it with correct dim.
    # Strategy:
    #  1) embed the first batch to get dim
    #  2) create collection via pymilvus with dim and checksum field
    #  3) insert first batch via pymilvus
    #  4) process subsequent batches (we will use pymilvus inserts to avoid re-embedding)
    log.info("Collection does not exist. Embedding first batch to determine dim and creating collection.")
    # Prepare records arrays for batches
    records = []
    for d in to_process_docs:
        chunk_id = d.metadata.get("chunk_id")
        checksum = d.metadata.get("checksum")
        text = d.page_content
        meta = d.metadata
        records.append((chunk_id, text, checksum, meta))

    # iterate batches: first batch embed -> create collection -> insert. Subsequent batches embed -> insert.
    # We always embed with LangChain embeddings, but insert with pymilvus to ensure schema.
    coll = None
    i = 0
    while i < len(records):
        batch_recs = records[i:i+insert_batch_size]
        ids = [r[0] for r in batch_recs]
        texts = [r[1] for r in batch_recs]
        checksums = [r[2] for r in batch_recs]
        metadatas = [r[3] for r in batch_recs]

        vectors = embed_batch_texts(texts, model=embedding_model, batch_size=embed_batch_size)
        if not vectors or len(vectors) != len(texts):
            raise RuntimeError("Unexpected embedding response")

        if coll is None:
            dim = len(vectors[0])
            coll = ensure_collection_exists_with_schema(collection_name, dim, host=milvus_host, port=milvus_port)
        # insert via pymilvus
        insert_with_pymilvus(coll, ids, vectors, checksums, metadatas)
        inserted += len(ids)
        i += insert_batch_size

    log.info("Inserted %d vectors into new collection '%s' (skipped=%d)", inserted, collection_name, skipped)
    return skipped, inserted
