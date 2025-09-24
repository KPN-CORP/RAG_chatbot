# ingest_final_audited.py
"""
Final audited ingest script - LangChain (Gemini) for embeddings, pymilvus for collection & inserts.
- No unused variables.
- GEMINI_API_KEY support.
- Batching, dedupe, schema guarantee, no double-embedding.
- dry_run mode for local testing.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import os, logging, time

# Embeddings (LangChain wrapper for Gemini)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore

# pymilvus for vector storage
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def chunks_to_records(chunks: List[Dict[str, Any]]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Convert chunk dicts -> list of tuples: (chunk_id, text, checksum, metadata_dict)
    metadata_dict is intentionally small.
    """
    records = []
    for c in chunks:
        chunk_id = c.get("chunk_id")
        text = c.get("chunk_text") or c.get("text")
        checksum = c.get("checksum")
        if not chunk_id or not text or not checksum:
            raise ValueError("Each chunk must have chunk_id, chunk_text (or text), and checksum. Offending: %r" % (c,))
        metadata = {
            "chunk_id": chunk_id,
            "chunk_text": text,
            "source": c.get("source"),
            "page": c.get("page"),
            "element_type": c.get("element_type"),
            "table_id": c.get("table_id"),
            "row_id": c.get("row_id"),
            "list_id": c.get("list_id"),
        }
        records.append((chunk_id, text, checksum, metadata))
    return records


def ensure_collection_exists_with_schema(collection_name: str, dim: int,
                                        host: str, port: str,
                                        index_params: Optional[dict] = None,
                                        dry_run: bool = False) -> Optional[Collection]:
    """Create or validate collection schema with id, embedding, checksum, metadata."""
    if index_params is None:
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
    if dry_run:
        log.info("[dry_run] Would ensure collection '%s' with dim=%d", collection_name, dim)
        return None
    connections.connect(alias="default", host=host, port=port)
    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        vec_field = next((f for f in coll.schema.fields if f.dtype == DataType.FLOAT_VECTOR), None)
        if vec_field and vec_field.dim != dim:
            raise RuntimeError(f"Existing collection '{collection_name}' dim {vec_field.dim} != expected dim {dim}")
        coll.load()
        return coll
    # create new collection
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="checksum", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, description="RAG KB - chunks with Gemini embeddings")
    coll = Collection(name=collection_name, schema=schema)
    coll.create_index(field_name="embedding", index_params=index_params)
    coll.load()
    log.info("Created Milvus collection '%s' with dim=%d", collection_name, dim)
    return coll


def milvus_query_checksums(collection_name: str, checksums: List[str], host: str, port: str, dry_run: bool = False) -> Set[str]:
    """Return set of checksums found in collection. Safe: returns empty set on failure or missing collection."""
    if not checksums:
        return set()
    if dry_run:
        log.info("[dry_run] Would query Milvus collection '%s' for %d checksums", collection_name, len(checksums))
        return set()
    try:
        connections.connect(alias="default", host=host, port=port)
    except Exception as e:
        log.warning("Milvus connect failed: %s", e)
        return set()
    if not utility.has_collection(collection_name):
        return set()
    try:
        coll = Collection(collection_name)
        quoted = ", ".join([f'"{c}"' for c in checksums])
        expr = f'checksum in [{quoted}]'
        rows = coll.query(expr=expr, output_fields=["checksum"])
        found = set(r.get("checksum") for r in rows if r.get("checksum"))
        log.info("Found %d existing checksums in Milvus", len(found))
        return found
    except Exception as e:
        log.warning("Milvus checksum query failed: %s", e)
        return set()


def insert_with_pymilvus(coll: Optional[Collection], ids: List[str], vectors: List[List[float]], checksums: List[str], metadatas: List[Dict[str, Any]], dry_run: bool = False):
    """Insert vectors (aligned lists) into Milvus; if dry_run True, only log."""
    if dry_run:
        log.info("[dry_run] Would insert %d vectors into collection", len(ids))
        return
    if coll is None:
        raise RuntimeError("pymilvus Collection object required for insertion")
    coll.insert([ids, vectors, checksums, metadatas])
    coll.flush()
    coll.load()


def embed_batch_texts(texts: List[str], model: str, batch_size: int, api_key: Optional[str], dry_run: bool = False) -> List[List[float]]:
    """Embed texts using LangChain wrappers (Gemini). If dry_run True, return fake vectors."""
    if dry_run:
        # return deterministic fake vectors (dim=1536 for simulation)
        dim = 1536
        fake = [[0.001 * ((i + 1) % 10) for _ in range(dim)] for i, _ in enumerate(texts)]
        return fake
    # instantiate embeddings wrapper; wrapper typically reads env but we attempt safe fallbacks

    # try passing explicit key param names
    if api_key:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)  # type: ignore
        except TypeError:
            emb = GoogleGenerativeAIEmbeddings(model=model, api_key=api_key)  # type: ignore
    else:
        emb = GoogleGenerativeAIEmbeddings(model=model)
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vs = emb.embed_documents(batch)
        vectors.extend(vs)
    return vectors


def store_chunks_to_milvus(
    chunks: List[Dict[str, Any]],
    collection_name: str,
    milvus_host: str = "127.0.0.1",
    milvus_port: int = 19530,
    embedding_model: str = "models/embedding-001",
    embed_batch_size: int = 32,
    insert_batch_size: int = 64,
    dedupe_with_milvus: bool = True,
    api_key: Optional[str] = None,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Ingest pipeline:
      - dedupe (if collection exists)
      - embed in batches (Gemini via LangChain)
      - ensure collection schema with checksum
      - insert into Milvus via pymilvus
    Returns (skipped, inserted)
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not dry_run:
        log.warning("No GEMINI_API_KEY/GOOGLE_API_KEY found; embedding may fail unless your provider reads env.")

    if not chunks:
        return 0, 0

    records = chunks_to_records(chunks)

    # Check if collection exists (dry_run safe)
    coll_exists = False
    try:
        if not dry_run:
            connections.connect(alias="default", host=milvus_host, port=str(milvus_port))
            coll_exists = utility.has_collection(collection_name)
        else:
            coll_exists = False
    except Exception as e:
        log.warning("Milvus connection check failed: %s", e)
        coll_exists = False

    # Dedupe only if collection exists
    to_process = records
    skipped = 0
    if dedupe_with_milvus and coll_exists:
        checksums = [r[2] for r in records]
        existing = milvus_query_checksums(collection_name, checksums, milvus_host, str(milvus_port), dry_run=dry_run)
        if existing:
            new_records = []
            for rec in records:
                if rec[2] in existing:
                    skipped += 1
                else:
                    new_records.append(rec)
            to_process = new_records

    if not to_process:
        log.info("No records to process after dedupe.")
        return skipped, 0

    # Process in insert batches: embed then insert
    inserted = 0
    coll_obj: Optional[Collection] = None

    i = 0
    while i < len(to_process):
        batch = to_process[i:i+insert_batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        checksums = [r[2] for r in batch]
        metadatas = [r[3] for r in batch]

        # embed
        vectors = embed_batch_texts(texts, model=embedding_model, batch_size=embed_batch_size, api_key=api_key, dry_run=dry_run)
        if not vectors or len(vectors) != len(texts):
            raise RuntimeError("Embedding API produced unexpected result count")

        # ensure collection exists (first batch will create with correct dim)
        if coll_obj is None:
            dim = len(vectors[0])
            coll_obj = ensure_collection_exists_with_schema(collection_name, dim, host=milvus_host, port=str(milvus_port), dry_run=dry_run)

        # insert
        insert_with_pymilvus(coll_obj, ids, vectors, checksums, metadatas, dry_run=dry_run)
        inserted += len(ids)
        i += insert_batch_size

    log.info("Finished ingest: inserted=%d, skipped=%d", inserted, skipped)
    return skipped, inserted


