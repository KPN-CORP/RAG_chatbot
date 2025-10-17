# langchain_milvus_ingest.py
"""
End-to-end ingest with Gemini embeddings + Milvus storage.
- Universal semantic chunking imported from ingest.chunking
- Dedupe by checksum
- Batching, dry_run, schema guarantee
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from config import google_api_key_name, semantic_embedding_model_name, embedding_model_name

# Chunking
from ingest.chunking import build_chunks_from_pages

# Embeddings (Gemini via LangChain)
from models.embbedings import get_gemini_embeddings, get_huggingface_embeddings
from modules.utils import resolve_embedding_api_key

# Milvus
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


chunking_embeddings = get_huggingface_embeddings(semantic_embedding_model_name)
gemini_embeddings = get_gemini_embeddings(embedding_model_name, api_key=resolve_embedding_api_key())

# -----------------------------
# Records shaping (unchanged)
# -----------------------------


def chunks_to_records(
    chunks: List[Dict[str, Any]],
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Convert chunk dicts -> tuples (id, text, checksum, metadata).
    """
    out: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for c in chunks:
        cid = c.get("chunk_id")
        text = c.get("chunk_text") or c.get("text")
        checksum = c.get("checksum")
        if not cid or not text or not checksum:
            raise ValueError(f"Chunk missing required fields: {c}")
        md = {
            "chunk_id": cid,
            "chunk_text": text,
            "source": c.get("source"),
            "page": c.get("page"),
            "element_type": c.get("element_type"),
            "table_id": c.get("table_id"),
            "row_id": c.get("row_id"),
            "list_id": c.get("list_id"),
            "breadcrumb": c.get("breadcrumb"),
            "col_name": c.get("col_name"),
            "value_num": c.get("value_num"),
        }
        out.append((cid, text, checksum, md))
    return out


# -----------------------------
# Milvus helpers
# -----------------------------


def ensure_collection_exists_with_schema(
    collection_name: str,
    dim: int,
    host: str,
    port: str,
    index_params: Optional[dict] = None,
    dry_run: bool = False,
    collection_description: str = "",
) -> Optional[Collection]:
    if index_params is None:
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
    if dry_run:
        log.info("[dry_run] Would ensure collection '%s' (dim=%d)", collection_name, dim)
        return None
    connections.connect(alias="default", host=host, port=port)
    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        vec_field = next((f for f in coll.schema.fields if f.dtype == DataType.FLOAT_VECTOR), None)
        if vec_field and vec_field.dim != dim:
            raise RuntimeError(
                f"Existing collection '{collection_name}' dim {vec_field.dim} != expected {dim}"
            )
        coll.load()
        return coll

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="checksum", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields, description=collection_description)
    coll = Collection(name=collection_name, schema=schema)
    coll.create_index(field_name="embedding", index_params=index_params)
    coll.load()
    log.info("Created collection '%s' (dim=%d)", collection_name, dim)
    return coll


def milvus_query_checksums(
    collection_name: str,
    checksums: List[str],
    host: str,
    port: str,
    dry_run: bool = False,
) -> Set[str]:
    if not checksums:
        return set()
    if dry_run:
        log.info(
            "[dry_run] Would query checksums for '%s' (%d items)",
            collection_name,
            len(checksums),
        )
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
        expr = f"checksum in [{quoted}]"
        rows = coll.query(expr=expr, output_fields=["checksum"])
        return set(r.get("checksum") for r in rows if r.get("checksum"))
    except Exception as e:
        log.warning("Milvus checksum query failed: %s", e)
        return set()


def insert_with_pymilvus(
    coll: Optional[Collection],
    ids: List[str],
    vectors: List[List[float]],
    checksums: List[str],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    dry_run: bool = False,
):
    if dry_run:
        log.info("[dry_run] Would insert %d vectors", len(ids))
        return
    if coll is None:
        raise RuntimeError("Collection object required for insertion")
    coll.insert([ids, vectors, checksums, texts, metadatas])
    coll.flush()
    coll.load()


# -----------------------------
# Embeddings
# -----------------------------


def embed_batch_texts(
    texts: List[str],
    batch_size: int,
    dry_run: bool = False,
) -> List[List[float]]:
    if dry_run:
        dim = 1536
        return [[0.001 * ((i + 1) % 10) for _ in range(dim)] for i, _ in enumerate(texts)]

    emb = gemini_embeddings
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vs = emb.embed_documents(batch)
        out.extend(vs)
    return out


# -----------------------------
# Storage pipeline
# -----------------------------


def store_chunks_to_milvus(
    chunks: List[Dict[str, Any]],
    collection_name: str,
    milvus_host: str = "127.0.0.1",
    milvus_port: int = 19530,
    embed_batch_size: int = 32,
    insert_batch_size: int = 64,
    dedupe_with_milvus: bool = True,
    dry_run: bool = False,
    collection_description: str = "",
) -> Tuple[int, int]:
    """
    Dedupe by checksum → batch-embed → ensure schema → insert.
    Returns (skipped, inserted).
    """

    if not chunks:
        return 0, 0

    records = chunks_to_records(chunks)

    # Collection existence (for dedupe)
    coll_exists = False
    try:
        if not dry_run:
            connections.connect(alias="default", host=milvus_host, port=str(milvus_port))
            coll_exists = utility.has_collection(collection_name)
    except Exception as e:
        log.warning("Milvus connection check failed: %s", e)

    # Dedupe
    skipped = 0
    to_process = records
    if dedupe_with_milvus and coll_exists:
        checks = [r[2] for r in records]
        existing = milvus_query_checksums(
            collection_name, checks, milvus_host, str(milvus_port), dry_run=dry_run
        )
        if existing:
            filtered = []
            for rec in records:
                if rec[2] in existing:
                    skipped += 1
                else:
                    filtered.append(rec)
            to_process = filtered

    if not to_process:
        log.info("No records to process after dedupe.")
        return skipped, 0

    # Insert loop
    inserted = 0
    coll_obj: Optional[Collection] = None
    i = 0
    while i < len(to_process):
        batch = to_process[i : i + insert_batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        checksums = [r[2] for r in batch]
        metadatas = []
        for r in batch:
            md = r[3].copy()
            md.pop("chunk_text", None)
            md.pop("text", None)
            metadatas.append(md)

        vectors = embed_batch_texts(
            texts,
            batch_size=embed_batch_size,
            dry_run=dry_run,
        )
        if not vectors or len(vectors) != len(texts):
            raise RuntimeError("Embedding API returned unexpected result count")

        if coll_obj is None:
            dim = len(vectors[0])
            coll_obj = ensure_collection_exists_with_schema(
                collection_name,
                dim,
                host=milvus_host,
                port=str(milvus_port),
                dry_run=dry_run,
                collection_description=collection_description
            )

        insert_with_pymilvus(coll_obj, ids, vectors, checksums, texts, metadatas, dry_run=dry_run)
        inserted += len(ids)
        i += insert_batch_size

    log.info("Finished ingest: inserted=%d, skipped=%d", inserted, skipped)
    return skipped, inserted


# -----------------------------
# Public: End-to-end ingest
# -----------------------------


def ingest_pages_to_milvus(
    pages: List[Dict[str, Any]],
    doc_id: str,
    source: str,
    collection_name: str,
    collection_description: str = "",
    milvus_host: str = "127.0.0.1",
    milvus_port: int = 19530,
    embed_batch_size: int = 32,
    insert_batch_size: int = 64,
    enable_semantic_fallback: bool = True,
    sim_threshold: float = 0.82,
    use_percentile: bool = False,
    percentile: int = 90,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Build chunks with universal semantic grouping + (optional) long-text fallback,
    then embed & store in Milvus.
    """
    # Embeddings object reused for grouping + fallback splitting. Use huggingface for cost.
    chunks = build_chunks_from_pages(
        pages=pages,
        doc_id=doc_id,
        source=source,
        embeddings_for_semantic=chunking_embeddings,
        enable_semantic_fallback=enable_semantic_fallback,
        semantic_fallback_max_chars=1000,
        sim_threshold=sim_threshold,
        use_percentile=use_percentile,
        percentile=percentile,
    )

    return store_chunks_to_milvus(
        chunks=chunks,
        collection_name=collection_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        embed_batch_size=embed_batch_size,
        insert_batch_size=insert_batch_size,
        dedupe_with_milvus=True,
        dry_run=dry_run,
        collection_description=collection_description
    )
