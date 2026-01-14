"""Hook functions used by the folder watcher to sync Milvus.

The watcher delegates all database work to these functions so the file-system
responsibilities stay focused. If your Milvus schema differs from the default
one defined in ``ingest/langchain_milvus_ingest.py`` (id, embedding, checksum,
text, metadata), adjust ``SOURCE_METADATA_KEY`` or the filtering logic below.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from config_loader import load_config
from ingest.langchain_milvus_ingest import ingest_pages_to_milvus, ingest_text_document
from ingest.pdf_with_tables import extract_pdf_content
from logging_config import get_logger
from modules.kb_status import update_kb_status
from modules.utils import _build_document_identifier
from pymilvus import Collection, connections, utility

logger = get_logger(__name__)

# The metadata key that stores the human-readable filename/source.
SOURCE_METADATA_KEY: Final[str] = "source"


def _connect_to_milvus(config) -> None:
    """Establish a connection to Milvus using the configured host/port."""

    connections.connect(alias="default", host=config.milvus_host, port=str(config.milvus_port))


def _unwrap_source(record: dict) -> str:
    """Extract the human-friendly source label from a Milvus record."""

    metadata = record.get("metadata") or {}
    if isinstance(metadata, dict):
        return str(metadata.get(SOURCE_METADATA_KEY) or metadata.get("filename") or "<unknown>")
    return "<unknown>"


def _log_current_documents(collection: Collection, *, sample_limit: int = 10) -> None:
    """Log a snapshot of document sources currently present in Milvus."""

    try:
        records = collection.query(
            expr="id != ''",
            output_fields=["metadata"],
            limit=sample_limit,
        )
    except Exception as exc:  # pragma: no cover - defensive logging aid
        logger.warning(
            "Could not fetch sample records for logging",
            extra={"collection": collection.name, "error": str(exc)},
        )
        return

    sources = Counter(_unwrap_source(record) for record in records)
    logger.info(
        "Knowledge base sample",
        extra={
            "collection": collection.name,
            "entities": collection.num_entities,
            "sources": dict(sources),
        },
    )


def delete_file_from_milvus(
    file_path: str,
    *,
    config=None,
    collection_name: str | None = None,
) -> None:
    """Delete all vectors associated with ``file_path`` from Milvus."""

    config = config or load_config()
    collection = collection_name or config.collection_name
    source_name = Path(file_path).name

    _connect_to_milvus(config)
    if not utility.has_collection(collection):
        logger.info(
            "Milvus collection missing; skipping delete",
            extra={"collection": collection, "file": source_name},
        )
        return

    milvus_collection = Collection(collection)
    milvus_collection.load()
    expr = f'metadata["{SOURCE_METADATA_KEY}"] == "{source_name}"'
    try:
        result = milvus_collection.delete(expr)
        logger.info(
            "Deleted vectors for file",
            extra={
                "file": source_name,
                "collection": collection,
                "deleted_count": getattr(result, "delete_count", getattr(result, "delete_cnt", None)),
            },
        )
    except Exception as exc:  # pragma: no cover - defensive logging aid
        logger.warning(
            "Failed to delete vectors for file",
            extra={"file": source_name, "collection": collection, "error": str(exc)},
        )
        return

    _log_current_documents(milvus_collection)


def upsert_file_to_milvus(
    file_path: str,
    *,
    config=None,
    collection_name: str | None = None,
) -> None:
    """Upsert a file into Milvus and mirror its job status in the KB database."""

    config = config or load_config()
    collection = collection_name or config.collection_name
    path = Path(file_path)
    ext = path.suffix.lower()
    if not path.exists():
        logger.info("File disappeared before ingest", extra={"file": str(path)})
        return

    data = path.read_bytes()
    doc_id = _build_document_identifier(path.name or "document", data)
    logger.info(
        "Preparing Milvus upsert",
        extra={"file": str(path), "doc_id": doc_id, "ext": ext, "collection": collection},
    )

    # Mark the job as in-flight before touching Milvus.
    try:
        update_kb_status(str(path), status="processing", config=config)
    except Exception as exc:  # pragma: no cover - defensive logging aid
        logger.exception(
            "Failed to update processing status to the database",
            extra={"file": str(path), "doc_id": doc_id, "ext": ext, "error": str(exc)},
        )
        

    # Step 1: remove previous vectors for this source so updates stay in sync.
    delete_file_from_milvus(file_path, config=config, collection_name=collection)

    # Step 2: insert new vectors using the existing ingestion pipeline.
    try:
        if ext == ".pdf":
            pages = extract_pdf_content(str(path))
            skipped, inserted = ingest_pages_to_milvus(
                pages=pages,
                doc_id=doc_id,
                source=path.name,
                collection_name=collection,
                config=config,
                milvus_host=config.milvus_host,
                milvus_port=config.milvus_port,
                collection_description=config.collection_description,
            )
        elif ext == ".txt":
            text_content = data.decode("utf-8-sig", errors="replace")
            skipped, inserted = ingest_text_document(
                text=text_content,
                doc_id=doc_id,
                source=path.name,
                collection_name=collection,
                config=config,
                collection_description=config.collection_description,
                milvus_host=config.milvus_host,
                milvus_port=config.milvus_port,
            )
        else:
            logger.warning(
                "Unsupported file type for auto-ingest",
                extra={"file": str(path), "ext": ext},
            )
            return
    except Exception as exc:  # pragma: no cover - defensive logging aid
        logger.exception(
            "Failed to ingest file",
            extra={"file": str(path), "doc_id": doc_id, "ext": ext, "error": str(exc)},
        )
        return

    logger.info(
        "Upsert complete",
        extra={
            "file": str(path),
            "inserted": inserted,
            "skipped": skipped,
            "collection": collection,
        },
    )

    # Mark the job as done after the Milvus upsert succeeds.
    try:
        update_kb_status(str(path), status="done", config=config)
    except Exception as exc:  # pragma: no cover - defensive logging aid
        logger.exception(
            "Failed to update done status the database",
            extra={"file": str(path), "doc_id": doc_id, "ext": ext, "error": str(exc)},
        )

    # Log a quick snapshot of the knowledge base after the update.
    _connect_to_milvus(config)
    milvus_collection = Collection(collection)
    milvus_collection.load()
    _log_current_documents(milvus_collection)


__all__ = ["upsert_file_to_milvus", "delete_file_from_milvus", "SOURCE_METADATA_KEY"]
