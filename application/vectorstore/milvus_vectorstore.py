"""Helpers to initialise the Milvus vector store used by the application."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_milvus import Milvus
from logging_config import get_logger
from models.embbedings import get_gemini_embeddings
from modules.utils import resolve_embedding_api_key
from pymilvus import Collection, connections
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


def _resolve_api_key(google_api_key_name: str, explicit_key: Optional[str]) -> Optional[str]:
    """Return the API key to use for embeddings, preferring explicit overrides."""

    if explicit_key:
        return explicit_key
    env_value = resolve_embedding_api_key(google_api_key_name)
    if env_value:
        return env_value
    logger.warning(
        "Gemini API key not found in environment",
        extra={"env_var": google_api_key_name},
    )
    return None

@lru_cache(maxsize=None)
def _build_vectorstore(
    # config_module_name: str,
    collection_name: str,
    milvus_host: str,
    milvus_port: str,
    embedding_model_name: str,
    google_api_key_name: str,
) -> Milvus:
    """Cache and return a Milvus vector store for the given configuration."""

    logger.info(
        "Connecting to Milvus",
        extra={"host": milvus_host, "port": milvus_port, "collection": collection_name},
    )
    connections.connect(alias="default", host=milvus_host, port=milvus_port)

    api_key = _resolve_api_key(google_api_key_name, os.environ.get(google_api_key_name))
    embeddings = get_gemini_embeddings(
        embedding_model_name,
        api_key,
        google_api_key_env=google_api_key_name,
    )

    vectorstore_kwargs = {
        "embedding_function": embeddings,
        "collection_name": collection_name,
        "connection_args": {"host": milvus_host, "port": str(milvus_port)},
        "text_field": "text",
        "enable_dynamic_field": True,
        "consistency_level": "Strong",
    }

    try:
        logger.debug("Attempting to create Milvus vector store with vector_field parameter")
        store = Milvus(vector_field="embedding", **vectorstore_kwargs)
        # Force-load to avoid “exists but returns empty”
        col = getattr(store, "col", None) or getattr(store, "_collection", None) or Collection(collection_name)
        col.load()
        logger.info("Initialised Milvus vector store", extra={"collection": collection_name})
        return store
    except TypeError as exc:
        logger.info(
            "Falling back to embedding_field parameter for Milvus vector store",
            extra={"error": str(exc)},
        )
        store = Milvus(embedding_field="embedding", **vectorstore_kwargs)
        # Force-load to avoid “exists but returns empty”
        col = getattr(store, "col", None) or getattr(store, "_collection", None) or Collection(collection_name)
        col.load()
        logger.info(
            "Initialised Milvus vector store using embedding_field fallback",
            extra={"collection": collection_name},
        )
        return store
    except Exception:
        logger.exception("Failed to initialise Milvus vector store")
        raise


def clear_vectorstore_cache() -> None:
    """Clear the cached Milvus vector stores."""

    _build_vectorstore.cache_clear()


def collection_exists(config, collection_name: str) -> bool:
    """Return True if the given collection exists in Milvus.

    Per-business-unit collections (``{collection_name}_{unit}``) are only created
    once documents are ingested for that unit, so querying a unit with no data
    would otherwise raise a low-level Milvus error. Callers use this to respond
    gracefully instead of surfacing a 500.
    """

    from pymilvus import connections, utility

    try:
        connections.connect(
            alias="default", host=config.milvus_host, port=str(config.milvus_port)
        )
        return utility.has_collection(collection_name, using="default")
    except Exception:
        logger.exception(
            "Failed to check collection existence",
            extra={"collection": collection_name},
        )
        return False


def get_vectorstore(config, *, collection_name: Optional[str] = None) -> Milvus:
    """Initialise and return the Milvus vector store for the provided config."""

    return _build_vectorstore(
        collection_name or config.collection_name,
        config.milvus_host,
        str(config.milvus_port),
        config.embedding_model_name,
        config.google_api_key_name,
    )


__all__ = ["clear_vectorstore_cache", "get_vectorstore"]
