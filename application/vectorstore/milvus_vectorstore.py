"""Helpers to initialise the Milvus vector store used by the application."""

from __future__ import annotations

import os
from typing import Optional

from config import (
    collection_name,
    embedding_model_name,
    google_api_key_name,
    milvus_host,
    milvus_port,
)
from langchain_milvus import Milvus
from logging_config import get_logger
from models.embbedings import get_gemini_embeddings
from pymilvus import connections
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


def _resolve_api_key(explicit_key: Optional[str]) -> Optional[str]:
    """Return the API key to use for embeddings, preferring explicit overrides."""

    if explicit_key:
        return explicit_key
    env_value = os.environ.get(google_api_key_name)
    if env_value:
        return env_value
    logger.warning(
        "Gemini API key not found in environment",
        extra={"env_var": google_api_key_name},
    )
    return None


def get_vectorstore() -> Milvus:
    """Initialise and return the Milvus vector store configured for Gemini embeddings."""

    logger.info(
        "Connecting to Milvus",
        extra={"host": milvus_host, "port": milvus_port, "collection": collection_name},
    )
    connections.connect(alias="default", host=milvus_host, port=milvus_port)

    api_key = _resolve_api_key(os.environ.get(google_api_key_name))
    embeddings = get_gemini_embeddings(embedding_model_name, api_key)

    vectorstore_kwargs = {
        "embedding_function": embeddings,
        "collection_name": collection_name,
        "connection_args": {"host": milvus_host, "port": str(milvus_port)},
        "text_field": "text",
        "enable_dynamic_field": True,
    }

    try:
        logger.debug("Attempting to create Milvus vector store with vector_field parameter")
        store = Milvus(vector_field="embedding", **vectorstore_kwargs)
        logger.info("Initialised Milvus vector store", extra={"collection": collection_name})
        return store
    except TypeError as exc:
        logger.info(
            "Falling back to embedding_field parameter for Milvus vector store",
            extra={"error": str(exc)},
        )
        store = Milvus(embedding_field="embedding", **vectorstore_kwargs)
        logger.info(
            "Initialised Milvus vector store using embedding_field fallback",
            extra={"collection": collection_name},
        )
        return store
    except Exception:
        logger.exception("Failed to initialise Milvus vector store")
        raise
