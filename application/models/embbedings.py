"""Embedding model helpers used by the RAG pipeline."""

import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from logging_config import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

# Embeddings (Gemini via LangChain)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception as exc:  # pragma: no cover - fallback for older LangChain versions
    logger.info(
        "Falling back to legacy GoogleGenerativeAIEmbeddings import",
        extra={"error": str(exc)},
    )
    from langchain.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore


def get_huggingface_embeddings(semantic_embedding_model_name: str) -> HuggingFaceEmbeddings:
    """Return a CPU-friendly HuggingFace embedding model configured for semantic search."""

    model_kwargs = {"device": "cpu"}  # Use 'cuda' for GPU
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=semantic_embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


def get_gemini_embeddings(
    embedding_model_name: str,
    api_key: Optional[str] = None,
    google_api_key_env: Optional[str] = None,
):
    """Return Google Gemini embeddings, attempting sensible API-key fallbacks."""

    api_key_eff = (
        api_key
        or (os.environ.get(google_api_key_env) if google_api_key_env else None)
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key_eff:
        logger.warning("No Gemini API key available for embeddings")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name, google_api_key=api_key_eff
        )  # type: ignore
    except TypeError as exc:
        logger.info(
            "Falling back to api_key argument when initialising embeddings",
            extra={"error": str(exc)},
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name, api_key=api_key_eff
        )  # type: ignore

    return embeddings
