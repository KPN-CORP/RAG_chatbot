"""Wrapper that sanitises retrieval results before feeding them into the LLM."""
# not used

import json
from typing import List, Any
try:
    from langchain.schema import Document, BaseRetriever
except Exception:
    from langchain.retrievers.base import BaseRetriever
    from langchain.schema import Document

from pydantic import PrivateAttr


def sanitize_docs(docs: List[Document]) -> List[Document]:
    """Coerce document page content to plain strings for downstream consumers."""

    sanitized: List[Document] = []
    for d in docs:
        pc = d.page_content
        if isinstance(pc, str):
            sanitized.append(d)
            continue

        new_text = None
        if isinstance(pc, dict):
            for key in ("text", "chunk_text", "content"):
                v = pc.get(key)
                if isinstance(v, str) and v.strip():
                    new_text = v
                    break
            if new_text is None and isinstance(pc.get("summary"), str):
                new_text = pc.get("summary")
        if new_text is None:
            try:
                new_text = json.dumps(pc, ensure_ascii=False)
            except Exception:
                new_text = str(pc)
        sanitized.append(Document(page_content=new_text, metadata=d.metadata))
    return sanitized

class SanitizingRetriever(BaseRetriever):
    """LangChain retriever wrapper that ensures string page content."""

    _base_retriever: Any = PrivateAttr()

    # attributes we will NOT forward to the underlying retriever to avoid recursion/special-handling
    _DO_NOT_FORWARD = {
        "__dict__", "__getstate__", "__setstate__", "__getattr__", "__getattribute__",
        "model_dump", "model_dump_json", "dict", "json", "schema", "schema_json",
        "__class__", "__repr__", "__str__", "__reduce__", "__reduce_ex__",
        "get_relevant_documents", "aget_relevant_documents"
    }

    def __init__(self, base_retriever: Any, **kwargs):
        """Store the wrapped retriever while preserving Pydantic compatibility."""

        super().__init__(**kwargs)
        self._base_retriever = base_retriever

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve documents synchronously and normalise their contents."""

        docs = self._base_retriever.get_relevant_documents(query, **kwargs)
        return sanitize_docs(docs)

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve documents asynchronously when supported and normalise contents."""

        if hasattr(self._base_retriever, "aget_relevant_documents"):
            docs = await self._base_retriever.aget_relevant_documents(query, **kwargs)
        else:
            docs = self._base_retriever.get_relevant_documents(query, **kwargs)
        return sanitize_docs(docs)

    def __getattr__(self, name: str):
        """Forward unknown attributes to the wrapped retriever while avoiding recursion."""

        # block forwarding of internal/pydantic/serialization attributes to avoid recursion
        if name in self._DO_NOT_FORWARD or name.startswith("_") or name.endswith("_json"):
            raise AttributeError(name)
        # else forward to underlying retriever
        return getattr(self._base_retriever, name)
