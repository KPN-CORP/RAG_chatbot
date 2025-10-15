# RAG/schemas/rag_outputs.py

from pathlib import Path
from typing import Optional, List

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    """Data model for a single source document."""

    filename: str = Field(description="The name of the source PDF file.")
    page: Optional[int] = Field(description="The page number from the source file.")
    # Optional: coerce common bad values
    @field_validator("page", mode="before")
    @classmethod
    def coerce_page(cls, v):
        if v in (None, "", "N/A", "n/a", "na", "-"):
            return None
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v
    

class AnswerWithSources(BaseModel):
    """The final structured output of the RAG chain."""

    answer: str = Field(description="The concise answer to the user's question.")
    sources: List[Source] = Field(
        description="A list of source documents used to formulate the answer."
    )


def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Format retrieved docs into a readable context string, showing real filename/page
    even when your metadata is nested under 'metadata'.
    """

    if not docs:
        return "No relevant documents found."

    def unwrap_meta(meta: dict) -> dict:
        # Flatten one level if there's an inner "metadata" dict
        if not meta:
            return {}
        inner = meta.get("metadata") or {}
        flat = {k: v for k, v in meta.items() if k != "metadata"}
        flat.update(inner)
        return flat

    parts = []
    for i, d in enumerate(docs, 1):
        m = unwrap_meta(d.metadata)

        # Filename: prefer explicit filename; else basename of source/path/file_name
        src = m.get("filename") or m.get("source") or m.get("path") or m.get("file_name")
        filename = Path(src).name if isinstance(src, str) and src else "N/A"

        # Page: accept common variants
        page = m.get("page") or m.get("page_number") or m.get("pagenum")
        page_str = str(page) if page is not None else "N/A"

        breadcrumb = m.get("breadcrumb")
        element_type = m.get("element_type")

        block = [
            f"--- Source {i} ---",
            f"Filename: {filename}",
            f"Page: {page_str}",
        ]
        if breadcrumb:
            block.append(f"Breadcrumb: {breadcrumb}")
        if element_type:
            block.append(f"Type: {element_type}")
        block.append(f"Content: {d.page_content}")
        parts.append("\n".join(block))

    return "\n\n".join(parts)
