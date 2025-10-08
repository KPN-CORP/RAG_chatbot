# RAG/schemas/rag_outputs.py

from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Data model for a single source document."""

    filename: str = Field(description="The name of the source PDF file.")
    page: int = Field(description="The page number from the source file.")


class AnswerWithSources(BaseModel):
    """The final structured output of the RAG chain."""

    answer: str = Field(description="The concise answer to the user's question.")
    sources: List[Source] = Field(
        description="A list of source documents used to formulate the answer."
    )


def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Formats documents to include their content and metadata for the LLM.
    """
    formatted_strings = []
    for i, doc in enumerate(docs):
        # Safely get metadata, providing defaults if keys are missing
        filename = doc.metadata.get("filename", "N/A")
        page = doc.metadata.get("page", "N/A")

        # Create a detailed block for each source
        doc_string = (
            f"--- Source {i+1} ---\n"
            f"Filename: {filename}\n"
            f"Page: {page}\n"
            f"Content: {doc.page_content}"
        )
        formatted_strings.append(doc_string)

    if not formatted_strings:
        return "No relevant documents found."

    return "\n\n".join(formatted_strings)
