# RAG/schemas/api.py

from typing import List

from pydantic import BaseModel

# Import the Source model from your other schema file
# The '.' makes it a relative import within the 'schemas' package
from .rag_outputs import Source


class CreateSessionResp(BaseModel):
    """Response model for the session creation endpoint."""

    session_id: str


class ChatReq(BaseModel):
    """Request body for the main chat endpoint."""

    session_id: str
    question: str
    debug_retrieval: bool = False


class ChatResp(BaseModel):
    """Response body for the main chat endpoint."""

    answer: str
    sources: List[Source]  # <-- More robust and type-safe!
