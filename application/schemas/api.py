# RAG/schemas/api.py

from __future__ import annotations

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
    """Response body for the main chat endpoint.

    The IRIS profile omits ``sources`` in its payload to satisfy a lightweight
    contract, while other profiles keep returning sources alongside the answer.
    ``not_found`` is always included to hint whether any citations were
    available.
    """

    answer: str
    sources: List[Source] | None = None
    not_found: bool = False
