# RAG/schemas/api.py

from typing import List, Optional

from pydantic import BaseModel, validator

# Import the Source model from your other schema file
# The '.' makes it a relative import within the 'schemas' package
from .rag_outputs import Source
from config import BUSINESS_UNITS


class CreateSessionResp(BaseModel):
    """Response model for the session creation endpoint."""

    session_id: str


class ChatReq(BaseModel):
    """Request body for the main chat endpoint."""

    session_id: str
    question: str
    debug_retrieval: bool = False
    # Optional scoping for per-business-unit collections in Milvus.
    business_unit: Optional[str] = None

    @validator("business_unit")
    def validate_business_unit(cls, value: Optional[str]) -> Optional[str]:
        """Ensure the business unit matches the configured list."""
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned not in BUSINESS_UNITS:
            raise ValueError(
                f"business_unit must be one of: {', '.join(BUSINESS_UNITS)}"
            )
        return cleaned


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
