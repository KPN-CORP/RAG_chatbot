# RAG/schemas/api.py

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Import the Source model from your other schema file
# The '.' makes it a relative import within the 'schemas' package
from .rag_outputs import Source
from config import BUSINESS_UNITS


# Build a str-based enum from the configured business units so the OpenAPI/Swagger
# UI renders a dropdown of valid values (instead of a free-text box that lets users
# submit "string" and hit a 422). Member names are identifier-safe; the enum values
# remain the exact strings expected by the rest of the app.
BusinessUnit = Enum(  # type: ignore[misc]
    "BusinessUnit",
    {unit.replace(" ", "_"): unit for unit in BUSINESS_UNITS},
    type=str,
)


class CreateSessionResp(BaseModel):
    """Response model for the session creation endpoint."""

    session_id: str


class ChatReq(BaseModel):
    """Request body for the main chat endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "<paste the session_id from POST /session>",
                "question": "Berapa hari cuti tahunan karyawan?",
                "debug_retrieval": False,
            }
        }
    )

    session_id: str
    question: str
    debug_retrieval: bool = False
    # Optional scoping for per-business-unit collections in Milvus. Leave empty
    # to search the default collection. Validated automatically against the enum.
    business_unit: Optional[BusinessUnit] = Field(
        default=None,
        description=(
            "Optional. Scope the search to one business unit's Milvus collection. "
            "Leave null/omit to use the default collection. "
            f"Valid values: {', '.join(BUSINESS_UNITS)}."
        ),
    )


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
