"""Pydantic schema for Gemini table extraction outputs."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ExtractedTable(BaseModel):
    """Validated representation of a table parsed from Gemini responses."""

    rows: List[Dict[str, Optional[str]]] = Field(
        default_factory=list,
        description=(
            "List of row objects keyed by column headers. Values should preserve"
            " line breaks when present in the source table."
        ),
    )

    @field_validator("rows", mode="before")
    @classmethod
    def coerce_rows(cls, value):
        if value in (None, ""):
            return []
        return value
