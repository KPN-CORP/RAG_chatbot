"""Pydantic schema for Gemini table extraction outputs.

The Gemini call now returns *chunks* that already satisfy the "atomic cell" rule
so downstream ingest does not need to rebuild per-cell chunks manually. Tables
are also grouped with their table index to keep multi-table pages aligned. The
legacy ``rows`` payload remains for backward compatibility and is populated from
``tables`` when absent.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator


class ExtractedChunk(BaseModel):
    """Atomic chunk returned by Gemini.

    ``table_cell`` chunks must include ``table_index`` and ``column_header`` so
    the ingest pipeline can preserve header context, while ``narrative`` chunks
    simply carry the text Gemini segmented from surrounding prose.
    """

    chunk_type: Literal["table_cell", "narrative", "table_note"] = Field(
        description=(
            "Type of chunk. ``table_cell`` is a single cell with headers;"
            " ``table_note`` is a footnote or merged note; ``narrative`` is"
            " plain text outside tables."
        )
    )
    text: str = Field(
        ..., description="Raw text for the chunk, preserving line breaks."
    )
    chunk_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional fully-enriched chunk string (title + headers + notes)"
            " ready for ingestion."
        ),
    )
    title: Optional[str] = Field(
        default=None,
        description="Nearest title or section heading associated with the chunk.",
    )
    title_chain: Optional[str] = Field(
        default=None,
        description="Full hierarchy of titles from the page for enrichment.",
    )
    page_notes: Optional[str] = Field(
        default=None,
        description="Notes/footnotes that apply to the page as a whole.",
    )
    table_index: Optional[int] = Field(
        default=None,
        description="0-based order of the table on the page when chunk_type is table-related.",
    )
    row_index: Optional[int] = Field(
        default=None, description="Row index within the table for table-related chunks."
    )
    column_header: Optional[str] = Field(
        default=None,
        description="Full header path for the cell (concatenate multi-level headers with ' - ').",
    )
    row_header: Optional[str] = Field(
        default=None,
        description="Identifier for the row if the table includes a row header column.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Inline notes/comments associated with the cell, if present.",
    )

    @field_validator("text", "chunk_text", "title", "title_chain", "page_notes", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Optional[str]) -> Optional[str]:
        """Ensure text-like fields are always strings to satisfy downstream parsing."""

        if value is None:
            return None
        return str(value)


class GeminiChunkList(RootModel[List[str]]):
    """Simplified Gemini output when only chunk strings are returned."""

    root: List[str]

    @property
    def chunks(self) -> List[str]:
        """Expose the root list using the same name expected by callers."""

        return self.root


class GeminiTable(BaseModel):
    """Structured table representation scoped to a single page."""

    table_index: int = Field(
        description="0-based order of the table on the page (top-to-bottom, left-to-right)."
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional table title/caption detected by Gemini, if visible on the page.",
    )
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


class ExtractedTable(BaseModel):
    """Validated representation of a Gemini response for a PDF page."""

    rows: List[Dict[str, Optional[str]]] = Field(
        default_factory=list,
        description=(
            "Flattened list of row objects across all detected tables. Kept for"
            " backward compatibility; populated from ``tables`` when absent."
        ),
    )
    tables: List[GeminiTable] = Field(
        default_factory=list,
        description="Per-table rows with ordering metadata for pages containing multiple tables.",
    )
    chunks: List[ExtractedChunk] = Field(
        default_factory=list,
        description=(
            "Atomic chunks (table cells, notes, and narrative text) Gemini"
            " segmented from the page following the atomic-cell rule."
        ),
    )

    @field_validator("rows", mode="before")
    @classmethod
    def coerce_rows(cls, value):
        if value in (None, ""):
            return []
        return value

    @model_validator(mode="after")
    def populate_legacy_rows(self):
        """Fill ``rows`` from ``tables`` so legacy callers remain compatible."""

        if not self.rows and self.tables:
            aggregated: List[Dict[str, Optional[str]]] = []
            for table in sorted(self.tables, key=lambda t: t.table_index):
                aggregated.extend(table.rows)
            self.rows = aggregated
        return self
