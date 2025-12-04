"""Image-based table extraction helper using pdf2image and Gemini.

This module provides a single entry point, :func:`extract_table_with_gemini`,
that can be used when the regular PDF table detector produces messy structures
(e.g., highly nested or multi-span headers). It converts a single PDF page to a
PNG image via ``pdf2image`` and sends that image to Gemini with a clear
extraction prompt so the model can return structured JSON for the table **and**
ready-made atomic chunks (one cell per chunk, plus narrative).

Unlike the initial prototype, this version leverages LangChain's
``ChatGoogleGenerativeAI`` together with ``PydanticOutputParser`` so the Gemini
response is validated against a schema before returning it to callers. This
keeps the ingest pipeline consistent with the rest of the app (see
``application/app.py`` and ``schemas/rag_outputs.py``).

New behavior: the Gemini call is no longer treated as a "last resort" for only
the most complex layouts. When a page contains a table, the entire page image
is rendered and sent to Gemini so every grid — regardless of header depth — is
normalized into header-aware, cell-level chunks. The previous conditional
fallback is preserved below as documentation for reference.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config_loader import load_config
from logging_config import get_logger
from schemas.table_extraction import ExtractedTable

logger = get_logger(__name__)

EXAMPLE_OUTPUT_FORMAT = """{
  "tables": [
    {
      "table_index": 0,
      "title": "Optional caption on the page",
      "rows": [
        {
          "Row": "Row 1 label",
          "Header 1": "row 1 value",
          "Header 2 - Sub A": "row 1 nested detail",
          "Notes": "line 1\\nline 2"
        },
        {
          "Row": "Row 2 label",
          "Header 1": "row 2 value",
          "Header 2 - Sub A": "row 2 nested detail",
          "Notes": null
        }
      ]
    }
  ],
  "chunks": [
    {
      "chunk_type": "table_cell",
      "table_index": 0,
      "row_index": 0,
      "row_header": "Row 1 label",
      "column_header": "Header 1",
      "text": "row 1 value",
      "notes": "line 1\\nline 2"
    },
    {"chunk_type": "narrative", "text": "Narrative paragraph outside any table."}
  ]
}"""


DEFAULT_EXTRACTION_PROMPT = (
    "You are a data extraction assistant. Read the provided full page image and "
    "return JSON that matches the schema instructions exactly. Identify *all* "
    "tables on the page (not just complex ones) and keep multi-level headers "
    "concatenated with ' - ' in the same key. For each table, return its rows "
    "under `tables` with the correct `table_index` order (top-to-bottom, left-to-"
    "right on the page). Also segment the entire page into atomic chunks: every "
    "table cell is one `table_cell` chunk with its `table_index`, `row_index`, "
    "`column_header`, row identifier, and any notes/comments; all non-table text "
    "becomes `narrative` chunks. Preserve line breaks inside cell text and notes. "
    "If a cell embeds another grid, flatten it into the cell text with line "
    "breaks. Do not use null for blank cells or notes—return an empty string "
    "instead so every cell produces a chunk. Follow this example shape and do "
    "not add any text before or after the JSON object:\n"
    f"{EXAMPLE_OUTPUT_FORMAT}"
)


def _sanitize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce Gemini rows into a chunking-friendly format.

    The downstream pipeline expects a list of dictionaries with string keys and
    string-or-``None`` values. This function defensively converts unexpected
    types so that chunking does not fail if Gemini returns numbers or other
    primitives.
    """

    sanitized: List[Dict[str, Optional[str]]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        clean_row: Dict[str, Optional[str]] = {}
        for key, value in row.items():
            clean_key = str(key) if key is not None else ""
            clean_value: Optional[str]
            if value is None:
                clean_value = None
            elif isinstance(value, (dict, list)):
                # Nested structures are stringified as JSON so downstream chunking
                # continues to see plain strings.
                clean_value = json.dumps(value, ensure_ascii=False)
            else:
                clean_value = str(value)
            clean_row[clean_key] = clean_value
        if clean_row:
            sanitized.append(clean_row)
    return sanitized


def _sanitize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize Gemini chunks into serializable dictionaries."""

    cleaned: List[Dict[str, Any]] = []
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        normalized: Dict[str, Any] = {}
        for key in (
            "chunk_type",
            "text",
            "table_index",
            "row_index",
            "column_header",
            "row_header",
            "notes",
            "title",
        ):
            if key in chunk:
                value = chunk[key]
                if isinstance(value, (dict, list)):
                    normalized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    normalized[key] = "" if value is None else value
        if "text" in normalized:
            # Preserve empty-string cells so that blank grid entries still produce
            # atomic chunks.
            cleaned.append(normalized)
    return cleaned


def _rows_to_cell_chunks(
    rows: List[Dict[str, Any]],
    *,
    doc_id: Optional[str],
    source: Optional[str],
    page_number: int,
    table_index: int = 0,
    table_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert Gemini rows into the atomic cell-chunk format used by ingest.

    The chunking layer expects one chunk per cell with the headers preserved in the
    text payload. By reusing :func:`table_cells_to_chunks` here, the Gemini output
    can feed directly into Milvus/embedding without re-parsing later.
    """

    if not doc_id or not source:
        return []

    # Local import avoids a top-level dependency for callers that only need rows.
    from application.ingest.chunking import table_cells_to_chunks

    return table_cells_to_chunks(
        table_dict_rows=rows,
        doc_id=doc_id,
        source=source,
        page=page_number,
        table_index=table_index,
        table_title=table_title,
    )


"""
Legacy manual chunking logic retained for documentation:
    The helper above converts Gemini row dictionaries into the ingest pipeline's
    cell-level chunk format. With the new Gemini prompt producing ready-made
    chunks, callers should prefer the ``chunks`` returned by the model, but this
    fallback remains for clarity and in case model outputs omit chunk metadata.
"""


def _extract_json_fragment(text: str) -> Optional[Any]:
    """Return the first JSON-like fragment from a text blob if present."""

    openers = [(text.find("["), "]"), (text.find("{"), "}")]
    starts = [(idx, closer) for idx, closer in openers if idx != -1]
    if not starts:
        return None

    start, closer = sorted(starts, key=lambda tup: tup[0])[0]
    window = text[start:]
    end = window.rfind(closer)
    if end == -1:
        return None

    candidate = window[: end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _page_to_image(pdf_path: Path, page_number: int, dpi: int = 300) -> Path:
    """Render a single PDF page to a PNG image and return the image path."""

    try:
        from pdf2image import convert_from_path
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pdf2image is required for image-based extraction") from exc

    images = convert_from_path(
        str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number
    )
    if not images:
        raise ValueError(f"Failed to render page {page_number} from {pdf_path}")

    image_path = pdf_path.with_suffix("")
    image_path = image_path.with_name(f"{image_path.name}_page{page_number}.png")
    images[0].save(image_path, format="PNG")
    logger.debug("Rendered %s page %s to %s", pdf_path.name, page_number, image_path)
    return image_path


def extract_table_with_gemini(
    pdf_path: str,
    page_number: int,
    *,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    dpi: int = 300,
    return_cell_chunks: bool = True,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    table_index: int = 0,
    table_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract a complex table by rendering the PDF page and delegating to Gemini.

    The Gemini call is executed through LangChain with a :class:`PydanticOutputParser`
    so the returned structure is guaranteed to match the ``ExtractedTable``
    schema (a list of row dictionaries). This mirrors the pattern used in
    :mod:`application.app` for chat responses. When ``return_cell_chunks`` is
    ``True`` and ``doc_id``/``source`` are provided, the rows are immediately
    converted into the atomic cell-chunk format to simplify downstream ingest.
    """

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    image_path = _page_to_image(pdf_file, page_number, dpi=dpi)
    prompt_text = prompt or DEFAULT_EXTRACTION_PROMPT

    try:
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        config = load_config()
        parser = PydanticOutputParser(pydantic_object=ExtractedTable)

        system_instructions = (
            "You are a precise data extraction assistant. Return only JSON that "
            "matches the provided schema. Keep column headers exactly as they "
            "appear and preserve line breaks inside cell values."
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_instructions}\n\nSchema instructions:\n{{format_instructions}}",
                ),
                (
                    "human",
                    [
                        {"type": "text", "text": "{user_prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/png;base64,{image_b64}",
                        },
                    ],
                ),
            ]
        )

        chat_model = ChatGoogleGenerativeAI(
            model=model_name or getattr(config, "gemini_chat_model", None),
            google_api_key=api_key
            or os.environ.get(getattr(config, "google_api_key_name", "GEMINI_API_KEY")),
            temperature=0,
        )

        chain = prompt_template.partial(
            format_instructions=parser.get_format_instructions()
        ) | chat_model | parser

        try:
            parsed: ExtractedTable = chain.invoke(
                {"user_prompt": prompt_text, "image_b64": encoded_image}
            )
            rows = _sanitize_rows(parsed.rows)
            chunks = _sanitize_chunks([c.model_dump() for c in parsed.chunks])
            if return_cell_chunks and not chunks:
                # Preserve legacy behavior if the model did not supply chunks.
                chunks = _rows_to_cell_chunks(
                    rows,
                    doc_id=doc_id,
                    source=source,
                    page_number=page_number,
                    table_index=table_index,
                    table_title=table_title,
                )
            return {
                "page": page_number,
                "image_path": str(image_path),
                "rows": rows,
                "tables": [tbl.model_dump() for tbl in parsed.tables],
                "chunks": chunks,
                "model_response": parsed,
                "cell_chunks": chunks if return_cell_chunks else [],
            }
        except Exception as exc:  # pragma: no cover - network/model variability
            logger.warning(
                "Gemini chain parsing failed; attempting raw response recovery",
                extra={"page": page_number},
                exc_info=exc,
            )

        # Fallback: request the model output directly and try to coerce JSON even if
        # extra text surrounds it. This prevents ingestion from breaking when the
        # model returns prose plus JSON.
        messages = prompt_template.format_messages(
            format_instructions=parser.get_format_instructions(),
            user_prompt=prompt_text,
            image_b64=encoded_image,
        )
        raw_response = chat_model.invoke(messages)
        raw_text = getattr(raw_response, "content", None) or getattr(
            raw_response, "text", None
        )
        if raw_text is None:
            raw_text = str(raw_response)

        fallback_rows: List[Dict[str, Any]] = []
        fallback_chunks: List[Dict[str, Any]] = []
        fallback_tables: List[Dict[str, Any]] = []

        if isinstance(raw_text, str):
            stripped = raw_text.strip()
            parsed_payload: Any = None
            if stripped.startswith(("[", "{")):
                try:
                    parsed_model = ExtractedTable.model_validate_json(
                        raw_text, strict=False
                    )
                    parsed_payload = parsed_model.model_dump()
                except Exception:
                    parsed_payload = None
            if parsed_payload is None:
                parsed_payload = _extract_json_fragment(stripped)
            if parsed_payload is None:
                try:
                    parsed_payload = json.loads(stripped)
                except Exception:
                    parsed_payload = None

            if isinstance(parsed_payload, list):
                fallback_rows = _sanitize_rows(parsed_payload)
            elif isinstance(parsed_payload, dict):
                maybe_rows = parsed_payload.get("rows") or parsed_payload.get("table")
                maybe_tables = parsed_payload.get("tables")
                maybe_chunks = parsed_payload.get("chunks")
                if isinstance(maybe_rows, list):
                    fallback_rows = _sanitize_rows(maybe_rows)
                if isinstance(maybe_tables, list):
                    fallback_tables = [tbl for tbl in maybe_tables if isinstance(tbl, dict)]
                if isinstance(maybe_chunks, list):
                    fallback_chunks = _sanitize_chunks(maybe_chunks)

        final_rows = _sanitize_rows(fallback_rows or [])
        final_tables = fallback_tables
        final_chunks = fallback_chunks or (
            _rows_to_cell_chunks(
                final_rows,
                doc_id=doc_id,
                source=source,
                page_number=page_number,
                table_index=table_index,
                table_title=table_title,
            )
            if return_cell_chunks
            else []
        )
        return {
            "page": page_number,
            "image_path": str(image_path),
            "rows": final_rows,
            "tables": final_tables,
            "chunks": final_chunks,
            "model_response": raw_response,
            "cell_chunks": final_chunks if return_cell_chunks else [],
        }
    finally:
        try:
            Path(image_path).unlink(missing_ok=True)
            logger.debug("Deleted temporary image %s", image_path)
        except Exception as cleanup_exc:  # pragma: no cover - best-effort cleanup
            logger.debug(
                "Unable to delete temporary image %s", image_path, exc_info=cleanup_exc
            )
