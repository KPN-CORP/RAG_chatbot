"""Image-based table extraction helper using pdf2image and Gemini.

This module provides a single entry point, :func:`extract_table_with_gemini`,
that can be used when the regular PDF table detector produces messy structures
(e.g., highly nested or multi-span headers). It converts a single PDF page to a
PNG image via ``pdf2image`` and sends that image to Gemini with a clear
extraction prompt so the model can return structured JSON for the table.

Unlike the initial prototype, this version leverages LangChain's
``ChatGoogleGenerativeAI`` together with ``PydanticOutputParser`` so the Gemini
response is validated against a schema before returning it to callers. This
keeps the ingest pipeline consistent with the rest of the app (see
``application/app.py`` and ``schemas/rag_outputs.py``).
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

EXAMPLE_OUTPUT_FORMAT = """[
  {"Header 1": "row 1 value", "Header 2 - Sub A": "row 1 nested detail", "Notes": "line 1\\nline 2"},
  {"Header 1": "row 2 value", "Header 2 - Sub A": "row 2 nested detail", "Notes": null}
]"""


DEFAULT_EXTRACTION_PROMPT = (
    "You are a data extraction assistant. Read the provided table image and "
    "return a JSON array of row objects. Each row must include every column "
    "from the table header. Preserve text line breaks inside cell values. If "
    "the header spans multiple levels (double header, triple header, or more), "
    "concatenate the levels in a single key using ' - ' between levels. If a "
    "cell contains a nested table or sub-grid, capture its contents as text "
    "(preserving rows with line breaks) within that cell's value. Use the "
    "following example output format (list of dictionaries) and do not add "
    "any text before or after the JSON array:\n"
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
) -> Dict[str, Any]:
    """Extract a complex table by rendering the PDF page and delegating to Gemini.

    The Gemini call is executed through LangChain with a :class:`PydanticOutputParser`
    so the returned structure is guaranteed to match the ``ExtractedTable``
    schema (a list of row dictionaries). This mirrors the pattern used in
    :mod:`application.app` for chat responses.
    """

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    image_path = _page_to_image(pdf_file, page_number, dpi=dpi)
    prompt_text = prompt or DEFAULT_EXTRACTION_PROMPT

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
        return {
            "page": page_number,
            "image_path": str(image_path),
            "rows": rows,
            "model_response": parsed,
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

    if isinstance(raw_text, str):
        stripped = raw_text.strip()
        parsed_payload: Any = None
        if stripped.startswith(("[", "{")):
            try:
                parsed_payload = ExtractedTable.model_validate_json(
                    raw_text, strict=False
                ).rows
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
            if isinstance(maybe_rows, list):
                fallback_rows = _sanitize_rows(maybe_rows)

    return {
        "page": page_number,
        "image_path": str(image_path),
        "rows": _sanitize_rows(fallback_rows or []),
        "model_response": raw_response,
    }
