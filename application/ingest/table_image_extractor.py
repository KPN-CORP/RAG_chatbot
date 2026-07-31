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
import shutil
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config_loader import load_config
from logging_config import get_logger
from schemas.table_extraction import ExtractedTable, GeminiChunkList

logger = get_logger(__name__)


def _log_context(**kwargs: Any) -> Dict[str, Any]:
    """Return a compact logging context without ``None`` values for tidy logs."""

    return {key: value for key, value in kwargs.items() if value is not None}


DEFAULT_EXTRACTION_PROMPT = textwrap.dedent(
    f"""
    You are an OCR + Vision-Language assistant used in a Retrieval-Augmented Generation (RAG) system.

    You receive exactly one document page at a time (image or rendered page). Your job is NOT to mechanically extract tables. Your job is to read, understand, and segment the page into the best possible set of useful information chunks.

    STRICT NON-HALLUCINATION RULES

    You MUST only output information that is explicitly visible on the page.

    - Never invent, infer, or correct policy content, numbers, dates, or ranges.
    - Never complete partially visible patterns based on prior or outside knowledge.
    - If some text or number is unreadable or too blurry, write "[UNREADABLE]" instead of guessing.
    - If you are not sure whether a value belongs to one column or another, mark the ambiguous part as "[UNSURE]" or skip that chunk entirely.
    - If a cell appears empty, do not fill it with "-", "0", or any assumed value unless that exact symbol is present.
    - Do not summarize, paraphrase, or expand the policy. Copy the wording as literally as possible.
    - If you are unsure whether a chunk's content is truly present on the page, it is safer to SKIP that chunk than to guess.

    CHUNKING RULES

    Each chunk must:
    - Express one clear piece of information that can stand alone (semantically atomic).
    - Include the title(s) of the page or section and any relevant notes (so it is self-contained).
    - Be useful for retrieval and answering questions later.

    You must handle complex layouts, including:
    - Simple or messy tables, multi-level headers, merged cells, nested tables, matrices, or non-grid layouts.
    - Narrative text, bullets, numbered lists, and footnotes outside tables.

    MERGED / SPANNING CELLS (critical for correctness):
    - When a header, category, or label cell is visually merged so that it spans several rows or several columns, treat its value as applying to EVERY row and column it covers.
    - Repeat that spanning value in the semantic context of every chunk derived from the rows/columns it governs. For example, if a leftmost category cell is merged down over five rows, all five rows' chunks must carry that category.
    - A cell that is blank ONLY because it sits underneath (or to the right of) a merged cell is NOT empty: use the spanning value, never leave it out.
    - Never emit a chunk that drops the merged header/category that gives it meaning (e.g., do not output "12 hari" without the leave type it belongs to).

    1. Title chain:
    - Detect the main title and any subtitles or section titles.
    - For every chunk, include this title chain.

    2. Notes:
    - Detect notes/footnotes/legends (for example *, 1), 2), "Catatan", "Note").
    - If a note clearly applies to some information, include its text in the chunk.

    3. Focus on INFORMATION, not structure:
    - Keep policy statements, conditions, ranges, durations, entitlements, monetary values, percentages, eligibility rules, procedures, etc.
    - Ignore empty cells, pure layout labels, row/column indices, decorative elements, or "CONTROLLED COPY"-type stamps unless they change policy meaning.
    - Ignore "No" columns or row numbers if they have no semantic meaning.

    4. Tables:
    - First understand the table logically (headers and what they mean).
    - Create chunks that represent meaningful rules, not raw grid cells.
    - In table-like layouts where a left column contains a short label or heading and the right column contains the main narrative text, TREAT THE LEFT LABEL AS CONTEXT AND COMBINE IT WITH ITS TEXT INTO ONE CHUNK. Do not output chunks that consist only of the label without its explanation.
    - Example: a row like "Overtime on Working Days, ≥2-5 hours, Level 2: X, Level 3: Y, Level 4: Z" should be one chunk describing that rule, not multiple tiny chunks for each cell.
    - If different cells in a row clearly express separate rules, you may split them, but never in a way that loses meaning.

    5. Narrative text outside tables:
    - Each coherent paragraph, bullet list, or numbered list becomes one chunk if it expresses a single concept or procedure.
    - Each numbered item (1., 2., 3., etc.) must be kept as a single chunk; do not split one item across multiple chunks.
    - Always include the associated heading or section label as part of the semantic context of the chunk; do not output chunks that consist only of a heading with no explanatory text.
    - If a paragraph contains several distinct rules, you may split it into multiple chunks, but do not cut sentences in half.

    6. Language and fidelity:
    - Preserve the original language, numbers, symbols, ranges, and units exactly (for example "≥ 2 s.d. ≤ 5 jam", "Rp60.000,00").
    - Do not rewrite in your own words; keep the meaning faithful to the original.
    - If OCR is uncertain, write your best guess, but never add information that is not there.

    CHUNK TEXT FORMAT

    For each chunk, output a single string that combines (if present):
    - The title chain.
    - A short semantic context (for example a section name, row label, or condition).
    - The actual rule or information.
    - Any relevant notes.

    Use a consistent pattern:

    "<Title chain> | <Semantic context> | <Information> | Notes: <note text or '-'>"

    Examples of semantic context:
    - "Lembur di Hari Kerja, ≥2-5 jam, Gol.2-4"
    - "Pengajuan lembur"
    - "Durasi lembur"
    - "Perhitungan realisasi jam kerja lembur"

    OUTPUT FORMAT

    Your final output must be ONLY a JSON array of strings.
    Each string is one chunk text as described above.
    Do not include any objects, keys, metadata, comments, or explanations.
    Do not add any text before or after the JSON.

    Example output shape (structure only):

    [
    "<Title chain> | <Semantic context> | <Information> | Notes: -",
    "<Title chain> | <Semantic context> | <Information> | Notes: -"
    ]

    BEFORE OUTPUT

    Before returning the JSON array, check each candidate chunk:
    - Does it contain real information (not just structure)?
    - Can it stand alone and still make sense?
    - Does it include the title chain and any relevant notes?
    - Is it faithful to the original page?

    If the answer is yes, include it. If not, adjust or discard the chunk.

    The JSON array you output must be freshly generated from the current page.
    Do not copy any example chunks from this prompt.
    Do not include any explanation, comments, code fences, or extra text before or after the JSON array.
    """
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
            "chunk_text",
            "table_index",
            "row_index",
            "column_header",
            "row_header",
            "notes",
            "title",
            "title_chain",
            "page_notes",
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


def _sanitize_chunk_strings(chunks: List[Any]) -> List[str]:
    """Ensure Gemini chunk outputs are plain, non-empty strings."""

    sanitized: List[str] = []
    for chunk in chunks or []:
        text = chunk if isinstance(chunk, str) else str(chunk)
        text = text.strip()
        if text:
            sanitized.append(text)
    return sanitized


def _chunk_strings_to_dicts(chunk_strings: List[str]) -> List[Dict[str, Any]]:
    """Represent bare Gemini chunk strings as minimal chunk dictionaries."""

    return [
        {"chunk_text": text, "chunk_type": "narrative", "text": text}
        for text in chunk_strings
    ]


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


@lru_cache(maxsize=1)
def _resolve_poppler_path() -> Optional[str]:
    """Locate Poppler's binaries so ``pdf2image`` works regardless of activation.

    ``pdf2image`` shells out to ``pdfinfo``/``pdftoppm``. When the app is launched
    without activating the conda env (service, cron, direct interpreter), the
    env's ``Library/bin`` (Windows) or ``bin`` (POSIX) is not on PATH and Gemini
    table extraction silently fails, falling back to lower-quality parsing. We
    first honour PATH, then fall back to the interpreter's own env dir. The
    ``POPPLER_PATH`` env var overrides everything.
    """

    override = os.environ.get("POPPLER_PATH")
    if override and os.path.isdir(override):
        return override

    # Already on PATH → let pdf2image find it (return None = default behaviour).
    if shutil.which("pdfinfo") or shutil.which("pdftoppm"):
        return None

    probe = "pdfinfo.exe" if os.name == "nt" else "pdfinfo"
    candidates = [
        os.path.join(sys.prefix, "Library", "bin"),  # Windows conda layout
        os.path.join(sys.prefix, "bin"),             # POSIX conda/venv layout
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, probe)):
            return candidate

    return None  # Not found; pdf2image will raise its usual informative error.


def _page_to_image(pdf_path: Path, page_number: int, dpi: int = 300) -> Path:
    """Render a single PDF page to a PNG image and return the image path."""

    try:
        from pdf2image import convert_from_path
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pdf2image is required for image-based extraction") from exc

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number,
        last_page=page_number,
        poppler_path=_resolve_poppler_path(),
    )
    if not images:
        raise ValueError(f"Failed to render page {page_number} from {pdf_path}")

    image_path = pdf_path.with_suffix("")
    image_path = image_path.with_name(f"{image_path.name}_page{page_number}.png")
    images[0].save(image_path, format="PNG")
    logger.debug(
        "Rendered page image",
        extra=_log_context(pdf=pdf_path.name, page=page_number, image=str(image_path)),
    )
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
    so the returned structure is guaranteed to match the simplified chunk list
    schema (now just a list of strings). When ``return_cell_chunks`` is
    ``True`` and ``doc_id``/``source`` are provided, the chunk strings are
    converted into the atomic chunk format to simplify downstream ingest.
    """

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    image_path = _page_to_image(pdf_file, page_number, dpi=dpi)
    prompt_text = prompt or DEFAULT_EXTRACTION_PROMPT

    log_meta = _log_context(
        pdf=pdf_file.name,
        page=page_number,
        doc_id=doc_id,
        source=source,
        table_index=table_index,
    )

    logger.info("Gemini page extraction starting", extra=log_meta)

    try:
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        config = load_config()
        chunk_parser = PydanticOutputParser(pydantic_object=GeminiChunkList)

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
            model=model_name or getattr(config, "gemini_image_model", None),
            google_api_key=api_key
            or os.environ.get(getattr(config, "google_api_key_name", "GEMINI_API_KEY")),
            temperature=0,
        )

        chain = prompt_template.partial(
            format_instructions=chunk_parser.get_format_instructions()
        ) | chat_model | chunk_parser

        try:
            parsed: GeminiChunkList = chain.invoke(
                {"user_prompt": prompt_text, "image_b64": encoded_image}
            )
            # Gemini now responds with a list of chunk strings; wrap them so the
            # ingest pipeline keeps a consistent payload shape.
            chunk_strings = _sanitize_chunk_strings(parsed.chunks)
            chunks = _chunk_strings_to_dicts(chunk_strings)
            logger.info(
                "Gemini page extraction completed",
                extra={**log_meta, "rows": 0, "chunks": len(chunks)},
            )
            return {
                "page": page_number,
                "image_path": str(image_path),
                "rows": [],
                "tables": [],
                "chunks": chunks,
                "chunk_strings": chunk_strings,
                "model_response": parsed,
                "cell_chunks": chunks if return_cell_chunks else [],
            }
        except Exception as exc:  # pragma: no cover - network/model variability
            logger.warning(
                "Gemini chain parsing failed; attempting raw response recovery",
                extra=log_meta,
                exc_info=exc,
            )

        # Fallback: request the model output directly and try to coerce JSON even if
        # extra text surrounds it. This prevents ingestion from breaking when the
        # model returns prose plus JSON.
        messages = prompt_template.format_messages(
            format_instructions=chunk_parser.get_format_instructions(),
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
        fallback_chunk_strings: List[str] = []

        if isinstance(raw_text, str):
            stripped = raw_text.strip()
            parsed_payload: Any = None
            if stripped.startswith(("[", "{")):
                try:
                    parsed_model = GeminiChunkList.model_validate_json(
                        raw_text, strict=False
                    )
                    parsed_payload = parsed_model.chunks
                except Exception:
                    parsed_payload = None
                if parsed_payload is None:
                    try:
                        table_model = ExtractedTable.model_validate_json(
                            raw_text, strict=False
                        )
                        parsed_payload = table_model.model_dump()
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
                fallback_chunk_strings = _sanitize_chunk_strings(parsed_payload)
                if not fallback_chunk_strings:
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
                    fallback_chunk_strings = _sanitize_chunk_strings(maybe_chunks)
                    if not fallback_chunk_strings:
                        fallback_chunks = _sanitize_chunks(maybe_chunks)

        if fallback_chunk_strings:
            fallback_chunks = _chunk_strings_to_dicts(fallback_chunk_strings)

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
        final_chunk_strings = fallback_chunk_strings or [
            chunk.get("chunk_text", "")
            for chunk in final_chunks
            if isinstance(chunk, dict)
        ]

        logger.info(
            "Gemini fallback extraction completed",
            extra={**log_meta, "rows": len(final_rows), "chunks": len(final_chunks)},
        )
        return {
            "page": page_number,
            "image_path": str(image_path),
            "rows": final_rows,
            "tables": final_tables,
            "chunks": final_chunks,
            "chunk_strings": final_chunk_strings,
            "model_response": raw_response,
            "cell_chunks": final_chunks if return_cell_chunks else [],
        }
    finally:
        try:
            Path(image_path).unlink(missing_ok=True)
            logger.debug(
                "Deleted temporary image",
                extra=_log_context(pdf=pdf_file.name, page=page_number, image=str(image_path)),
            )
        except Exception as cleanup_exc:  # pragma: no cover - best-effort cleanup
            logger.debug(
                "Unable to delete temporary image",
                extra=_log_context(
                    pdf=pdf_file.name, page=page_number, image=str(image_path)
                ),
                exc_info=cleanup_exc,
            )
