"""PDF extraction helpers that rely on table utilities.

This module identifies tables on a PDF page, keeps narrative text, and delegates
table normalization to :func:`modules.table_utils.change_table_format`, which
now always invokes the Gemini page extractor whenever a table exists on a page.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable

import numpy as np
import pdfplumber
from sklearn.cluster import KMeans

from config_loader import load_config
from ingest.ocr import _extract_textract_lines
from ingest.ocr_gemini_chunking import apply_gemini_ocr_chunking

from logging_config import get_logger

from modules.table_utils import (
    bbox_intersects,
    build_cell_border_grid,
    build_cell_border_grid_from_cells_only,
    check_and_fix_nested_tables,
    change_table_format,
)

from modules.chunking_utils import TRAIL_NUM_RE, should_strip_trailing_number

from modules.narrative_utils import(
    BULLET_TOKENS,
    _strip_bullet,
    _same_line,
    _is_probable_header_footer,
)

logger = get_logger(__name__)

# Textract geometry is normalized to 0..1. Scale it into a page-like coordinate
# space so the existing indentation heuristics (tuned for pdfplumber points) can
# still detect nested bullets/lists on scanned PDFs.
OCR_GEOMETRY_SCALE = 1000.0
LANGUAGE_FILTER_MIN_LINES = 4
LANGUAGE_FILTER_COLUMN_SPLIT_MIN_RATIO = 0.25
_INDONESIAN_HINT_TOKENS = {
    "dan",
    "yang",
    "untuk",
    "dengan",
    "pada",
    "dari",
    "tidak",
    "akan",
    "adalah",
    "karyawan",
    "cuti",
    "izin",
}

try:
    from langdetect import detect as _langdetect_detect  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _langdetect_detect = None


def _line_sort_key(line: Dict[str, Any]) -> tuple[float, float]:
    """Return a stable top/left sort key for Textract LINE blocks."""

    bbox = (((line.get("Geometry") or {}).get("BoundingBox")) or {})
    return (float(bbox.get("Top", 0.0)), float(bbox.get("Left", 0.0)))


def _build_ocr_narrative_lines(textract_lines: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Textract LINE blocks into the narrative-line shape used downstream."""

    narrative_lines: List[Dict[str, Any]] = []
    for line in sorted(textract_lines, key=_line_sort_key):
        text = str(line.get("Text") or "").strip()
        if not text:
            continue

        bbox = (((line.get("Geometry") or {}).get("BoundingBox")) or {})
        left = float(bbox.get("Left", 0.0)) * OCR_GEOMETRY_SCALE
        top = float(bbox.get("Top", 0.0)) * OCR_GEOMETRY_SCALE
        height = float(bbox.get("Height", 0.0)) * OCR_GEOMETRY_SCALE
        narrative_lines.append(
            {
                "raw": text,
                "text": text,
                "had_bullet": False,
                "min_x": left,
                "hanging_x": left,
                "top": top,
                "bottom": top + height,
                "in_band": False,
                "trail_num_token": None,
            }
        )
    return narrative_lines


def kmeans_column_clustering(x_centers: List[float], k: int = 2, random_state: int = 42) -> tuple[List[int], List[float]]:
    """Cluster line x-centers into columns and return left->right labels.

    This follows the exact approach from the previous reference function:
    sklearn KMeans in 1D, then remap cluster IDs based on sorted centers.
    """

    if not x_centers:
        return [], []
    if len(x_centers) < k:
        center = float(x_centers[0])
        return [0 for _ in x_centers], [center, center]

    X = np.array(x_centers, dtype=float).reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)

    centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    remapped_labels = np.array([remap[int(label)] for label in labels], dtype=int)
    sorted_centers = [float(centers[int(idx)]) for idx in order]
    return remapped_labels.tolist(), sorted_centers


def _detect_language_code(text: str) -> str:
    """Detect language code for right-column text.

    The optional ``langdetect`` package is used when present. If unavailable,
    this falls back to a compact Indonesian keyword heuristic.
    """

    cleaned = (text or "").strip()
    if not cleaned:
        return "unknown"

    if _langdetect_detect is not None:
        try:
            return _langdetect_detect(cleaned)
        except Exception:
            logger.debug("langdetect failed; falling back to Indonesian keyword heuristic")

    words = re.findall(r"[a-zA-Z]+", cleaned.lower())
    if not words:
        return "unknown"
    id_hits = sum(1 for token in words if token in _INDONESIAN_HINT_TOKENS)
    return "id" if id_hits >= max(1, len(words) // 12) else "non-id"


def _filter_multilingual_two_columns(
    narrative_lines: List[Dict[str, Any]],
    *,
    page_width: float,
    page_number: int,
    source_mode: str,
) -> List[Dict[str, Any]]:
    """Apply bilingual two-column filtering on one page of narrative lines."""

    if len(narrative_lines) < LANGUAGE_FILTER_MIN_LINES:
        return narrative_lines

    x_centers = [float(line.get("hanging_x", line.get("min_x", 0.0))) for line in narrative_lines]
    labels, centers = kmeans_column_clustering(x_centers, k=2, random_state=42)
    center_gap = abs(float(centers[0]) - float(centers[1]))
    split_threshold = LANGUAGE_FILTER_COLUMN_SPLIT_MIN_RATIO * max(page_width, 1.0)
    if center_gap < split_threshold:
        logger.debug(
            "Skipping bilingual column filter on near single-column page",
            extra={
                "page": page_number,
                "source_mode": source_mode,
                "center_gap": round(center_gap, 2),
                "split_threshold": round(split_threshold, 2),
            },
        )
        return narrative_lines

    # Labels are already remapped to left(0) -> right(1).
    right_label = 1
    right_lines = [line for idx, line in enumerate(narrative_lines) if labels[idx] == right_label]
    right_text = "\n".join(str(line.get("text") or "").strip() for line in right_lines if line.get("text")).strip()
    if not right_text:
        logger.debug(
            "No right-column text detected; keeping full page",
            extra={"page": page_number, "source_mode": source_mode},
        )
        return narrative_lines

    detected_lang = _detect_language_code(right_text)
    logger.info(
        "Detected right-column language",
        extra={"page": page_number, "source_mode": source_mode, "detected_language": detected_lang},
    )
    if detected_lang == "id":
        return narrative_lines

    filtered_lines = [line for idx, line in enumerate(narrative_lines) if labels[idx] != right_label]
    logger.info(
        "Right column is not Indonesian; keeping left column only",
        extra={
            "page": page_number,
            "source_mode": source_mode,
            "original_lines": len(narrative_lines),
            "kept_lines": len(filtered_lines),
        },
    )
    return filtered_lines or narrative_lines


def _apply_scanned_pdf_ocr_fallback(
    pdf_path: str,
    pages_output: List[Dict[str, Any]],
    *,
    profile: str = "",
) -> List[Dict[str, Any]]:
    """Fill empty page text using Textract when pdfplumber cannot read a scanned PDF.

    The fallback is intentionally conservative: it only calls Textract when at
    least one page has neither narrative text nor detected tables. Pages that
    already contain text extracted by ``pdfplumber`` are left unchanged. When
    OCR text is produced and ``config.use_gemini_chunking`` is enabled, Gemini
    is used strictly as a chunker over the Textract text (not as the OCR
    engine).
    """

    empty_pages = [
        page for page in pages_output if len(page.get("narrative_text")) < 100 and not page.get("tables")
    ]
    if not empty_pages:
        logger.debug(
            "No fully empty pages detected after initial extraction; skipping OCR fallback",
        )
        return pages_output

    logger.debug(
        "Detected %d empty pages after initial extraction; checking OCR fallback",
        len(empty_pages),
        extra={"pdf": Path(pdf_path).name, "empty_pages": [p["page"] for p in empty_pages]},
    )
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if not s3_bucket:
        logger.warning(
            "Skipping OCR fallback because S3_BUCKET_NAME is not configured",
            extra={"pdf": Path(pdf_path).name},
        )
        return pages_output

    logger.debug(
        "Running Textract OCR fallback for scanned PDF pages",
        extra={"pdf": Path(pdf_path).name, "empty_pages": [p["page"] for p in empty_pages]},
    )

    textract_lines = _extract_textract_lines(pdf_path=pdf_path, s3_bucket=s3_bucket)
    lines_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for line in textract_lines:
        page_number = int(line.get("Page") or 0)
        if page_number <= 0:
            continue
        lines_by_page.setdefault(page_number, []).append(line)

    config = load_config(profile)
    multilingual_filter_enabled = getattr(config, "enable_multilingual_column_filter", True)

    textract_pages_updated = False
    for page in pages_output:
        if len(page.get("narrative_text", "")) >= 100 and not page.get("tables"):
            continue

        ocr_lines = _build_ocr_narrative_lines(lines_by_page.get(page["page"], []))
        if not ocr_lines:
            continue
        if multilingual_filter_enabled:
            ocr_lines = _filter_multilingual_two_columns(
                ocr_lines,
                page_width=OCR_GEOMETRY_SCALE,
                page_number=page["page"],
                source_mode="ocr",
            )

        page["narrative_lines"] = ocr_lines
        page["narrative_text"] = "\n".join(line["text"] for line in ocr_lines).strip()
        page["ocr_used"] = True
        page["ocr_engine"] = "textract"
        textract_pages_updated = True

    if textract_pages_updated and getattr(config, "use_gemini_chunking", True):
        logger.debug(
            "Applying Gemini chunking to OCR-derived narrative text"
        )
        pages_output = apply_gemini_ocr_chunking(pages_output, profile=profile)

    return pages_output


def _looks_like_floating_trailing_number(line_words: List[Dict[str, Any]], prev_words: List[Dict[str, Any]]) -> bool:
    """Return True if *line_words* looks like a dangling superscript number for the previous line."""
    if not line_words or not prev_words:
        return False

    # Merge tokens to inspect their textual content
    token = "".join(w.get("text", "") for w in line_words).strip()
    if not token:
        return False

    # Allow a narrow set of patterns: 1-4 digits, optionally separated by thin spaces
    normalized = token.replace(" ", "")
    if not normalized.isdigit() or len(normalized) > 4:
        return False

    # Superscripts typically sit near the end of the previous line and are narrow
    prev_right = max(float(w.get("x1", 0.0)) for w in prev_words)
    current_left = min(float(w.get("x0", 0.0)) for w in line_words)
    current_right = max(float(w.get("x1", 0.0)) for w in line_words)
    if current_left < (prev_right - 12.0):
        return False
    if (current_right - current_left) > 32.0:
        return False

    prev_top = min(float(w.get("top", 0.0)) for w in prev_words)
    prev_bottom = max(float(w.get("bottom", 0.0)) for w in prev_words)
    current_top = min(float(w.get("top", 0.0)) for w in line_words)
    current_bottom = max(float(w.get("bottom", 0.0)) for w in line_words)

    # Accept small vertical offsets (a little above or overlapping), but reject numbers
    # that clearly belong to the next baseline.
    if current_top > (prev_bottom + 6.0):
        return False
    if current_bottom < (prev_top - 12.0):
        return False

    return True


def _merge_floating_trailing_numbers(grouped_words: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """Attach stray superscript-like number lines back to the previous line."""

    if not grouped_words:
        return grouped_words

    merged: List[List[Dict[str, Any]]] = []
    for line_words in grouped_words:
        if merged and _looks_like_floating_trailing_number(line_words, merged[-1]):
            merged[-1].extend(line_words)
            merged[-1].sort(key=lambda w: (float(w.get("x0", 0.0)), float(w.get("top", 0.0))))
        else:
            merged.append(line_words)
    return merged


def get_page_elements(
    pdf_path,
    *,
    prefer_table_detection: bool = False,
    ocr_fallback: bool = True,
    profile: str = "",
):
    """Return page dictionaries containing narrative text and table metadata.

    When ``prefer_table_detection`` is ``True`` (set by ``extract_pdf_content``
    when Gemini is in always-use mode), the heuristic that reclassifies framed
    paragraph blocks as narrative text is skipped so ambiguous "tables" still
    reach the Gemini extractor. Manual parsing remains available as a fallback
    when Gemini fails.

    When ``ocr_fallback`` is enabled, empty pages are filled from Textract in
    the same pass so callers only need a single extraction entry point.
    """
    pdf_name = Path(pdf_path).name
    config = load_config(profile)
    multilingual_filter_enabled = getattr(config, "enable_multilingual_column_filter", True)
    logger.info("Scanning PDF for tables and narrative", extra={"pdf": pdf_name})
    pages_output = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            page_width = float(page.width or 0)
            page_height = float(page.height or 0)

            table_bboxes = []
            tables = []
            try:
                table_objs = page.find_tables()
            except Exception:
                table_objs = []
            for t_idx, t in enumerate(table_objs):
                tbbox = t.bbox
                table_bboxes.append({"bbox": tbbox, "table_index": t_idx})
                try:
                    rows = t.extract()
                except Exception:
                    rows = None
                try:
                    grid = build_cell_border_grid(page, t)
                    if grid is None:
                        grid = build_cell_border_grid_from_cells_only(page, t)
                except Exception:
                    grid = None
                tables.append(
                    {
                        "table_index": t_idx,
                        "bbox": tbbox,
                        "rows": rows,
                        "grid": grid,
                    }
                )

            def _is_paragraph_like_table(table_record: dict) -> bool:
                """Return True when a detected table is really a paragraph block.

                Some PDFs are laid out with a single framed text block that
                ``pdfplumber`` detects as a 1–2 column table. When the block is
                only a few rows high, spans most of the page, and contains a lot
                of uninterrupted text, we should treat it as narrative instead
                of a table so the downstream parser does not route it through
                the complex-table logic or Gemini.
                """

                rows = table_record.get("rows") or []
                if not rows:
                    return False

                row_count = len(rows)
                col_count = max((len(r) if isinstance(r, (list, tuple)) else 0) for r in rows)
                if row_count > 3 or col_count > 2:
                    return False

                cell_texts = [
                    str(cell).strip()
                    for row in rows
                    for cell in (row or [])
                    if cell not in (None, "")
                ]
                total_chars = sum(len(t) for t in cell_texts)
                if total_chars < 120:
                    return False

                bbox = table_record.get("bbox")
                if not bbox:
                    return False

                width_ratio = abs(float(bbox[2]) - float(bbox[0])) / max(page_width, 1.0)
                height_ratio = abs(float(bbox[3]) - float(bbox[1])) / max(page_height, 1.0)

                # Frame-like blocks covering most of the page and carrying a lot
                # of text are better treated as narrative paragraphs.
                return width_ratio >= 0.6 and height_ratio >= 0.35

            def _is_full_page_text_table(table_record: dict) -> bool:
                """Return True when a large framed block is really body text.

                Some pages wrap long paragraphs in a full-width, multi-row box
                that ``pdfplumber`` flags as a table, which would otherwise
                strip all text from the narrative stream. When the detection:

                * spans most of the page,
                * uses only one or two columns, and
                * contains substantial paragraph text,

                we reclassify it as narrative so the page text is preserved and
                the downstream parser does not treat the page as a complex table.
                """

                rows = table_record.get("rows") or []
                if not rows:
                    return False

                row_count = len(rows)
                col_count = max((len(r) if isinstance(r, (list, tuple)) else 0) for r in rows)
                if col_count == 0 or col_count > 2:
                    return False

                # Require enough body text to justify reclassifying as narrative.
                cell_texts = [
                    str(cell).strip()
                    for row in rows
                    for cell in (row or [])
                    if cell not in (None, "")
                ]
                total_chars = sum(len(t) for t in cell_texts)
                if total_chars < 200:
                    return False

                bbox = table_record.get("bbox")
                if not bbox:
                    return False

                width_ratio = abs(float(bbox[2]) - float(bbox[0])) / max(page_width, 1.0)
                height_ratio = abs(float(bbox[3]) - float(bbox[1])) / max(page_height, 1.0)

                # Full-page framed definitions/procedure blocks should stay in
                # the narrative stream instead of being treated as tables.
                return width_ratio >= 0.6 and height_ratio >= 0.5 and row_count <= 8

            def _is_heading_banner(table_record: dict) -> bool:
                """Return True when the detection is a single-row heading bar.

                Some documents place page titles or section headers inside a
                thin framed box that ``pdfplumber`` reports as a tiny table. When
                the detection has a single row, only one non-empty cell, and a
                shallow height, treat it as narrative text so the heading is
                preserved in the text stream instead of being routed through the
                table pipeline.
                """

                rows = table_record.get("rows") or []
                if len(rows) != 1:
                    return False

                non_empty_cells = [
                    cell for cell in rows[0] if cell not in (None, "")
                ]
                if len(non_empty_cells) != 1:
                    return False

                bbox = table_record.get("bbox")
                if not bbox:
                    return False

                height_ratio = abs(float(bbox[3]) - float(bbox[1])) / max(page_height, 1.0)
                return height_ratio <= 0.15

            def _drop_tables(indices_to_remove):
                if not indices_to_remove:
                    return
                remaining_tables = [
                    table for idx, table in enumerate(tables) if idx not in indices_to_remove
                ]
                remaining_bboxes = [
                    bbox for idx, bbox in enumerate(table_bboxes) if idx not in indices_to_remove
                ]
                removed = sorted(indices_to_remove)
                logger.debug(
                    "Reclassifying framed text as narrative text; removing tables",
                    extra={"pdf": pdf_name, "page": pidx, "tables_removed": removed},
                )
                return remaining_tables, remaining_bboxes

            # Reclassify paragraph-like and heading detections as narrative blocks
            # so we do not exclude their words from the text stream nor send them
            # to the complex table parser.
            if not prefer_table_detection:
                paragraph_table_indices = [
                    idx for idx, table in enumerate(tables) if _is_paragraph_like_table(table)
                ]
                heading_table_indices = [
                    idx for idx, table in enumerate(tables) if _is_heading_banner(table)
                ]
                full_page_text_table_indices = [
                    idx
                    for idx, table in enumerate(tables)
                    if _is_full_page_text_table(table)
                ]

                indices_to_remove = (
                    set(paragraph_table_indices)
                    | set(heading_table_indices)
                    | set(full_page_text_table_indices)
                )

                if indices_to_remove:
                    updated = _drop_tables(indices_to_remove)
                    if updated:
                        tables, table_bboxes = updated
                        tables = [
                            {**table, "table_index": new_idx}
                            for new_idx, table in enumerate(tables)
                        ]

            words = page.extract_words(
                x_tolerance=1.5,
                y_tolerance=3.0,
                keep_blank_chars=False,
                use_text_flow=True
            )

            narrative_words = []
            for w in words:
                wbox = (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
                if not any(bbox_intersects(wbox, tb["bbox"]) for tb in table_bboxes):
                    narrative_words.append(w)

            # 3) Group words into lines (geometry aware)
            # narrative_words = sorted(
            #     narrative_words,
            #     key=lambda w: (round(float(w["top"]), 1), float(w["x0"]))
            # )

            grouped = []
            cur = []
            for w in narrative_words:
                if not cur or _same_line(cur[-1], w, tol=3.0):
                    cur.append(w)
                else:
                    grouped.append(cur)
                    cur = [w]
            if cur:
                grouped.append(cur)

            grouped = _merge_floating_trailing_numbers(grouped)

            # 4) Build indent-aware narrative lines; filter headers/footers robustly
            narrative_lines = []
            for ws in grouped:
                raw = " ".join(w["text"] for w in ws).strip()
                top = float(ws[0]["top"])          # you already have this
                page_h = float(page.height)
                page_no = pidx
                in_band = (top < 0.10 * page_h) or (top > 0.90 * page_h)
                if not raw:
                    continue

                top = float(ws[0]["top"])
                
                m = TRAIL_NUM_RE.search(raw)
                trail_num_token = None
                if m:
                    candidate_token = m.group(1)
                    if should_strip_trailing_number(
                        line=raw,
                        y_top=top,
                        page_height=page_h,
                        page_no=page_no,
                        captured_token=candidate_token,
                    ):
                        raw = raw[: m.start()].rstrip()
                    else:
                        trail_num_token = candidate_token

                # Universal header/footer & page-marker filter (position + content)
                if _is_probable_header_footer(raw, top, page_height):
                    continue

                # Trailing page numbers are removed above when requested; reuse the text as-is.
                clean_tail = raw
                
                content, had_bullet = _strip_bullet(clean_tail)
                min_x = min(float(w["x0"]) for w in ws)

                if had_bullet:
                    if ws[0]["text"] in BULLET_TOKENS and len(ws) > 1:
                        hanging_x = float(ws[1]["x0"])
                    else:
                        hanging_x = float(ws[0]["x0"])
                else:
                    hanging_x = float(ws[0]["x0"])

                narrative_lines.append({
                    "raw": raw,                 # full text as seen on the page
                    "text": content,            # without the leading bullet token
                    "had_bullet": had_bullet,   # True if a bullet marker was detected
                    "min_x": float(min_x),      # line left edge
                    "hanging_x": float(hanging_x),  # first content x
                    "top": top,
                    "bottom": float(ws[0]["bottom"]),
                    "in_band": bool(in_band),            # NEW
                    "trail_num_token": trail_num_token,  # NEW
                })

            # 5) Optional bilingual two-column filtering + plain text (compat)
            if multilingual_filter_enabled:
                narrative_lines = _filter_multilingual_two_columns(
                    narrative_lines,
                    page_width=page_width,
                    page_number=pidx,
                    source_mode="pdfplumber",
                )

            narrative_text = "\n".join(nl["text"] for nl in narrative_lines).strip()
            table_records = []
            for t in tables:
                if t["rows"]:
                    table_dct = t["rows"]
                else:
                    table_dct = None
                table_records.append(
                    {
                        "table_index": t["table_index"],
                        "bbox": t["bbox"],
                        "table_dct": table_dct,
                        "grid": t["grid"],
                    }
                )

            pages_output.append(
                {
                    "page": pidx,
                    "narrative_text": narrative_text,
                    # NEW: export the per-line metadata so the parser can use indentation
                    "narrative_lines": narrative_lines,
                    "tables": table_records,
                }
            )
    if ocr_fallback:
        logger.info("Checking for scanned PDF fallback after initial extraction",)
        pages_output = _apply_scanned_pdf_ocr_fallback(pdf_path, pages_output, profile=profile)
    return pages_output


def extract_pdf_content(file, use_gemini: str = "use", profile: str = ""):
    """Extract page text/tables from a PDF, with OCR fallback for scanned files.

    When ``config.use_gemini_chunking`` is enabled, Textract-derived OCR
    narrative pages are additionally chunked with the configured Gemini chat
    model before the standard ingest builder runs. This keeps scanned pages
    aligned with the structured hierarchy used by the manual parser without
    changing the separate Gemini image/table extraction path.
    """
    # When Gemini is explicitly enabled, keep ambiguous framed text blocks so
    # the vision model can decide whether they are true tables. Manual parsing
    # still runs as a fallback if Gemini fails.
    prefer_tables = (use_gemini or "").lower() == "use"
    pdf_pages = get_page_elements(
        file,
        prefer_table_detection=prefer_tables,
        ocr_fallback=True,
        profile=profile,
    )

    pdf_pages = check_and_fix_nested_tables(pdf_pages)
    # Gemini now runs for any page that contains at least one table so the
    # extractor can directly produce atomic cell + narrative chunks. The
    # ``use_gemini_for_complex`` flag is retained for compatibility, but the
    # change_table_format helper always handles the Gemini invocation.
    pdf_pages = change_table_format(
        pdf_pages,
        pdf_path=file,
        use_gemini_mode=use_gemini,
        use_gemini_for_complex=True,
    )
    return pdf_pages

'''
Use this as a reference for bilinguan page
'''
'''
def extract_document_and_cluster_language(
    pdf_file_path: str,
    s3_key: str,
    s3_bucket: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """Extract OCR text for scanned PDFs as page-numbered tuples.

    Returns a list of (page_number, page_text) entries to match the poppler
    extraction format used elsewhere in the pipeline. Missing pages are filled
    with empty strings to preserve the original page count when possible.
    """
    logger.info("Running OCR Extraction")
    # if pdf_has_extractable_text(pdf_file_path):
    #     logger.debug("Using poppler for text-based PDF extraction")
    #     return extract_pdf_text_with_poppler(pdf_file_path)
    s3_bucket = os.getenv("S3_BUCKET_NAME") if not s3_bucket else s3_bucket
    if not s3_bucket:
        logger.error("S3 Bucket name is not defined!")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="S3 Bucket name is not defined!",
        )
 
    logger.debug("Using Textract OCR for scanned PDF extraction")
    resp = _extract_textract_lines(
        pdf_path=pdf_file_path, s3_bucket=s3_bucket, s3_key=s3_key
    )
    logger.debug("Extracted doc with %d LINE blocks", len(resp))
    Lines, __ = create_blocks(resp)
    if not Lines:
        logger.debug("No LINE blocks returned from OCR")
        return []
    # Prefer the real PDF page count so the output mirrors poppler extraction.
    total_pages = 0
    try:
        import pdfplumber

        with pdfplumber.open(pdf_file_path) as pdf:
            total_pages = len(pdf.pages)
    except Exception as exc:
        logger.warning("Failed to read PDF page count: %s", exc)
    x_centers = [Line.left + Line.width / 4 for Line in Lines]
    logger.debug("Running K-means for clustering the geometry")
    labels, centers = kmeans_column_clustering(x_centers, k=2)
    Line_pages, Lines = get_line_per_page_and_add_position(Lines, labels)
    # for i, Line in enumerate(Lines):
    #     label = 'left' if labels[i] == 0 else 'right'
    #     setattr(Line, 'column_position', label)
    logger.debug("K-means clustering done")

    if total_pages <= 0:
        total_pages = max((line.page for line in Lines), default=0)
    pages_by_number: dict[int, str] = {}

    if float(abs(centers[0] - centers[1])) < 0.25:
        logger.debug("No multi-column detected, sending whole page")
        for line_page in Line_pages:
            if not line_page:
                continue
            page_number = line_page[0].page
            pages_by_number[page_number] = convert_resp_to_text(line_page)

        return [
            (page_num, pages_by_number.get(page_num, ""))
            for page_num in range(1, total_pages + 1)
        ]
    
    # mid_x = float(centers[1])*0.5 + float(centers[0])*0.5

    for i, line_page in enumerate(Line_pages, start=1):
        if not line_page:
            continue
        right_column_text = "\n".join(
            [line.text for line in line_page if line.column_position == 'right']
        ).strip()
        if right_column_text:
            detected_lang = langdetect.detect(right_column_text)
            logger.debug("The right column language in page %d is: %s", i, detected_lang)
            if detected_lang != 'id':
                pages_by_number[line_page[0].page] = convert_resp_to_text(line_page, 'left')
                logger.debug("Sending left column only")
            else:
                pages_by_number[line_page[0].page] = convert_resp_to_text(line_page)
                logger.debug("Sending all column")
        else:
            logger.debug("No right column text found on page %d; sending all column", i)
            pages_by_number[line_page[0].page] = convert_resp_to_text(line_page)

    return [
        (page_num, pages_by_number.get(page_num, ""))
        for page_num in range(1, total_pages + 1)
    ]
'''
