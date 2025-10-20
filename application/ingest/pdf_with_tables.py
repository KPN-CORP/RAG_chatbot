"""PDF extraction helpers that rely on table utilities."""

import pdfplumber

from typing import List, Dict, Any

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


def get_page_elements(pdf_path):
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

            # 5) Plain text (compat) + tables normalization
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
    return pages_output


def extract_pdf_content(file):
    pdf_pages = get_page_elements(file)
    pdf_pages = check_and_fix_nested_tables(pdf_pages)
    pdf_pages = change_table_format(pdf_pages)
    return pdf_pages
