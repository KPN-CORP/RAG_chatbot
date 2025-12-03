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
                    extra={"page": pidx, "tables_removed": removed},
                )
                return remaining_tables, remaining_bboxes

            # Reclassify paragraph-like and heading detections as narrative blocks
            # so we do not exclude their words from the text stream nor send them
            # to the complex table parser.
            paragraph_table_indices = [
                idx for idx, table in enumerate(tables) if _is_paragraph_like_table(table)
            ]
            heading_table_indices = [
                idx for idx, table in enumerate(tables) if _is_heading_banner(table)
            ]
            full_page_text_table_indices = [
                idx for idx, table in enumerate(tables) if _is_full_page_text_table(table)
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
                        {**table, "table_index": new_idx} for new_idx, table in enumerate(tables)
                    ]

            heading_row_overrides = []
            for table in tables:
                rows = table.get("rows") or []
                if len(rows) <= 1:
                    continue

                first_row = rows[0] or []
                non_empty_cells = [cell for cell in first_row if cell not in (None, "")]
                if len(non_empty_cells) != 1:
                    continue

                grid = table.get("grid") or {}
                cells = grid.get("cells") or []
                first_row_cells = [c for c in cells if c.get("row") == 0 and c.get("bbox")]
                if not first_row_cells:
                    continue

                x0 = min(float(c["bbox"][0]) for c in first_row_cells)
                y0 = min(float(c["bbox"][1]) for c in first_row_cells)
                x1 = max(float(c["bbox"][2]) for c in first_row_cells)
                y1 = max(float(c["bbox"][3]) for c in first_row_cells)

                height_ratio = abs(y1 - y0) / max(page_height, 1.0)
                if height_ratio > 0.12:
                    continue

                heading_row_overrides.append({"bbox": (x0, y0, x1, y1)})

            words = page.extract_words(
                x_tolerance=1.5,
                y_tolerance=3.0,
                keep_blank_chars=False,
                use_text_flow=True
            )

            narrative_words = []
            for w in words:
                wbox = (float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
                in_table_bbox = any(bbox_intersects(wbox, tb["bbox"]) for tb in table_bboxes)
                in_heading_override = any(
                    bbox_intersects(wbox, hb["bbox"]) for hb in heading_row_overrides
                )

                if (not in_table_bbox) or in_heading_override:
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
    # When the table detector encounters complex layouts (e.g., triple headers),
    # try the Gemini image-based extractor before falling back to the legacy
    # header-normalization logic.
    pdf_pages = change_table_format(
        pdf_pages,
        pdf_path=file,
        use_gemini_for_complex=True,
    )
    return pdf_pages
