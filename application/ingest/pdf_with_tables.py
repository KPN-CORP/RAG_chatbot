"""PDF extraction helpers that rely on table utilities."""

import pdfplumber

from logging_config import get_logger

from .table_utils import (
    bbox_intersects,
    build_cell_border_grid,
    build_cell_border_grid_from_cells_only,
    check_and_fix_nested_tables,
    change_table_format,
)

from .narrative_utils import(
    BULLET_TOKENS,
    _strip_bullet,
    _same_line,
    _is_probable_header_footer,
) 

logger = get_logger(__name__)


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
            narrative_words = sorted(
                narrative_words,
                key=lambda w: (round(float(w["top"]), 1), float(w["x0"]))
            )

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

            # 4) Build indent-aware narrative lines; filter headers/footers robustly
            narrative_lines = []
            for ws in grouped:
                raw = " ".join(w["text"] for w in ws).strip()
                if not raw:
                    continue

                top = float(ws[0]["top"])
                # Universal header/footer & page-marker filter (position + content)
                if _is_probable_header_footer(raw, top, page_height):
                    continue

                content, had_bullet = _strip_bullet(raw)
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
                })

            # 5) Plain text (compat) + tables normalization
            narrative_text = "\n".join(nl["raw"] for nl in narrative_lines).strip()
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
                    "tables": table_records,
                }
            )
    return pages_output


def extract_pdf_content(file):
    pdf_pages = get_page_elements(file)
    pdf_pages = check_and_fix_nested_tables(pdf_pages)
    pdf_pages = change_table_format(pdf_pages)
    return pdf_pages
