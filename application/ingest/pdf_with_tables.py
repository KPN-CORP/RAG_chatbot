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

logger = get_logger(__name__)


def get_page_elements(pdf_path):
    pages_output = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
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
                grid = build_cell_border_grid(page, t)
                if grid is None:
                    grid = build_cell_border_grid_from_cells_only(page, t)
                tables.append(
                    {
                        "table_index": t_idx,
                        "bbox": tbbox,
                        "rows": rows,
                        "grid": grid,
                    }
                )

            words = page.extract_words()
            narrative_words = []
            for w in words:
                wbox = (
                    float(w["x0"]),
                    float(w["top"]),
                    float(w["x1"]),
                    float(w["bottom"]),
                )
                in_table = any(bbox_intersects(wbox, tb["bbox"]) for tb in table_bboxes)
                if not in_table:
                    narrative_words.append(w)

            lines = {}
            tol = 3
            for w in narrative_words:
                top = round(float(w["top"]) / tol) * tol
                lines.setdefault(top, []).append((float(w["x0"]), w["text"]))

            sorted_lines = [
                " ".join([t[1] for t in sorted(line, key=lambda x: x[0])])
                for top, line in sorted(lines.items())
            ]

            narrative_text = "\n".join(sorted_lines).strip()
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
