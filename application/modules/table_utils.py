"""Utility helpers for working with tables extracted from PDFs.

This module also contains the logic that decides when to fall back to the
image-based Gemini extractor for particularly complex tables (for example,
those with three header rows that the default parser struggles to normalize).
"""

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Tuple

from fuzzywuzzy import fuzz, process

from config_loader import load_config
from logging_config import get_logger

logger = get_logger(__name__)


def _log_context(*, pdf_path: Optional[str], page: Optional[int] = None, table: Optional[int] = None) -> dict:
    """Return a compact logging context for tidy, document-aware messages."""

    pdf_name = Path(pdf_path).name if pdf_path else None
    return {k: v for k, v in {"pdf": pdf_name, "page": page, "table": table}.items() if v is not None}


# --- geometry helpers -----------------------------------------------------

def interval_overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

# --- interval helpers -----------------------------------------------------

def _merge_intervals(intervals, min_gap=0.5):
    """Merge 1D intervals with small gaps tolerated (pt)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1] + min_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _edge_coverage_fraction(edge_min, edge_max, seg_intervals):
    """Total covered length / edge length after merging."""
    if edge_max <= edge_min:
        return 0.0
    clipped = []
    for a, b in seg_intervals:
        lo = max(edge_min, min(a, b))
        hi = min(edge_max, max(a, b))
        if hi > lo:
            clipped.append([lo, hi])
    merged = _merge_intervals(clipped)
    covered = sum(hi - lo for lo, hi in merged)
    return covered / (edge_max - edge_min)


# --- border checks --------------------------------------------------------

def has_vertical_border_strong(
    page,
    x_edge,
    y0,
    y1,
    x_tol: float = 2.5,
    min_len: float = 1.0,
    need_fraction: float = 0.5,
    edge_pad: float = 0.75,
    use_edges: bool = True,
):
    segs = []

    for ln in getattr(page, "lines", []):
        if abs(ln["x0"] - ln["x1"]) <= x_tol:
            x = 0.5 * (ln["x0"] + ln["x1"])
            if abs(x - x_edge) <= x_tol and abs(ln["y1"] - ln["y0"]) >= min_len:
                segs.append((min(ln["y0"], ln["y1"]), max(ln["y0"], ln["y1"])))

    for rc in getattr(page, "rects", []):
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        cx = 0.5 * (rx0 + rx1)
        if rw <= x_tol and rh >= min_len:
            if (
                abs(rx0 - x_edge) <= x_tol
                or abs(rx1 - x_edge) <= x_tol
                or abs(cx - x_edge) <= x_tol
            ):
                segs.append((min(ry0, ry1), max(ry0, ry1)))

    if use_edges:
        for ed in getattr(page, "edges", []):
            if abs(ed["x0"] - ed["x1"]) <= x_tol:
                x = 0.5 * (ed["x0"] + ed["x1"])
                if abs(x - x_edge) <= x_tol and abs(ed["y1"] - ed["y0"]) >= min_len:
                    segs.append((min(ed["y0"], ed["y1"]), max(ed["y0"], ed["y1"])))
        for cv in getattr(page, "curves", []):
            try:
                xs = [p[0] for p in cv["pts"]]
                ys = [p[1] for p in cv["pts"]]
                if (max(xs) - min(xs)) <= x_tol and (max(ys) - min(ys)) >= min_len:
                    cx = 0.5 * (max(xs) + min(xs))
                    if abs(cx - x_edge) <= x_tol:
                        segs.append((min(ys), max(ys)))
            except Exception:
                pass

    y_min = min(y0, y1) - edge_pad
    y_max = max(y0, y1) + edge_pad
    frac = _edge_coverage_fraction(y_min, y_max, segs)
    return frac >= need_fraction


def has_horizontal_border_strong(
    page,
    y_edge,
    x0,
    x1,
    y_tol: float = 2.5,
    min_len: float = 1.0,
    need_fraction: float = 0.5,
    edge_pad: float = 0.75,
    use_edges: bool = True,
):
    segs = []

    for ln in getattr(page, "lines", []):
        if abs(ln["y0"] - ln["y1"]) <= y_tol:
            y = 0.5 * (ln["y0"] + ln["y1"])
            if abs(y - y_edge) <= y_tol and abs(ln["x1"] - ln["x0"]) >= min_len:
                segs.append((min(ln["x0"], ln["x1"]), max(ln["x0"], ln["x1"])))

    for rc in getattr(page, "rects", []):
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        cy = 0.5 * (ry0 + ry1)
        if rh <= y_tol and rw >= min_len:
            if (
                abs(ry0 - y_edge) <= y_tol
                or abs(ry1 - y_edge) <= y_tol
                or abs(cy - y_edge) <= y_tol
            ):
                segs.append((min(rx0, rx1), max(rx0, rx1)))

    if use_edges:
        for ed in getattr(page, "edges", []):
            if abs(ed["y0"] - ed["y1"]) <= y_tol:
                y = 0.5 * (ed["y0"] + ed["y1"])
                if abs(y - y_edge) <= y_tol and abs(ed["x1"] - ed["x0"]) >= min_len:
                    segs.append((min(ed["x0"], ed["x1"]), max(ed["x0"], ed["x1"])))
        for cv in getattr(page, "curves", []):
            try:
                xs = [p[0] for p in cv["pts"]]
                ys = [p[1] for p in cv["pts"]]
                if (max(ys) - min(ys)) <= y_tol and (max(xs) - min(xs)) >= min_len:
                    cy = 0.5 * (max(ys) + min(ys))
                    if abs(cy - y_edge) <= y_tol:
                        segs.append((min(xs), max(xs)))
            except Exception:
                pass

    x_min = min(x0, x1) - edge_pad
    x_max = max(x0, x1) + edge_pad
    frac = _edge_coverage_fraction(x_min, x_max, segs)
    return frac >= need_fraction


# --- cell border APIs -----------------------------------------------------

def has_left_border(page, cell_bbox, x_tol: float = 2.5, y_overlap_ratio: float = 0.5):
    x0, top, x1, bottom = cell_bbox
    return has_vertical_border_strong(
        page,
        x_edge=x0,
        y0=top,
        y1=bottom,
        x_tol=x_tol,
        need_fraction=y_overlap_ratio,
        edge_pad=0.75,
    )


def has_upper_border(page, cell_bbox, y_tol: float = 2.5, x_overlap_ratio: float = 0.5):
    x0, top, x1, bottom = cell_bbox
    return has_horizontal_border_strong(
        page,
        y_edge=top,
        x0=x0,
        x1=x1,
        y_tol=y_tol,
        need_fraction=x_overlap_ratio,
        edge_pad=0.75,
    )


def build_cell_border_grid(page, table):
    row_edges = _extract_row_edges(table)
    col_edges = _extract_col_edges(table)

    if row_edges and not col_edges and hasattr(table, "cells") and table.cells:
        col_edges = _derive_col_edges_from_cells(table, x_tol=2.0)

    if not (row_edges and col_edges):
        return None

    n_rows = len(row_edges) - 1
    n_cols = len(col_edges) - 1
    cells = []
    for r in range(n_rows):
        top = float(row_edges[r])
        bottom = float(row_edges[r + 1])
        for c in range(n_cols):
            x0 = float(col_edges[c])
            x1 = float(col_edges[c + 1])
            bbox = (x0, top, x1, bottom)
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "bbox": bbox,
                    "has_left": has_left_border(page, bbox),
                    "has_top": has_upper_border(page, bbox),
                }
            )
    return {"n_rows": n_rows, "n_cols": n_cols, "cells": cells}


def _coerce_cell(c):
    if isinstance(c, dict):
        keys = {"x0", "top", "x1", "bottom", "row0", "row1", "col0", "col1"}
        if keys.issubset(c.keys()):
            return {
                "x0": float(c["x0"]),
                "top": float(c["top"]),
                "x1": float(c["x1"]),
                "bottom": float(c["bottom"]),
                "row0": int(c["row0"]),
                "row1": int(c["row1"]),
                "col0": int(c["col0"]),
                "col1": int(c["col1"]),
            }

    if hasattr(c, "_asdict"):
        return _coerce_cell(c._asdict())

    if isinstance(c, (list, tuple)):
        vals = list(c)
        n = len(vals)

        if n >= 8:
            try:
                x0, top, x1, bottom, row0, row1, col0, col1 = vals[:8]
                return {
                    "x0": float(x0),
                    "top": float(top),
                    "x1": float(x1),
                    "bottom": float(bottom),
                    "row0": int(row0),
                    "row1": int(row1),
                    "col0": int(col0),
                    "col1": int(col1),
                }
            except Exception:
                pass
            try:
                row0, row1, col0, col1, x0, top, x1, bottom = vals[:8]
                return {
                    "x0": float(x0),
                    "top": float(top),
                    "x1": float(x1),
                    "bottom": float(bottom),
                    "row0": int(row0),
                    "row1": int(row1),
                    "col0": int(col0),
                    "col1": int(col1),
                }
            except Exception:
                pass

        if n >= 6:
            floats = [
                v
                for v in vals
                if isinstance(v, float)
                or (isinstance(v, int) and not float(v).is_integer())
            ]
            ints = [v for v in vals if isinstance(v, int)]
            if len(floats) >= 4 and len(ints) >= 2:
                x0, top, x1, bottom = floats[:4]
                row0 = ints[0]
                col0 = ints[1]
                return {
                    "x0": float(x0),
                    "top": float(top),
                    "x1": float(x1),
                    "bottom": float(bottom),
                    "row0": int(row0),
                    "row1": int(row0 + 1),
                    "col0": int(col0),
                    "col1": int(col0 + 1),
                }

    for attrset in (
        ("x0", "top", "x1", "bottom", "row0", "row1", "col0", "col1"),
        ("x0", "y0", "x1", "y1", "row0", "row1", "col0", "col1"),
    ):
        try:
            return {
                "x0": float(getattr(c, attrset[0])),
                "top": float(getattr(c, attrset[1])),
                "x1": float(getattr(c, attrset[2])),
                "bottom": float(getattr(c, attrset[3])),
                "row0": int(getattr(c, attrset[4])),
                "row1": int(getattr(c, attrset[5])),
                "col0": int(getattr(c, attrset[6])),
                "col1": int(getattr(c, attrset[7])),
            }
        except Exception:
            pass

    return None


def _cluster_sorted(values, tol: float = 2.0):
    if not values:
        return []
    vals = sorted(values)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _as_float(v):
    if isinstance(v, (int, float)):
        return float(v)

    if isinstance(v, (list, tuple)) and v:
        if len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            y0, y1 = float(v[0]), float(v[1])
            if abs(y1 - y0) < 1e-6:
                return y0
        if len(v) >= 4 and isinstance(v[1], (int, float)):
            return float(v[1])
        try:
            return float(v[0])
        except Exception:
            pass

    for y_attrs in (("y",), ("y0", "y1"), ("top", "bottom")):
        try:
            if len(y_attrs) == 1 and hasattr(v, y_attrs[0]):
                return float(getattr(v, y_attrs[0]))
            if len(y_attrs) == 2 and hasattr(v, y_attrs[0]) and hasattr(v, y_attrs[1]):
                y0 = float(getattr(v, y_attrs[0]))
                y1 = float(getattr(v, y_attrs[1]))
                if abs(y1 - y0) < 1e-6:
                    return y0
                return (y0 + y1) / 2.0
        except Exception:
            pass
    for x_attrs in (("x",), ("x0", "x1"), ("left", "right")):
        try:
            if len(x_attrs) == 1 and hasattr(v, x_attrs[0]):
                return float(getattr(v, x_attrs[0]))
            if len(x_attrs) == 2 and hasattr(v, x_attrs[0]) and hasattr(v, x_attrs[1]):
                x0 = float(getattr(v, x_attrs[0]))
                x1 = float(getattr(v, x_attrs[1]))
                if abs(x1 - x0) < 1e-6:
                    return x0
                return (x0 + x1) / 2.0
        except Exception:
            pass
    return None


def _extract_row_edges(table, cluster_tol: float = 1.5):
    if not hasattr(table, "rows") or not table.rows:
        return None
    ys = []
    for r in table.rows:
        y = _as_float(r)
        if y is not None:
            ys.append(y)
    ys = _cluster_sorted(ys, tol=cluster_tol)
    return ys if len(ys) >= 2 else None


def _extract_col_edges(table, cluster_tol: float = 1.5):
    if not hasattr(table, "cols") or not table.cols:
        return None
    xs = []
    for c in table.cols:
        x = _as_float(c)
        if x is not None:
            xs.append(x)
    xs = _cluster_sorted(xs, tol=cluster_tol)
    return xs if len(xs) >= 2 else None


def _derive_col_edges_from_cells(table, x_tol: float = 2.0):
    if not (hasattr(table, "cells") and table.cells):
        return None
    xs = []
    for c in table.cells:
        try:
            x0, top, x1, bottom = c[:4]
            xs.append(float(x0))
            xs.append(float(x1))
        except Exception:
            continue
    edges = _cluster_sorted(xs, tol=x_tol)
    return edges if len(edges) >= 2 else None


def build_cell_border_grid_from_cells_only(page, table, cluster_tol: float = 2.0):
    if not (hasattr(table, "cells") and table.cells):
        return None

    xs, ys = [], []
    for c in table.cells:
        if not isinstance(c, (list, tuple)) or len(c) < 4:
            continue
        x0, top, x1, bottom = c[:4]
        xs.extend([float(x0), float(x1)])
        ys.extend([float(top), float(bottom)])

    if not xs or not ys:
        return None

    def _cluster(vals, tol):
        vals = sorted(vals)
        clusters = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - clusters[-1][-1]) <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [sum(c) / len(c) for c in clusters]

    col_edges = _cluster(xs, cluster_tol)
    row_edges = _cluster(ys, cluster_tol)
    if len(col_edges) < 2 or len(row_edges) < 2:
        return None

    n_rows = len(row_edges) - 1
    n_cols = len(col_edges) - 1
    cells = []
    for r in range(n_rows):
        top = row_edges[r]
        bottom = row_edges[r + 1]
        for c in range(n_cols):
            x0 = col_edges[c]
            x1 = col_edges[c + 1]
            bbox = (x0, top, x1, bottom)
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "bbox": bbox,
                    "has_left": has_left_border(page, bbox),
                    "has_top": has_upper_border(page, bbox),
                }
            )
    return {"n_rows": n_rows, "n_cols": n_cols, "cells": cells}


def bbox_intersects(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    horiz = not (ax1 <= bx0 or bx1 <= ax0)
    vert = not (ay1 <= by0 or by1 <= ay0)
    return horiz and vert


def get_all_tables(pdf_pages: Iterable[dict]) -> Tuple[List[Any], List[str]]:
    tables: List[Any] = []
    tables_flatted: List[str] = []

    append_table = tables.append
    append_flat = tables_flatted.append

    for page in pdf_pages:
        for table in page.get("tables", []):
            td = (
                table.get("table_dct")
                if isinstance(table, dict)
                else getattr(table, "table_dct", None)
            )
            append_table(td)

            if td is None:
                append_flat("")
                continue

            flat = "\n".join(
                " ".join(("" if cell is None else str(cell)) for cell in row) for row in td
            )
            flat = re.sub(r"\s+", " ", flat)

            append_flat(flat)

    return tables, tables_flatted


def clean_table_left_focus(table):
    for row in table:
        for i, cell in enumerate(row):
            if cell is None and i > 0:
                row[i] = row[i - 1]
    return table


def _lookup_borders(table, row_idx, col_idx):
    grid = table.get("grid")
    if not grid or not grid.get("cells"):
        return (None, None)
    for c in grid["cells"]:
        if c.get("row") == row_idx and c.get("col") == col_idx:
            return (c.get("has_left"), c.get("has_top"))
    return (None, None)


def clean_table(table):
    table_dct = copy.deepcopy(table.get("table_dct", []))

    for i, row in enumerate(table_dct):
        for j, cell in enumerate(row):
            if cell is None:
                has_left, has_upper = _lookup_borders(table, i, j)

                if has_upper is False and i > 0:
                    up_val = table_dct[i - 1][j]
                    if up_val is not None:
                        table_dct[i][j] = up_val
                        continue

                if has_left is False and j > 0:
                    left_val = table_dct[i][j - 1]
                    if left_val is not None:
                        table_dct[i][j] = left_val
                        continue

    return table_dct


def change_to_records(table):
    headers = table[0]
    data_rows = table[1:]
    return [dict(zip(headers, row)) for row in data_rows]


def fix_nested_table(
    table,
    tables,
    tables_flatted,
    *,
    current_index: Optional[int] = None,
):
    """Attempt to repair genuine nested tables while avoiding false positives.

    A previous implementation occasionally replaced long paragraph cells with the
    table itself (because the fuzzy matcher would pick the current table's flat
    text), which then caused `_table_has_nested_cells` to flag the page as
    nested. We now skip self-matches, require a high confidence score, and only
    embed candidate tables that have multiple rows (a stronger signal of an
    actual nested grid) so simple headers are not rewritten as nested content.
    """

    for index_row, row in enumerate(table):
        for index_col, cell in enumerate(row):
            if cell and isinstance(cell, str) and len(cell) > 20:
                one, score = process.extractOne(
                    cell, tables_flatted, scorer=fuzz.token_sort_ratio
                )
                if score <= 88:
                    continue

                idx = tables_flatted.index(one)
                if current_index is not None and idx == current_index:
                    # Avoid replacing a cell with the current table itself.
                    continue

                candidate = tables[idx]
                if not candidate or len(candidate) < 2:
                    # Single-row candidates are unlikely to be true nested tables.
                    continue

                cleaned_table = clean_table_left_focus(candidate)
                cleaned_table = change_to_records(cleaned_table)
                table[index_row][index_col] = cleaned_table

    return table


def _count_non_empty_cells(row: Iterable[Any]) -> int:
    """Return the number of cells that contain some non-empty content."""

    return sum(1 for cell in row if cell not in (None, ""))


def _table_has_nested_cells(table_data: Iterable[Iterable[Any]]) -> bool:
    """Return True if any cell already contains a nested structure.

    Cells that hold ``list`` or ``dict`` instances are treated as nested tables
    that should trigger the Gemini fallback, because the built-in normalizer
    cannot reliably flatten deeply nested content. This check deliberately runs
    *after* ``check_and_fix_nested_tables`` to catch any residual nested
    structures the repair pass could not fully unwind.
    """

    for row in table_data or []:
        for cell in row or []:
            if isinstance(cell, (list, dict)):
                return True
    return False


def _table_shape_summary(table_data: Iterable[Iterable[Any]]) -> Tuple[int, int]:
    """Return ``(row_count, max_col_count)`` for quick complexity checks."""

    row_count = 0
    max_cols = 0
    for row in table_data or []:
        row_count += 1
        if isinstance(row, dict):
            max_cols = max(max_cols, len(row))
        elif isinstance(row, (list, tuple)):
            max_cols = max(max_cols, len(row))
    return row_count, max_cols


def _looks_like_plain_text_block(table_data: Iterable[Iterable[Any]]) -> bool:
    """Heuristic: reject Gemini fallback for paragraph-shaped "tables".

    pdfplumber sometimes mis-detects framed paragraphs as tiny tables. When the
    structure is shallow (<=3 rows, <=2 columns) and dominated by a single long
    cell, treat it as plain text so we do not route it through the complex-table
    or Gemini logic.
    """

    rows, cols = _table_shape_summary(table_data)
    if rows <= 3 and cols <= 1:
        return True

    if rows <= 3 and cols <= 2:
        non_empty_cells = 0
        total_chars = 0
        for row in table_data or []:
            for cell in row or []:
                if cell not in (None, ""):
                    non_empty_cells += 1
                    total_chars += len(str(cell))
        # A single dense cell or a couple of long cells suggests prose rather
        # than a true table structure.
        if non_empty_cells <= 3 and total_chars >= 120:
            return True

    return False


def _looks_like_definition_table(table_data: Iterable[Iterable[Any]]) -> bool:
    """Detect simple 1–2 column definition layouts so we skip Gemini.

    Some pages contain a short banner plus long explanatory paragraphs laid out
    in one or two columns. These are structurally simple and should stay on the
    manual parsing path even if the top rows superficially resemble stacked
    headers. The heuristic looks for:

    * At most two columns.
    * At least one very short column paired with a consistently longer column.
    * Long-form text (>= 80 chars) in the wider column.
    """

    rows, cols = _table_shape_summary(table_data)
    if cols == 0 or cols > 2 or rows == 0:
        return False

    col_lengths: list[list[int]] = [[] for _ in range(cols)]
    for row in table_data or []:
        if not isinstance(row, (list, tuple)):
            return False
        for idx in range(cols):
            val = row[idx] if idx < len(row) else None
            if val not in (None, ""):
                col_lengths[idx].append(len(str(val)))

    if not any(col_lengths):
        return False

    # Require the "long" column to carry meaningful paragraph text.
    long_col_has_body = any(any(l >= 80 for l in lengths) for lengths in col_lengths)
    if not long_col_has_body:
        return False

    # Compare average lengths to ensure we truly have a skinny label column.
    averages = [
        (sum(lengths) / len(lengths)) if lengths else 0.0 for lengths in col_lengths
    ]
    if len(averages) == 1:
        # Single-column framed paragraphs should stay as narrative-style tables.
        return True

    avg_short = min(averages)
    avg_long = max(averages)
    return avg_long >= max(120, 2.5 * max(avg_short, 1.0))


def check_double_header(table_data):
    """Detect the presence of two header rows.

    The previous implementation treated any table with a blank first row and a
    populated second row as a double header. That aggressively promoted body
    rows to headers and produced mangled structures. We now require that both
    header rows contain meaningful content and that at least one column in the
    second row fills a blank in the first row.
    """

    if len(table_data) < 2:
        return False

    header_row_1 = table_data[0]
    header_row_2 = table_data[1]

    non_empty_row_1 = _count_non_empty_cells(header_row_1)
    non_empty_row_2 = _count_non_empty_cells(header_row_2)
    if non_empty_row_1 == 0 or non_empty_row_2 == 0:
        # Do not mark as double-header if one of the header rows is entirely
        # empty; this usually means the detector misaligned on the first data
        # row.
        return False

    has_fill_down = any(
        (cell_1 in (None, "")) and (cell_2 not in (None, ""))
        for cell_1, cell_2 in zip(header_row_1, header_row_2)
    )

    return bool(has_fill_down)


def check_triple_header(table_data):
    """Detect the presence of three header rows.

    We only consider a triple-header table if we already pass the double-header
    check and every header row has at least one non-empty cell. This prevents
    sparse header rows from being misclassified as multi-level headers.
    """

    try:
        if len(table_data) < 3:
            return False

        if not check_double_header(table_data):
            return False

        header_row_2 = table_data[1]
        header_row_3 = table_data[2]

        if _count_non_empty_cells(header_row_3) == 0:
            return False

        for i, item in enumerate(header_row_2):
            __, has_upper = _lookup_borders(table_data, row_idx=1, col_idx=i)  # header_row_2[i] has top border
            if item in (None, "") and header_row_3[i] not in (None, "") and i > 0 and has_upper:
                return True
    except Exception as e:
        logger.debug(f"Error in check_triple_header: {e}")

    return False


def _has_more_than_double_headers(table_data: list) -> bool:
    """Detect multi-level headers deeper than two rows.

    Some complex layouts use three or more stacked header rows without fitting
    the stricter ``check_triple_header`` heuristic. If the first three rows all
    carry content and the table also looks like a double-header, treat it as a
    complex multi-header table so Gemini can re-extract cleaner rows.
    """

    if len(table_data) < 3:
        return False

    if not check_double_header(table_data):
        return False

    def _row_is_probably_header(row: Iterable[Any]) -> bool:
        """Return True when *row* is short enough to plausibly be a header.

        Long, paragraph-style cells (e.g., hundreds of characters or multiple
        newline breaks) typically indicate body content. Filtering them out
        prevents framed prose with a small title row from being mis-classified
        as a deep multi-header table that would wrongly trigger the Gemini
        fallback.
        """

        texts = [str(cell).strip() for cell in row or [] if cell not in (None, "")]
        if not texts:
            return False

        longest = max(len(t) for t in texts)
        total = sum(len(t) for t in texts)
        newline_breaks = sum(t.count("\n") for t in texts)

        if longest > 120 or total > 200:
            return False
        if newline_breaks >= 2:
            return False
        return True

    headerish_first_three = [_row_is_probably_header(row) for row in table_data[:3]]
    if not all(headerish_first_three):
        return False

    # Require multiple non-empty header cells across the top rows so a single
    # title bar plus a long paragraph row does not masquerade as a multi-level
    # header.
    non_empty_top_rows = sum(_count_non_empty_cells(row) for row in table_data[:3])
    return non_empty_top_rows >= 4


def process_double_header_table(table_data):
    if len(table_data) < 3:
        return []

    header_row_1 = table_data[0]
    header_row_2 = table_data[1]

    filled_header_1 = []
    last_header = ""
    for idx, header in enumerate(header_row_1):
        if header:
            last_header = str(header).replace("\n", " ")
        elif not last_header:
            last_header = f"Column {idx + 1}"
        filled_header_1.append(last_header)

    flattened_headers = []
    for i, sub_header in enumerate(header_row_2):
        main_header = filled_header_1[i] if i < len(filled_header_1) else f"Column {i + 1}"
        if sub_header:
            clean_sub_header = str(sub_header).replace("\n", " ")
            flattened_headers.append(f"{main_header} - {clean_sub_header}")
        else:
            flattened_headers.append(main_header or f"Column {i + 1}")

    records = []
    data_rows = table_data[2:]

    for row in data_rows:
        if not row or row[0] is None:
            continue
        # Zip after normalizing the row length so trailing empty cells do not
        # disappear from the record.
        normalized_row = list(row) + [None] * (len(flattened_headers) - len(row))
        record = dict(zip(flattened_headers, normalized_row))
        records.append(record)

    return records


def process_triple_header_table(table_data):
    if len(table_data) < 4:
        return []

    header_row_1 = table_data[0]
    header_row_2 = table_data[1]
    header_row_3 = table_data[2]

    filled_header_1 = []
    last_header = ""
    for idx, header in enumerate(header_row_1):
        if header:
            last_header = str(header).replace("\n", " ")
        elif not last_header:
            last_header = f"Column {idx + 1}"
        filled_header_1.append(last_header)

    filled_header_2 = []
    last_header = ""
    for i, header in enumerate(header_row_2):
        if i > 0 and filled_header_1[i] != filled_header_1[i - 1]:
            last_header = ""
        if header:
            last_header = str(header).replace("\n", " ")
        elif not last_header:
            last_header = f"Column {i + 1}"
        filled_header_2.append(last_header)

    flattened_headers = []
    for i in range(len(header_row_1)):
        h1 = filled_header_1[i] if i < len(filled_header_1) else f"Column {i + 1}"
        h2 = filled_header_2[i] if i < len(filled_header_2) else ""
        h3_raw = header_row_3[i] if i < len(header_row_3) else ""
        h3 = str(h3_raw).replace("\n", " ") if h3_raw else ""
        parts = [part for part in [h1, h2, h3] if part]
        unique_parts = list(dict.fromkeys(parts))
        final_header = " - ".join(unique_parts)
        flattened_headers.append(final_header)

    records = []
    data_rows = table_data[3:]

    for row in data_rows:
        if not row or row[0] is None:
            continue

        normalized_row = list(row) + [None] * (len(flattened_headers) - len(row))
        record = dict(zip(flattened_headers, normalized_row))
        records.append(record)

    return records


def check_and_fix_nested_tables(pdf_pages):
    """Attempt to flatten detected nested tables before header normalization.

    The helper uses fuzzy matching to repair obviously nested grids but may not
    fully resolve every embedded structure. Downstream consumers therefore keep
    a secondary guard (``_table_has_nested_cells``) to route any remaining
    nested content through the Gemini extractor when needed.
    """

    tables, tables_flatted = get_all_tables(pdf_pages)
    table_counter = 0
    for page_idx, page in enumerate(pdf_pages):
        if page["tables"]:
            for i, dct in enumerate(page["tables"]):
                fixed_table = dct.get("table_dct")
                if fixed_table:
                    try:
                        fixed_table = fix_nested_table(
                            fixed_table,
                            tables,
                            tables_flatted,
                            current_index=table_counter,
                        )
                    except Exception as e:
                        logger.debug(
                            "Error in fix_nested_table for page %s, table %s: %s",
                            page_idx + 1,
                            i,
                            e,
                        )
                pdf_pages[page_idx]["tables"][i]["table_dct"] = fixed_table
                table_counter += 1
    return pdf_pages


def _extract_json_fragment(text: str) -> Optional[Any]:
    """Pull the first JSON-like fragment out of a string.

    Gemini sometimes prepends or appends natural language around the JSON
    payload. This helper searches for the first ``[`` or ``{`` and attempts to
    parse up to the last matching bracket/brace so we can recover the table rows
    instead of failing hard.
    """

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


def _sanitize_gemini_rows(rows: Iterable[Any]) -> list:
    """Align Gemini rows with the manual parser's expectations."""

    sanitized = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        clean_row = {}
        for key, value in row.items():
            clean_key = str(key) if key is not None else ""
            clean_row[clean_key] = None if value is None else str(value)
        if clean_row:
            sanitized.append(clean_row)
    return sanitized


def _parse_gemini_table_response(model_response: Any) -> Optional[list]:
    """Attempt to coerce the Gemini response into a list of sanitized rows."""

    if model_response is None:
        return None

    # If Gemini already returned a Python object, accept it directly.
    if isinstance(model_response, list):
        return _sanitize_gemini_rows(model_response)
    if isinstance(model_response, dict):
        maybe_rows = model_response.get("rows") or model_response.get("table")
        if isinstance(maybe_rows, list):
            return _sanitize_gemini_rows(maybe_rows)
        return None

    raw_text = getattr(model_response, "text", None) or getattr(
        model_response, "content", None
    )
    if raw_text is None:
        raw_text = model_response
    if not isinstance(raw_text, str):
        return None

    parsed = None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = _extract_json_fragment(raw_text)

    if isinstance(parsed, list):
        return _sanitize_gemini_rows(parsed)
    if isinstance(parsed, dict):
        for key in ("rows", "table"):
            if isinstance(parsed.get(key), list):
                return _sanitize_gemini_rows(parsed[key])
    return None


def _maybe_use_gemini_for_table(
    *,
    pdf_path: Optional[str],
    page_number: int,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """Extract a page with Gemini, returning the structured response when possible.

    Gemini is invoked for *every* page containing a table so the model can own
    the cell-level chunking. Any failure returns ``None`` so ingestion can still
    proceed using the legacy parser documented below.
    """

    if not pdf_path:
        return None

    try:
        from ingest.table_image_extractor import extract_table_with_gemini
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning(
            "Gemini extractor unavailable",
            extra=_log_context(pdf_path=pdf_path, page=page_number),
            exc_info=exc,
        )
        return None

    try:
        result = extract_table_with_gemini(
            pdf_path=pdf_path,
            page_number=page_number,
            prompt=prompt,
            model_name=model_name,
            api_key=api_key,
        )
    except Exception as exc:  # pragma: no cover - remote call issues
        logger.warning(
            "Gemini extraction failed; falling back to default parser",
            extra=_log_context(pdf_path=pdf_path, page=page_number),
            exc_info=exc,
        )
        return None

    if result.get("rows") or result.get("chunks"):
        logger.info(
            "Using Gemini image extractor output",
            extra=_log_context(pdf_path=pdf_path, page=page_number),
        )
    return result


def _page_needs_gemini(pdf_page: dict) -> bool:
    """Lightweight heuristic to decide when Gemini should run in fallback mode."""

    for tbl in pdf_page.get("tables") or []:
        records = tbl.get("table_dct")
        if not records:
            return True
        try:
            if _table_has_nested_cells(records) or _has_more_than_double_headers(records):
                return True
        except Exception:
            continue
    return False

def change_table_format(
    pdf_pages,
    *,
    pdf_path: Optional[str] = None,
    use_gemini_mode: Literal["use", "skip", "fallback"] = "use",
    use_gemini_for_complex: bool = False,
    gemini_prompt: Optional[str] = None,
    gemini_model: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
):
    """Normalize table rows into records, invoking Gemini for every table page."""

    # ``use_gemini_for_complex`` is retained for backward compatibility, but the
    # Gemini image extractor now runs for any page containing a table so the
    # model can enforce the atomic cell-chunk rule directly.

    # Resolve Gemini defaults from the active config so we mirror app.py usage.
    config = load_config()
    api_key_env = getattr(config, "google_api_key_name", "GEMINI_API_KEY")
    resolved_api_key = gemini_api_key or os.environ.get(api_key_env)
    if not resolved_api_key and api_key_env != "GEMINI_API_KEY":
        resolved_api_key = os.environ.get("GEMINI_API_KEY")
    resolved_model = gemini_model or getattr(config, "gemini_chat_model", None)

    # Avoid calling Gemini multiple times for the same page. The image extractor
    # renders an entire page (not a specific table bbox), so one invocation per
    # page is sufficient and much cheaper than re-rendering for every table on
    # that page.
    gemini_page_cache: dict[tuple[Optional[str], int], Optional[dict]] = {}

    for page_idx, page in enumerate(pdf_pages):
        mode = (use_gemini_mode or "use").lower()
        if page["tables"]:
            logger.info(
                "Normalizing page tables",
                extra=_log_context(pdf_path=pdf_path, page=page_idx + 1),
            )
            should_use_gemini = mode == "use" or (
                mode == "fallback" and (_page_needs_gemini(page) or use_gemini_for_complex)
            )

            gemini_result: Optional[dict] = None
            if should_use_gemini:
                cache_key = (pdf_path, page_idx + 1)
                if cache_key in gemini_page_cache:
                    gemini_result = gemini_page_cache[cache_key]
                else:
                    gemini_result = _maybe_use_gemini_for_table(
                        pdf_path=pdf_path,
                        page_number=page_idx + 1,
                        prompt=gemini_prompt,
                        model_name=resolved_model,
                        api_key=resolved_api_key,
                    )
                    gemini_page_cache[cache_key] = gemini_result

            # Gemini now owns the chunking; store chunks on the page for later
            # ingestion and keep the legacy parser below as documentation only.
            if gemini_result and gemini_result.get("chunks"):
                page["gemini_chunks"] = gemini_result.get("chunks")

            for i, dct in enumerate(page["tables"]):
                records = dct.get("table_dct")
                if records:
                    try:
                        is_double_header = check_double_header(records)
                        is_triple_header = (
                            check_triple_header(records) if is_double_header else False
                        )
                    except Exception as exc:
                        logger.warning(
                            "Header detection failed; treating table as simple",  # noqa: TRY400
                            extra=_log_context(
                                pdf_path=pdf_path, page=page_idx + 1, table=i
                            ),
                            exc_info=exc,
                        )
                        is_double_header = False
                        is_triple_header = False

                    definition_table = _looks_like_definition_table(records)
                    has_nested_cells = _table_has_nested_cells(records)
                    if has_nested_cells:
                        logger.debug(
                            "Nested cells detected post-repair; enabling Gemini fallback",
                            extra=_log_context(
                                pdf_path=pdf_path, page=page_idx + 1, table=i
                            ),
                        )
                    has_multi_level_header = (
                        is_triple_header or _has_more_than_double_headers(records)
                    )

                    row_count, col_count = _table_shape_summary(records)
                    looks_like_plain_text = _looks_like_plain_text_block(records)

                    table = clean_table(dct)
                    gemini_rows: Optional[list] = None
                    gemini_tables: Optional[list] = None

                    # Always use Gemini when a table exists on the page. Gemini
                    # decides chunking (cell-level + narrative) so manual parsing
                    # is only used if the model output is missing.
                    if gemini_result:
                        gemini_rows = gemini_result.get("rows")
                        gemini_tables = gemini_result.get("tables")

                    """
                    Legacy Gemini gating (retained for documentation):
                    if (
                        use_gemini_for_complex
                        and not looks_like_plain_text
                        and not definition_table
                        and (has_multi_level_header or has_nested_cells)
                        and col_count > 1
                    ):
                        ...
                    elif (
                        (looks_like_plain_text or definition_table)
                        and (has_multi_level_header or has_nested_cells)
                    ):
                        logger.debug("Skipping Gemini …")
                    """

                    try:
                        if gemini_tables and isinstance(gemini_tables, list):
                            matching = [
                                tbl for tbl in gemini_tables if tbl.get("table_index") == i
                            ]
                            if matching:
                                records = matching[0].get("rows") or records
                        elif gemini_rows and all(
                            isinstance(r, dict) for r in gemini_rows
                        ):
                            records = gemini_rows
                        elif is_triple_header:
                            records = process_triple_header_table(table)
                        elif is_double_header:
                            records = process_double_header_table(table)
                        else:
                            headers = [
                                (str(h).replace("\n", " ") if h else f"Column {idx + 1}")
                                for idx, h in enumerate(table[0])
                            ]
                            data_rows = table[1:]
                            records = [dict(zip(headers, row)) for row in data_rows]

                        if not records:
                            # The table likely was misclassified as a multi-header
                            # structure. Fall back to a single-header interpretation
                            # to avoid returning malformed nested dictionaries.
                            headers = [
                                (str(h).replace("\n", " ") if h else f"Column {idx + 1}")
                                for idx, h in enumerate(table[0])
                            ]
                            data_rows = table[1:]
                            records = [dict(zip(headers, row)) for row in data_rows]
                    except Exception as e:
                        logger.debug(
                            "Error in change_table_format for page %s, table %s: %s",
                            page_idx + 1,
                            i,
                            e,
                        )
                        try:
                            headers = table[0]
                            data_rows = table[1:]
                            records = [dict(zip(headers, row)) for row in data_rows]
                        except Exception as e2:
                            logger.debug("  Fallback also failed: %s", e2)
                            records = [{"table": table}]
                pdf_pages[page_idx]["tables"][i]["table_dct"] = records
    return pdf_pages
