"""Utility helpers for working with tables extracted from PDFs."""

import copy
import re
from typing import Any, Iterable, List, Tuple

from fuzzywuzzy import fuzz, process

from logging_config import get_logger

logger = get_logger(__name__)


# --- geometry helpers -----------------------------------------------------

def interval_overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# --- stroke finders -------------------------------------------------------

def _has_vertical_border_at_x(
    page,
    x_target,
    y0,
    y1,
    x_tol: float = 1.5,
    min_len: float = 2.0,
    min_overlap_ratio: float = 0.6,
) -> bool:
    span = abs(y1 - y0)
    min_overlap = max(min_len, span * min_overlap_ratio)

    for ln in getattr(page, "lines", []):
        if abs(ln["x0"] - ln["x1"]) <= x_tol:
            x = (ln["x0"] + ln["x1"]) / 2.0
            if abs(x - x_target) <= x_tol:
                overlap = interval_overlap(y0, y1, ln["y0"], ln["y1"])
                if overlap >= min_overlap and abs(ln["y1"] - ln["y0"]) >= min_len:
                    return True

    for rc in getattr(page, "rects", []):
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        if rw <= x_tol and rh >= min_len:
            if abs(rx0 - x_target) <= x_tol or abs(rx1 - x_target) <= x_tol:
                overlap = interval_overlap(y0, y1, ry0, ry1)
                if overlap >= min_overlap:
                    return True

    return False


def _has_horizontal_border_at_y(
    page,
    y_target,
    x0,
    x1,
    y_tol: float = 1.5,
    min_len: float = 2.0,
    min_overlap_ratio: float = 0.6,
) -> bool:
    span = abs(x1 - x0)
    min_overlap = max(min_len, span * min_overlap_ratio)

    for ln in getattr(page, "lines", []):
        if abs(ln["y0"] - ln["y1"]) <= y_tol:
            y = (ln["y0"] + ln["y1"]) / 2.0
            if abs(y - y_target) <= y_tol:
                overlap = interval_overlap(x0, x1, ln["x0"], ln["x1"])
                if overlap >= min_overlap and abs(ln["x1"] - ln["x0"]) >= min_len:
                    return True

    for rc in getattr(page, "rects", []):
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        if rh <= y_tol and rw >= min_len:
            if abs(ry0 - y_target) <= y_tol or abs(ry1 - y_target) <= y_tol:
                overlap = interval_overlap(x0, x1, rx0, rx1)
                if overlap >= min_overlap:
                    return True

    return False


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


def fix_nested_table(table, tables, tables_flatted):
    for index_row, row in enumerate(table):
        for index_col, cell in enumerate(row):
            if cell and isinstance(cell, str) and len(cell) > 20:
                one, score = process.extractOne(
                    cell, tables_flatted, scorer=fuzz.token_sort_ratio
                )
                if score > 80:
                    idx = tables_flatted.index(one)
                    cleaned_table = clean_table_left_focus(tables[idx])
                    cleaned_table = change_to_records(cleaned_table)
                    table[index_row][index_col] = cleaned_table
    return table


def check_double_header(table_data):
    if len(table_data) < 2:
        return False

    header_row_1 = table_data[0]
    header_row_2 = table_data[1]

    for i, item in enumerate(header_row_1):
        if item is None and header_row_2[i] is not None:
            return True

    if all(isinstance(h, str) for h in header_row_1):
        return False

    return False


def check_triple_header(table_data):
    try:
        if len(table_data) < 3:
            return False

        header_row_2 = table_data[1]
        header_row_3 = table_data[2]

        for i, item in enumerate(header_row_2):
            if item is None and header_row_3[i] is not None and i > 0:
                return True

        if all(isinstance(h, str) for h in header_row_2):
            return False
    except Exception as e:
        logger.debug(f"Error in check_triple_header: {e}")

    return False


def process_double_header_table(table_data):
    if len(table_data) < 3:
        return []

    header_row_1 = table_data[0]
    header_row_2 = table_data[1]

    filled_header_1 = []
    last_header = ""
    for header in header_row_1:
        if header is not None:
            last_header = header
        filled_header_1.append(last_header)

    flattened_headers = []
    for i, sub_header in enumerate(header_row_2):
        main_header = filled_header_1[i]
        if sub_header:
            clean_sub_header = sub_header.replace("\n", " ")
            flattened_headers.append(f"{main_header} - {clean_sub_header}")
        else:
            flattened_headers.append(main_header)

    records = []
    data_rows = table_data[2:]

    for row in data_rows:
        if not row or row[0] is None:
            continue
        record = dict(zip(flattened_headers, row))
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
    for header in header_row_1:
        if header is not None:
            last_header = header.replace("\n", " ")
        filled_header_1.append(last_header)

    filled_header_2 = []
    last_header = ""
    for i, header in enumerate(header_row_2):
        if i > 0 and filled_header_1[i] != filled_header_1[i - 1]:
            last_header = ""
        if header is not None:
            last_header = header.replace("\n", " ")
        filled_header_2.append(last_header)

    flattened_headers = []
    for i in range(len(header_row_1)):
        h1 = filled_header_1[i]
        h2 = filled_header_2[i]
        h3 = header_row_3[i].replace("\n", " ") if header_row_3[i] else ""
        parts = [part for part in [h1, h2, h3] if part]
        unique_parts = list(dict.fromkeys(parts))
        final_header = " - ".join(unique_parts)
        flattened_headers.append(final_header)

    records = []
    data_rows = table_data[3:]

    for row in data_rows:
        if not row or row[0] is None:
            continue

        record = dict(zip(flattened_headers, row))
        records.append(record)

    return records


def check_and_fix_nested_tables(pdf_pages):
    tables, tables_flatted = get_all_tables(pdf_pages)
    for page_idx, page in enumerate(pdf_pages):
        if page["tables"]:
            for i, dct in enumerate(page["tables"]):
                fixed_table = dct.get("table_dct")
                if fixed_table:
                    try:
                        fixed_table = fix_nested_table(fixed_table, tables, tables_flatted)
                    except Exception as e:
                        logger.debug(
                            "Error in fix_nested_table for page %s, table %s: %s",
                            page_idx + 1,
                            i,
                            e,
                        )
                pdf_pages[page_idx]["tables"][i]["table_dct"] = fixed_table
    return pdf_pages


def change_table_format(pdf_pages):
    for page_idx, page in enumerate(pdf_pages):
        if page["tables"]:
            for i, dct in enumerate(page["tables"]):
                records = dct.get("table_dct")
                if records:
                    is_double_header = check_double_header(records)
                    is_triple_header = (
                        check_triple_header(records) if is_double_header else False
                    )

                    table = clean_table(dct)
                    try:
                        if is_triple_header:
                            records = process_triple_header_table(table)
                        elif is_double_header:
                            records = process_double_header_table(table)
                        else:
                            headers = table[0]
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
