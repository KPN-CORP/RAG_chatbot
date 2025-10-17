# from collections import Counter
import re
import copy
from typing import Any, Iterable, List, Tuple

import pdfplumber
from fuzzywuzzy import fuzz, process
from logging_config import get_logger

logger = get_logger(__name__)


# --- geometry helpers (add if you don't already have them) ---
def interval_overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

# --- stroke finders (new) ---
def _has_vertical_border_at_x(page, x_target, y0, y1,
                              x_tol=1.5, min_len=2.0, min_overlap_ratio=0.6) -> bool:
    span = abs(y1 - y0)
    min_overlap = max(min_len, span * min_overlap_ratio)

    # lines (near-vertical)
    for ln in page.lines:
        if abs(ln["x0"] - ln["x1"]) <= x_tol:
            x = (ln["x0"] + ln["x1"]) / 2.0
            if abs(x - x_target) <= x_tol:
                overlap = interval_overlap(y0, y1, ln["y0"], ln["y1"])
                if overlap >= min_overlap and abs(ln["y1"] - ln["y0"]) >= min_len:
                    return True

    # rects (thin vertical)
    for rc in page.rects:
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        if rw <= x_tol and rh >= min_len:
            if abs(rx0 - x_target) <= x_tol or abs(rx1 - x_target) <= x_tol:
                overlap = interval_overlap(y0, y1, ry0, ry1)
                if overlap >= min_overlap:
                    return True

    return False

def _has_horizontal_border_at_y(page, y_target, x0, x1,
                                y_tol=1.5, min_len=2.0, min_overlap_ratio=0.6) -> bool:
    span = abs(x1 - x0)
    min_overlap = max(min_len, span * min_overlap_ratio)

    # lines (near-horizontal)
    for ln in page.lines:
        if abs(ln["y0"] - ln["y1"]) <= y_tol:
            y = (ln["y0"] + ln["y1"]) / 2.0
            if abs(y - y_target) <= y_tol:
                overlap = interval_overlap(x0, x1, ln["x0"], ln["x1"])
                if overlap >= min_overlap and abs(ln["x1"] - ln["x0"]) >= min_len:
                    return True

    # rects (thin horizontal)
    for rc in page.rects:
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        if rh <= y_tol and rw >= min_len:
            if abs(ry0 - y_target) <= y_tol or abs(ry1 - y_target) <= y_tol:
                overlap = interval_overlap(x0, x1, rx0, rx1)
                if overlap >= min_overlap:
                    return True

    return False

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
    # Clip segments to edge and merge
    clipped = []
    for a, b in seg_intervals:
        lo = max(edge_min, min(a, b))
        hi = min(edge_max, max(a, b))
        if hi > lo:
            clipped.append([lo, hi])
    merged = _merge_intervals(clipped)
    covered = sum(hi - lo for lo, hi in merged)
    return covered / (edge_max - edge_min)

def has_vertical_border_strong(page, x_edge, y0, y1,
                               x_tol=2.5, min_len=1.0, need_fraction=0.5,
                               edge_pad=0.75, use_edges=True):
    """
    Strong vertical-border check:
    - Accepts lines, skinny rects, edges/curves (optional)
    - Merges fragmented segments
    - Allows slight padding around the edge
    """
    segs = []

    # Lines (near-vertical)
    for ln in getattr(page, "lines", []):
        if abs(ln["x0"] - ln["x1"]) <= x_tol:
            x = 0.5 * (ln["x0"] + ln["x1"])
            if abs(x - x_edge) <= x_tol:
                if abs(ln["y1"] - ln["y0"]) >= min_len:
                    segs.append((min(ln["y0"], ln["y1"]), max(ln["y0"], ln["y1"])))

    # Rects (thin vertical); also check center alignment
    for rc in getattr(page, "rects", []):
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        cx = 0.5 * (rx0 + rx1)
        if rw <= x_tol and rh >= min_len:
            if (abs(rx0 - x_edge) <= x_tol) or (abs(rx1 - x_edge) <= x_tol) or (abs(cx - x_edge) <= x_tol):
                segs.append((min(ry0, ry1), max(ry0, ry1)))

    # Optional: edges/curves (pdfplumber sometimes exposes these)
    if use_edges:
        for ed in getattr(page, "edges", []):
            # edges are dicts with x0,x1,y0,y1 too
            if abs(ed["x0"] - ed["x1"]) <= x_tol:
                x = 0.5 * (ed["x0"] + ed["x1"])
                if abs(x - x_edge) <= x_tol and abs(ed["y1"] - ed["y0"]) >= min_len:
                    segs.append((min(ed["y0"], ed["y1"]), max(ed["y0"], ed["y1"])))
        for cv in getattr(page, "curves", []):
            # curves store points; optional: approximate as min/max x/y segments
            try:
                xs = [p[0] for p in cv["pts"]]; ys = [p[1] for p in cv["pts"]]
                if (max(xs) - min(xs)) <= x_tol and (max(ys) - min(ys)) >= min_len:
                    cx = 0.5 * (max(xs) + min(xs))
                    if abs(cx - x_edge) <= x_tol:
                        segs.append((min(ys), max(ys)))
            except Exception:
                pass

    # Combine coverage along the edge (with small padding)
    y_min = min(y0, y1) - edge_pad
    y_max = max(y0, y1) + edge_pad
    frac = _edge_coverage_fraction(y_min, y_max, segs)
    return frac >= need_fraction

def has_horizontal_border_strong(page, y_edge, x0, x1,
                                 y_tol=2.5, min_len=1.0, need_fraction=0.5,
                                 edge_pad=0.75, use_edges=True):
    segs = []

    # Lines (near-horizontal)
    for ln in getattr(page, "lines", []):
        if abs(ln["y0"] - ln["y1"]) <= y_tol:
            y = 0.5 * (ln["y0"] + ln["y1"])
            if abs(y - y_edge) <= y_tol:
                if abs(ln["x1"] - ln["x0"]) >= min_len:
                    segs.append((min(ln["x0"], ln["x1"]), max(ln["x0"], ln["x1"])))

    # Rects (thin horizontal); include center alignment
    for rc in getattr(page, "rects", []):
        rx0, ry0, rx1, ry1 = rc["x0"], rc["y0"], rc["x1"], rc["y1"]
        rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
        cy = 0.5 * (ry0 + ry1)
        if rh <= y_tol and rw >= min_len:
            if (abs(ry0 - y_edge) <= y_tol) or (abs(ry1 - y_edge) <= y_tol) or (abs(cy - y_edge) <= y_tol):
                segs.append((min(rx0, rx1), max(rx0, rx1)))

    if use_edges:
        for ed in getattr(page, "edges", []):
            if abs(ed["y0"] - ed["y1"]) <= y_tol:
                y = 0.5 * (ed["y0"] + ed["y1"])
                if abs(y - y_edge) <= y_tol and abs(ed["x1"] - ed["x0"]) >= min_len:
                    segs.append((min(ed["x0"], ed["x1"]), max(ed["x0"], ed["x1"])))
        for cv in getattr(page, "curves", []):
            try:
                xs = [p[0] for p in cv["pts"]]; ys = [p[1] for p in cv["pts"]]
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


# --- cell border APIs you’ll use (new) ---
def has_left_border(page, cell_bbox, x_tol=2.5, y_overlap_ratio=0.5):
    x0, top, x1, bottom = cell_bbox
    return has_vertical_border_strong(
        page, x_edge=x0, y0=top, y1=bottom,
        x_tol=x_tol, need_fraction=y_overlap_ratio, edge_pad=0.75
    )

def has_upper_border(page, cell_bbox, y_tol=2.5, x_overlap_ratio=0.5):
    x0, top, x1, bottom = cell_bbox
    return has_horizontal_border_strong(
        page, y_edge=top, x0=x0, x1=x1,
        y_tol=y_tol, need_fraction=x_overlap_ratio, edge_pad=0.75
    )



def build_cell_border_grid(page, table):
    """
    Returns:
      {
        "n_rows": int,
        "n_cols": int,
        "cells": [
           {"row": r, "col": c, "bbox": (x0,top,x1,bottom), "has_left": bool, "has_top": bool},
        ]
      }
    or None if grid cannot be derived.
    """
    # Try explicit edges first (but coerce objects → floats)
    row_edges = _extract_row_edges(table)
    col_edges = _extract_col_edges(table)

    # If cols missing, derive from cell bboxes
    if row_edges and not col_edges and hasattr(table, "cells") and table.cells:
        col_edges = _derive_col_edges_from_cells(table, x_tol=2.0)

    if not (row_edges and col_edges):
        return None  # not enough info to build a grid conservatively

    n_rows = len(row_edges) - 1
    n_cols = len(col_edges) - 1
    cells = []
    for r in range(n_rows):
        top = float(row_edges[r]); bottom = float(row_edges[r+1])
        for c in range(n_cols):
            x0 = float(col_edges[c]); x1 = float(col_edges[c+1])
            bbox = (x0, top, x1, bottom)
            cells.append({
                "row": r, "col": c, "bbox": bbox,
                "has_left": has_left_border(page, bbox),
                "has_top": has_upper_border(page, bbox),
            })
    return {"n_rows": n_rows, "n_cols": n_cols, "cells": cells}


# --- NEW: normalize a table cell to a dict safely ----------------------------
def _coerce_cell(c):
    """
    Normalize a pdfplumber cell into:
    {x0, top, x1, bottom, row0, row1, col0, col1}
    Supports dict, namedtuple, tuple/list in common orders.
    """
    # dict?
    if isinstance(c, dict):
        # ensure required keys exist
        keys = {"x0","top","x1","bottom","row0","row1","col0","col1"}
        if keys.issubset(c.keys()):
            return {
                "x0": float(c["x0"]), "top": float(c["top"]), "x1": float(c["x1"]), "bottom": float(c["bottom"]),
                "row0": int(c["row0"]), "row1": int(c["row1"]), "col0": int(c["col0"]), "col1": int(c["col1"]),
            }

    # namedtuple?
    if hasattr(c, "_asdict"):
        d = c._asdict()
        return _coerce_cell(d)

    # tuple/list?
    if isinstance(c, (list, tuple)):
        vals = list(c)
        n = len(vals)

        # Common 8-length orders we’ve seen:
        # A) (x0, top, x1, bottom, row0, row1, col0, col1)
        # B) (row0, row1, col0, col1, x0, top, x1, bottom)
        if n >= 8:
            A = vals[:8]
            # try A-order
            try:
                x0, top, x1, bottom, row0, row1, col0, col1 = A
                return {
                    "x0": float(x0), "top": float(top), "x1": float(x1), "bottom": float(bottom),
                    "row0": int(row0), "row1": int(row1), "col0": int(col0), "col1": int(col1),
                }
            except Exception:
                pass
            # try B-order
            try:
                row0, row1, col0, col1, x0, top, x1, bottom = A
                return {
                    "x0": float(x0), "top": float(top), "x1": float(x1), "bottom": float(bottom),
                    "row0": int(row0), "row1": int(row1), "col0": int(col0), "col1": int(col1),
                }
            except Exception:
                pass

        # Some builds return shorter tuples (no spans). Try to infer.
        if n >= 6:
            # Heuristic: two small ints look like row/col; four floats look like bbox.
            floats = [v for v in vals if isinstance(v, (int, float)) and not float(v).is_integer() or isinstance(v, float)]
            ints   = [v for v in vals if isinstance(v, int)]
            if len(floats) >= 4 and len(ints) >= 2:
                x0, top, x1, bottom = floats[:4]
                row0 = ints[0]
                col0 = ints[1]
                # without spans, assume span of 1
                return {
                    "x0": float(x0), "top": float(top), "x1": float(x1), "bottom": float(bottom),
                    "row0": int(row0), "row1": int(row0 + 1), "col0": int(col0), "col1": int(col0 + 1),
                }

    # attribute-style object?
    for attrset in (
        ("x0","top","x1","bottom","row0","row1","col0","col1"),
        ("x0","y0","x1","y1","row0","row1","col0","col1"),
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

def _cluster_sorted(values, tol=2.0):
    """Cluster nearly-equal floats (within tol) and return sorted representatives."""
    if not values:
        return []
    vals = sorted(values)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c)/len(c) for c in clusters]

def _as_float(v):
    """Try hard to turn a row/col edge into a float y (for rows) or x (for cols)."""
    # direct numeric
    if isinstance(v, (int, float)):
        return float(v)
    # tuple/list candidates (x0, top, x1, bottom) or (y0,y1) or single value
    if isinstance(v, (list, tuple)) and v:
        # Try common y: (y0,y1) or bbox: (x0, top, x1, bottom)
        if len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            y0, y1 = float(v[0]), float(v[1])
            if abs(y1 - y0) < 1e-6:
                return y0
        if len(v) >= 4 and isinstance(v[1], (int, float)):
            return float(v[1])  # treat as "top"
        try:
            return float(v[0])
        except Exception:
            pass
    # objects with attributes (pdfplumber Ruling/Row/Column)
    for y_attrs in (("y",), ("y0","y1"), ("top","bottom")):
        try:
            if len(y_attrs) == 1 and hasattr(v, y_attrs[0]):
                return float(getattr(v, y_attrs[0]))
            if len(y_attrs) == 2 and hasattr(v, y_attrs[0]) and hasattr(v, y_attrs[1]):
                y0 = float(getattr(v, y_attrs[0])); y1 = float(getattr(v, y_attrs[1]))
                if abs(y1 - y0) < 1e-6:
                    return y0
                return (y0 + y1)/2.0
        except Exception:
            pass
    for x_attrs in (("x",), ("x0","x1"), ("left","right")):
        try:
            if len(x_attrs) == 1 and hasattr(v, x_attrs[0]):
                return float(getattr(v, x_attrs[0]))
            if len(x_attrs) == 2 and hasattr(v, x_attrs[0]) and hasattr(v, x_attrs[1]):
                x0 = float(getattr(v, x_attrs[0])); x1 = float(getattr(v, x_attrs[1]))
                if abs(x1 - x0) < 1e-6:
                    return x0
                return (x0 + x1)/2.0
        except Exception:
            pass
    return None

def _extract_row_edges(table, cluster_tol=1.5):
    """Return a sorted list of float Y edges for the grid rows."""
    if not hasattr(table, "rows") or not table.rows:
        return None
    ys = []
    for r in table.rows:
        y = _as_float(r)
        if y is not None:
            ys.append(y)
    ys = _cluster_sorted(ys, tol=cluster_tol)
    return ys if len(ys) >= 2 else None

def _extract_col_edges(table, cluster_tol=1.5):
    """Return a sorted list of float X edges for the grid cols."""
    if not hasattr(table, "cols") or not table.cols:
        return None
    xs = []
    for c in table.cols:
        x = _as_float(c)
        if x is not None:
            xs.append(x)
    xs = _cluster_sorted(xs, tol=cluster_tol)
    return xs if len(xs) >= 2 else None

def _derive_col_edges_from_cells(table, x_tol=2.0):
    """Build vertical edges from (x0, top, x1, bottom) cell tuples when cols are missing."""
    if not (hasattr(table, "cells") and table.cells):
        return None
    xs = []
    for c in table.cells:
        try:
            x0, top, x1, bottom = c[:4]
            xs.append(float(x0)); xs.append(float(x1))
        except Exception:
            continue
    edges = _cluster_sorted(xs, tol=x_tol)
    return edges if len(edges) >= 2 else None

# --- cluster helpers to rebuild vertical edges from cell boxes ---
def _cluster_sorted(values, tol=2.0):
    """
    Cluster nearly-equal float values (within tol) and return sorted representatives.
    """
    if not values:
        return []
    vals = sorted(values)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # pick the mean of each cluster (or min; mean is a bit stabler)
    return [sum(c)/len(c) for c in clusters]

def _derive_col_edges_from_cells(table, x_tol=2.0):
    """
    Build vertical grid edges from table.cells when they're (x0, top, x1, bottom) tuples.
    Returns a sorted list of x positions [x0, x1, ..., xN].
    """
    if not (hasattr(table, "cells") and table.cells):
        return None
    xs = []
    for c in table.cells:
        # your debug shows tuples of length 4
        try:
            x0, top, x1, bottom = c[:4]
            xs.append(float(x0)); xs.append(float(x1))
        except Exception:
            continue
    edges = _cluster_sorted(xs, tol=x_tol)
    # ensure at least two edges to form 1+ columns
    return edges if len(edges) >= 2 else None


# Fallback: build grid only from (x0, top, x1, bottom) tuples in table.cells
def build_cell_border_grid_from_cells_only(page, table, cluster_tol=2.0):
    if not (hasattr(table, "cells") and table.cells):
        return None

    # 1) collect all vertical & horizontal edges from cell bboxes
    xs, ys = [], []
    for c in table.cells:
        if not isinstance(c, (list, tuple)) or len(c) < 4:
            continue
        x0, top, x1, bottom = c[:4]
        xs.extend([float(x0), float(x1)])
        ys.extend([float(top), float(bottom)])

    if not xs or not ys:
        return None

    # 2) cluster into unique sorted edges
    def _cluster_sorted(vals, tol):
        vals = sorted(vals)
        clusters = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - clusters[-1][-1]) <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [sum(c)/len(c) for c in clusters]

    col_edges = _cluster_sorted(xs, cluster_tol)
    row_edges = _cluster_sorted(ys, cluster_tol)
    if len(col_edges) < 2 or len(row_edges) < 2:
        return None

    # 3) enumerate slots & test borders
    n_rows = len(row_edges) - 1
    n_cols = len(col_edges) - 1
    cells = []
    for r in range(n_rows):
        top = row_edges[r]; bottom = row_edges[r+1]
        for c in range(n_cols):
            x0 = col_edges[c]; x1 = col_edges[c+1]
            bbox = (x0, top, x1, bottom)
            cells.append({
                "row": r, "col": c, "bbox": bbox,
                "has_left": has_left_border(page, bbox),
                "has_top":  has_upper_border(page, bbox),
            })
    return {"n_rows": n_rows, "n_cols": n_cols, "cells": cells}


def bbox_intersects(a, b):
    # a and b = [x0, top, x1, bottom]
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    horiz = not (ax1 <= bx0 or bx1 <= ax0)
    vert = not (ay1 <= by0 or by1 <= ay0)
    return horiz and vert


def get_page_elements(pdf_path):
    pages_output = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            # 1) find table bboxes (pdfplumber Table objects via find_tables)
            table_bboxes = []
            tables = []
            try:
                table_objs = page.find_tables()
            except Exception:
                table_objs = []
            for t_idx, t in enumerate(table_objs):
                # pdfplumber Table has .bbox and .extract() method
                tbbox = t.bbox  # (x0, top, x1, bottom)
                table_bboxes.append({"bbox": tbbox, "table_index": t_idx})
                # get rows as list of lists
                try:
                    rows = t.extract()
                except Exception:
                    # fallback to page.extract_tables
                    rows = None
                    # NEW: compute per-cell borders (left/top) if grid available; else None
                grid = build_cell_border_grid(page, t)
                if grid is None:
                    grid = build_cell_border_grid_from_cells_only(page, t)
                # print("ADDING GRID")
                tables.append({
                    "table_index": t_idx,
                    "bbox": tbbox,
                    "rows": rows,
                    "grid": grid,            # <-- only addition
                })

            # 2) extract words with bboxes and filter out words that fall into any table bbox
            words = page.extract_words()  # each item has x0, x1, top (y0), bottom (y1), text
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
            # 3) rebuild narrative text using word order grouping by line (simple)
            # group by 'top' with small tolerance to form lines
            lines = {}
            tol = 3  # pixels tolerance
            for w in narrative_words:
                top = round(float(w["top"]) / tol) * tol
                lines.setdefault(top, []).append((float(w["x0"]), w["text"]))
            # sort lines by top, then words by x0
            sorted_lines = [
                " ".join([t[1] for t in sorted(line, key=lambda x: x[0])])
                for top, line in sorted(lines.items())
            ]

            narrative_text = "\n".join(sorted_lines).strip()
            # 4) for the page output, include tables (as DataFrames if available)
            table_records = []
            for t in tables:
                if t["rows"]:
                    table_dct = t["rows"]
                else:
                    #                     # fallback: try page.extract_tables and pick the matching by bbox proximity
                    table_dct = None
                table_records.append(
                    {
                        "table_index": t["table_index"],
                        "bbox": t["bbox"],
                        "table_dct": table_dct,
                        "grid": t["grid"]
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


def get_all_tables(pdf_pages: Iterable[dict]) -> Tuple[List[Any], List[str]]:
    """
    Return (tables, tables_flatted)
    - tables: list of each table['table_dct'] (or None if missing)
    - tables_flatted: string for each table where cells in a row are joined by spaces
      and rows are joined by newlines.
    Behavior:
      - treats None cells as empty string (but preserves 0 and False as values).
      - tolerant if 'tables' or 'table_dct' keys are missing.
    """
    tables: List[Any] = []
    tables_flatted: List[str] = []

    append_table = tables.append
    append_flat = tables_flatted.append

    for page in pdf_pages:
        # safer access to page['tables']
        for table in page.get("tables", []):
            # prefer dict-style access but be robust to different shapes
            td = (
                table.get("table_dct")
                if isinstance(table, dict)
                else getattr(table, "table_dct", None)
            )
            append_table(td)

            # guard: if td is not an iterable of rows, use empty list
            if td is None:
                append_flat("")  # keep alignment: one flat string per table entry
                continue

            # build flattened text efficiently using join() (linear time)
            # convert None -> '' but keep falsy numerics (0) as '0'
            flat = "\n".join(
                " ".join(("" if cell is None else str(cell)) for cell in row) for row in td
            )
            flat = re.sub(r"\s+", " ", flat)

            append_flat(flat)

    return tables, tables_flatted


def clean_table_left_focus(table):
    """
    clean None with previous cell (based on result, None value is the previous. Ex: CP_Perjalanan Dinas)
    """
    for row in table:
        for i, cell in enumerate(row):
            if cell is None and i > 0:
                row[i] = row[i - 1]
    return table

def _lookup_borders(table, row_idx, col_idx):
    """
    Returns (has_left, has_top). If grid or cell not found, returns (None, None).
    """
    grid = table.get("grid")
    if not grid or not grid.get("cells"):
        return (None, None)
    for c in grid["cells"]:
        if c.get("row") == row_idx and c.get("col") == col_idx:
            return (c.get("has_left"), c.get("has_top"))
    return (None, None)


def clean_table(table):
    """
    Clean None by backfilling from the previous cell (left) or from the upper cell,
    guided by detected borders: if no LEFT border → copy from left;
    elif no TOP border → copy from above.
    """
    # deep copy so the original table_dct is untouched
    table_dct = copy.deepcopy(table.get("table_dct", []))

    for i, row in enumerate(table_dct):
        for j, cell in enumerate(row):
            if cell is None:
                has_left, has_upper = _lookup_borders(table, i, j)

                # Prefer copying from ABOVE when allowed
                if has_upper is False and i > 0:
                    up_val = table_dct[i - 1][j]
                    if up_val is not None:
                        table_dct[i][j] = up_val
                        # print(f"\tcell {i},{j} <- up {i-1},{j}")
                        continue

                # Otherwise copy from LEFT when allowed
                if has_left is False and j > 0:
                    left_val = table_dct[i][j - 1]
                    if left_val is not None:
                        table_dct[i][j] = left_val
                        # print(f"\tcell {i},{j} <- left {i},{j-1}")
                        continue  # done
                
                # print('<- has both')
        # print('\n')
                # If borders unknown (None) or neighbors are None, leave as None
                # (you could add more heuristics here if desired)

    return table_dct


def change_to_records(table):
    """change table format from a list of list to records"""
    headers = table[0]
    data_rows = table[1:]
    # Use a list comprehension to build the list of dictionaries
    records = [dict(zip(headers, row)) for row in data_rows]
    return records


def fix_nested_table(table, tables, tables_flatted):
    """fix for a table inside a table
    if there is a table inside a table,
    returning a fixed table that the value is a table, not messy text"""
    for index_row, row in enumerate(table):
        for index_col, cell in enumerate(row):
            if cell and isinstance(cell, str):
                if len(cell) > 20:  # only check for long text
                    one, score = process.extractOne(
                        cell, tables_flatted, scorer=fuzz.token_sort_ratio
                    )
                    if score > 80:
                        idx = tables_flatted.index(one)
                        # update the table inside a table with real table
                        cleaned_table = clean_table_left_focus(tables[idx])
                        cleaned_table = change_to_records(cleaned_table)
                        table[index_row][index_col] = cleaned_table
    return table


def check_double_header(table_data):
    """
    Analyzes a table's structure and returns its type.
    """
    if len(table_data) < 2:
        return False  # or "too_small"

    header_row_1 = table_data[0]
    header_row_2 = table_data[1]

    # Heuristic for the complex table:
    # Does the second row have non-None values under None values in the first?
    # This is a strong sign of a merged, hierarchical header.

    for i, item in enumerate(header_row_1):
        if item is None and header_row_2[i] is not None:
            return True

    # Heuristic for a simple table:
    # The first row is full and the second row looks like data.
    if all(isinstance(h, str) for h in header_row_1):
        return False

    return False  # Default fallback


def check_triple_header(table_data):
    """
    Analyzes a table's structure and returns its type.
    """
    try:
        if len(table_data) < 2:
            return False  # or "too_small"

        header_row_2 = table_data[1]
        header_row_3 = table_data[2]

        # Heuristic for the complex table:
        # Does the third row have non-None values under None values in the second?
        # This is a strong sign of a merged, hierarchical header.

        for i, item in enumerate(header_row_2):
            #         print('item:', item, "below:", header_row_3[i])
            if item is None and header_row_3[i] is not None and i > 0:
                #             print('   ^True')
                return True

        # Heuristic for a simple table:
        # The first row is full and the second row looks like data.
        if all(isinstance(h, str) for h in header_row_2):
            return False
    except Exception as e:
        logger.debug(f"Error in check_triple_header: {e}")

    return False  # Default fallback


def process_double_header_table(table_data):
    """
    Processes a complex, multi-header table into a list of dictionaries (records),
    where each dictionary represents a row.
    """
    if len(table_data) < 3:
        # Not enough rows for headers and data
        return []

    # --- Step 1: Reconstruct and Flatten the Headers (No changes here) ---
    header_row_1 = table_data[0]
    header_row_2 = table_data[1]

    # Forward-fill the top-level headers
    filled_header_1 = []
    last_header = ""
    for header in header_row_1:
        if header is not None:
            last_header = header
        filled_header_1.append(last_header)

    # Combine headers into a final, flat list
    flattened_headers = []
    for i, sub_header in enumerate(header_row_2):
        main_header = filled_header_1[i]
        if sub_header:
            clean_sub_header = sub_header.replace("\n", " ")
            flattened_headers.append(f"{main_header} - {clean_sub_header}")
        else:
            flattened_headers.append(main_header)

    # --- Step 2: Build a List of Records (This part is new) ---
    records = []
    data_rows = table_data[2:]  # The rest of the list is data

    for row in data_rows:
        # Skip any row that is empty or doesn't have a primary identifier
        if not row or row[0] is None:
            continue

        # Create a dictionary for the current row by zipping headers and row values
        # The dict() constructor combined with zip() is a concise way to do this.
        record = dict(zip(flattened_headers, row))
        records.append(record)

    return records


def process_triple_header_table(table_data):
    """
    Processes a complex, TRIPLE-header table into a list of dictionaries (records).
    """
    if len(table_data) < 4:  # 3 header rows + at least 1 data row
        return []

    # --- Step 1: Reconstruct and Flatten the Three-Level Headers ---
    header_row_1 = table_data[0]
    header_row_2 = table_data[1]
    header_row_3 = table_data[2]

    # Level 1: Forward-fill the top-most header row
    filled_header_1 = []
    last_header = ""
    for header in header_row_1:
        if header is not None:
            last_header = header.replace("\n", " ")
        filled_header_1.append(last_header)

    # Level 2: Forward-fill the middle header row
    filled_header_2 = []
    last_header = ""
    for i, header in enumerate(header_row_2):
        # Reset fill if the top-level header changes, making it more robust
        if filled_header_1[i] != filled_header_1[i - 1] and i > 0:
            last_header = ""
        if header is not None:
            last_header = header.replace("\n", " ")
        filled_header_2.append(last_header)

    # Level 3: Combine all three filled headers into a final, flat list
    flattened_headers = []
    for i in range(len(header_row_1)):
        # Collect parts from all three levels, cleaning newlines
        h1 = filled_header_1[i]
        h2 = filled_header_2[i]
        h3 = header_row_3[i].replace("\n", " ") if header_row_3[i] else ""

        # Build the final header by joining the parts that are not empty
        parts = [part for part in [h1, h2, h3] if part]
        unique_parts = list(dict.fromkeys(parts))  # Remove duplicates like 'Gol.' - 'Gol.'
        final_header = " - ".join(unique_parts)
        flattened_headers.append(final_header)

    # --- Step 2: Build a List of Records (Same logic as before) ---
    records = []
    data_rows = table_data[3:]  # The rest is data

    for row in data_rows:
        if not row or row[0] is None:
            continue

        record = dict(zip(flattened_headers, row))
        records.append(record)

    return records


def check_and_fix_nested_tables(pdf_pages):
    """check and fix nested table"""
    tables, tables_flatted = get_all_tables(pdf_pages)
    for page_idx, page in enumerate(pdf_pages):
        if page["tables"]:
            for i, dct in enumerate(page["tables"]):
                if dct["table_dct"]:
                    table = dct["table_dct"]
                    try:
                        fixed_table = fix_nested_table(table, tables, tables_flatted)
                    except Exception as e:
                        logger.debug(f"Error in fix_nested_table for page {page_idx+1}, table {i}: {e}")
                        fixed_table = table  # fallback to original if error
                pdf_pages[page_idx]["tables"][i]["table_dct"] = fixed_table
    return pdf_pages


def change_table_format(pdf_pages):
    """change table format from a list of list to records and handle double header"""
    for page_idx, page in enumerate(pdf_pages):
        if page["tables"]:
            for i, dct in enumerate(page["tables"]):
                if dct["table_dct"]:
                    is_double_header = check_double_header(dct["table_dct"])
                    is_triple_header = (
                        check_triple_header(dct["table_dct"]) if is_double_header else False
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
                            # Use a list comprehension to build the list of dictionaries
                            records = [dict(zip(headers, row)) for row in data_rows]
                    except Exception as e:
                        logger.debug(f"Error in change_table_format for page {page_idx+1}, table {i}: {e}")
                        try:
                            headers = table[0]
                            data_rows = table[1:]
                            records = [dict(zip(headers, row)) for row in data_rows]
                        except Exception as e2:
                            logger.debug(f"  Fallback also failed: {e2}")
                            records = [{'table':table}]  # fallback to original table if error
                pdf_pages[page_idx]["tables"][i]["table_dct"] = records
    return pdf_pages


def extract_pdf_content(file):
    """
    Extract all texts from a PDF file, including narrative text and tables.

    Steps:
    1. Parse each page to extract narrative text and tables.
    2. Fix nested tables (tables embedded as text within other tables).
    3. Convert table format from list of lists to list of records (dicts).

    Args:
        file (str): Path to the PDF file.

    Returns:
        List[dict]: List of page dictionaries containing narrative text and tables.
    """
    pdf_pages = get_page_elements(file)  # Step 1: Extract page elements
    pdf_pages = check_and_fix_nested_tables(pdf_pages)  # Step 2: Fix nested tables
    pdf_pages = change_table_format(pdf_pages)  # Step 3: Format tables as records

    return pdf_pages
