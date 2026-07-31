"""Regression tests for merged-cell handling in table extraction/chunking.

Merged rows/columns arrive from pdfplumber as ``None`` continuation cells. If
they are not propagated, the downstream cell-chunker drops them and the spanning
category/header is lost from every row it governs, producing inaccurate chunks
(e.g. a duration chunk that no longer mentions its leave type). These tests lock
in the fill behaviour of ``clean_table`` and the end-to-end guarantee that a
merged category survives into the cell chunks.
"""

import sys
import types
from pathlib import Path

APPLICATION_DIR = Path(__file__).resolve().parents[1]
if str(APPLICATION_DIR) not in sys.path:
    sys.path.append(str(APPLICATION_DIR))

# ``modules.table_utils`` imports fuzzywuzzy at module load. Stub it so the test
# runs with only pytest installed (matching the rest of the suite's approach).
if "fuzzywuzzy" not in sys.modules:
    fuzzywuzzy_stub = types.ModuleType("fuzzywuzzy")
    fuzz_mod = types.ModuleType("fuzzywuzzy.fuzz")
    process_mod = types.ModuleType("fuzzywuzzy.process")
    fuzz_mod.token_sort_ratio = lambda *_a, **_k: 0
    process_mod.extractOne = lambda *_a, **_k: ("", 0)
    fuzzywuzzy_stub.fuzz = fuzz_mod
    fuzzywuzzy_stub.process = process_mod
    sys.modules["fuzzywuzzy"] = fuzzywuzzy_stub
    sys.modules["fuzzywuzzy.fuzz"] = fuzz_mod
    sys.modules["fuzzywuzzy.process"] = process_mod

# ``test_pdf_with_tables`` installs stub ``modules``/``ingest`` packages into
# sys.modules (and never restores them), so a plain import here would pick up
# those stubs instead of the real code. Temporarily drop the relevant entries,
# import the real modules to capture the functions under test, then restore the
# previous entries verbatim so this test leaves global import state untouched
# for the rest of the suite.
_REAL_IMPORT_NAMES = [
    "config_loader",
    "modules",
    "modules.table_utils",
    "modules.chunking_utils",
    "ingest",
    "ingest.chunking",
]
_saved_modules = {name: sys.modules.pop(name, None) for name in _REAL_IMPORT_NAMES}
try:
    from modules.table_utils import clean_table  # noqa: E402
    from ingest.chunking import table_cells_to_chunks  # noqa: E402
finally:
    for _name, _mod in _saved_modules.items():
        if _mod is not None:
            sys.modules[_name] = _mod
        else:
            sys.modules.pop(_name, None)


def _grid_with_top(has_top: bool, n_rows: int = 3, n_cols: int = 3) -> dict:
    """Build a border grid where every cell reports the given top-border state."""

    return {
        "cells": [
            {"row": r, "col": c, "has_left": True, "has_top": has_top}
            for r in range(n_rows)
            for c in range(n_cols)
        ]
    }


def _leave_table(grid) -> dict:
    """A leave-policy table whose first column is merged down over two rows."""

    return {
        "table_dct": [
            ["Jenis Cuti", "Syarat", "Durasi"],
            ["Cuti Tahunan", "Masa kerja >= 1 tahun", "12 hari"],
            [None, "Masa kerja >= 5 tahun", "15 hari"],
        ],
        "grid": grid,
    }


def test_clean_table_forward_fills_vertical_merge_when_no_grid():
    """Without a border grid, a merged cell is carried down from the row above."""

    filled = clean_table(_leave_table(grid=None))

    assert filled[2][0] == "Cuti Tahunan"  # merged category propagated
    assert filled[2][1] == "Masa kerja >= 5 tahun"  # real values untouched


def test_clean_table_fills_when_top_border_confirmed_absent():
    """A confirmed-absent top border means a genuine vertical merge -> fill."""

    filled = clean_table(_leave_table(grid=_grid_with_top(has_top=False)))

    assert filled[2][0] == "Cuti Tahunan"


def test_clean_table_keeps_blank_when_top_border_confirmed_present():
    """A confirmed top border means the cell is genuinely empty -> do NOT fill."""

    filled = clean_table(_leave_table(grid=_grid_with_top(has_top=True)))

    assert filled[2][0] is None  # no over-fill across a real border


def test_merged_category_survives_into_cell_chunks():
    """End-to-end: the second row's chunks still carry the merged leave type."""

    filled = clean_table(_leave_table(grid=None))
    headers = filled[0]
    records = [dict(zip(headers, row)) for row in filled[1:]]

    chunks = table_cells_to_chunks(
        table_dict_rows=records,
        doc_id="D",
        source="hr.pdf",
        page=1,
        table_index=0,
        table_title="Kebijakan Cuti",
    )

    texts = [c["chunk_text"] for c in chunks]
    # The "15 hari" cell belongs to the merged "Cuti Tahunan" category; its chunk
    # must mention the leave type so retrieval can connect the two.
    duration_chunks = [t for t in texts if "15 hari" in t]
    assert duration_chunks, "expected a chunk for the second-row duration"
    assert all("Cuti Tahunan" in t for t in duration_chunks)
