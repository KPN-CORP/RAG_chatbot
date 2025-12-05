import sys
import types

# Provide a lightweight pdfplumber stub so the module can import without the
# optional dependency installed in the test environment. Each test replaces
# ``pdfplumber.open`` with a fake implementation tailored to the scenario.
if "pdfplumber" not in sys.modules:
    stub = types.ModuleType("pdfplumber")
    stub.open = lambda *_args, **_kwargs: None
    sys.modules["pdfplumber"] = stub

if "logging_config" not in sys.modules:
    logging_stub = types.ModuleType("logging_config")

    class _DummyLogger:
        def debug(self, *_args, **_kwargs):
            return None

        info = warning = debug

    logging_stub.get_logger = lambda *_args, **_kwargs: _DummyLogger()
    sys.modules["logging_config"] = logging_stub

if "modules" not in sys.modules:
    sys.modules["modules"] = types.ModuleType("modules")

if "modules.table_utils" not in sys.modules:
    table_utils = types.ModuleType("modules.table_utils")
    table_utils.bbox_intersects = lambda *_args, **_kwargs: False
    table_utils.build_cell_border_grid = lambda *_args, **_kwargs: None
    table_utils.build_cell_border_grid_from_cells_only = lambda *_args, **_kwargs: None
    table_utils.check_and_fix_nested_tables = lambda pages: pages
    table_utils.change_table_format = lambda *args, **_kwargs: args[0]
    sys.modules["modules.table_utils"] = table_utils
    sys.modules["modules"].table_utils = table_utils

if "modules.chunking_utils" not in sys.modules:
    import re

    chunking_utils = types.ModuleType("modules.chunking_utils")
    chunking_utils.TRAIL_NUM_RE = re.compile(r"\d+")
    chunking_utils.should_strip_trailing_number = lambda *_args, **_kwargs: False
    sys.modules["modules.chunking_utils"] = chunking_utils
    sys.modules["modules"].chunking_utils = chunking_utils

if "modules.narrative_utils" not in sys.modules:
    narrative_utils = types.ModuleType("modules.narrative_utils")
    narrative_utils.BULLET_TOKENS = set()
    narrative_utils._strip_bullet = lambda text: (text, False)
    narrative_utils._same_line = lambda *_args, **_kwargs: True
    narrative_utils._is_probable_header_footer = lambda *_args, **_kwargs: False
    sys.modules["modules.narrative_utils"] = narrative_utils
    sys.modules["modules"].narrative_utils = narrative_utils

from application.ingest import pdf_with_tables


class _DummyTable:
    def __init__(self, bbox, rows):
        self.bbox = bbox
        self._rows = rows

    def extract(self):
        return self._rows


class _DummyPage:
    def __init__(self, width, height, tables):
        self.width = width
        self.height = height
        self._tables = tables

    def find_tables(self):
        return self._tables

    def extract_words(self, **_kwargs):
        return []


class _DummyPDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_minimal_pdf(monkeypatch, *, prefer_table_detection: bool):
    # Long text in a framed box that would normally be treated as narrative when
    # table preference is disabled.
    long_cell = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 3
    bbox = (0, 0, 700, 550)  # spans a large portion of the page
    tables = [_DummyTable(bbox=bbox, rows=[[long_cell], [long_cell]])]
    page = _DummyPage(width=1000, height=1000, tables=tables)
    dummy_pdf = _DummyPDF([page])

    monkeypatch.setattr(
        pdf_with_tables,
        "pdfplumber",
        types.SimpleNamespace(open=lambda _path: dummy_pdf),
    )
    monkeypatch.setattr(pdf_with_tables, "build_cell_border_grid", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        pdf_with_tables, "build_cell_border_grid_from_cells_only", lambda *_args, **_kw: None
    )

    return pdf_with_tables.get_page_elements(
        "placeholder.pdf", prefer_table_detection=prefer_table_detection
    )


def test_prefers_tables_when_gemini_enabled(monkeypatch):
    pages = _patch_minimal_pdf(monkeypatch, prefer_table_detection=True)
    assert pages[0]["tables"], "Tables should be preserved for Gemini extraction"


def test_reclassifies_paragraph_tables_when_not_preferring_tables(monkeypatch):
    pages = _patch_minimal_pdf(monkeypatch, prefer_table_detection=False)
    assert pages[0]["tables"] == [], "Paragraph-like boxes become narrative when Gemini is off"
