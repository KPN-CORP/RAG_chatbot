import sys
import types

# Provide a lightweight pdfplumber stub so the module can import without the
# optional dependency installed in the test environment. Each test replaces
# ``pdfplumber.open`` with a fake implementation tailored to the scenario.
if "pdfplumber" not in sys.modules:
    stub = types.ModuleType("pdfplumber")
    stub.open = lambda *_args, **_kwargs: None
    sys.modules["pdfplumber"] = stub


if "boto3" not in sys.modules:
    boto3_stub = types.ModuleType("boto3")

    class _DummySession:
        def client(self, *_args, **_kwargs):
            return types.SimpleNamespace()

    boto3_stub.Session = lambda *_args, **_kwargs: _DummySession()
    boto3_stub.client = lambda *_args, **_kwargs: types.SimpleNamespace()
    sys.modules["boto3"] = boto3_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    sys.modules["dotenv"] = dotenv_stub

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


def test_extract_pdf_content_uses_ocr_fallback_for_scanned_pages(monkeypatch):
    page = _DummyPage(width=1000, height=1000, tables=[])
    dummy_pdf = _DummyPDF([page])

    monkeypatch.setattr(
        pdf_with_tables,
        "pdfplumber",
        types.SimpleNamespace(open=lambda _path: dummy_pdf),
    )
    monkeypatch.setattr(pdf_with_tables, "check_and_fix_nested_tables", lambda pages: pages)
    monkeypatch.setattr(
        pdf_with_tables,
        "change_table_format",
        lambda pages, **_kwargs: pages,
    )
    monkeypatch.setattr(pdf_with_tables.os, "getenv", lambda key, default=None: "bucket" if key == "S3_BUCKET_NAME" else default)
    monkeypatch.setattr(
        pdf_with_tables,
        "_extract_textract_lines",
        lambda **_kwargs: [
            {
                "BlockType": "LINE",
                "Page": 1,
                "Text": "Scanned line 1",
                "Geometry": {"BoundingBox": {"Top": 0.1, "Left": 0.2, "Height": 0.05}},
            },
            {
                "BlockType": "LINE",
                "Page": 1,
                "Text": "Scanned line 2",
                "Geometry": {"BoundingBox": {"Top": 0.2, "Left": 0.2, "Height": 0.05}},
            },
        ],
    )

    pages = pdf_with_tables.extract_pdf_content("placeholder.pdf")

    assert pages[0]["narrative_text"] == "Scanned line 1\nScanned line 2"
    assert [line["text"] for line in pages[0]["narrative_lines"]] == [
        "Scanned line 1",
        "Scanned line 2",
    ]
    assert pages[0]["narrative_lines"][0]["hanging_x"] == 200.0


def test_extract_pdf_content_keeps_pdfplumber_text_without_ocr(monkeypatch):
    page = _DummyPage(width=1000, height=1000, tables=[])
    page.extract_words = lambda **_kwargs: [
        {"text": "Native", "x0": 0, "x1": 10, "top": 10, "bottom": 20},
        {"text": "text", "x0": 12, "x1": 20, "top": 10, "bottom": 20},
    ]
    dummy_pdf = _DummyPDF([page])

    monkeypatch.setattr(
        pdf_with_tables,
        "pdfplumber",
        types.SimpleNamespace(open=lambda _path: dummy_pdf),
    )
    monkeypatch.setattr(pdf_with_tables, "check_and_fix_nested_tables", lambda pages: pages)
    monkeypatch.setattr(
        pdf_with_tables,
        "change_table_format",
        lambda pages, **_kwargs: pages,
    )

    def _unexpected_ocr(**_kwargs):
        raise AssertionError("OCR fallback should not run when pdfplumber returns text")

    monkeypatch.setattr(pdf_with_tables, "_extract_textract_lines", _unexpected_ocr)

    pages = pdf_with_tables.extract_pdf_content("placeholder.pdf")

    assert pages[0]["narrative_text"] == "Native text"
