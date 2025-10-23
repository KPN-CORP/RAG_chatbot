import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.chunking_utils import _clean_line, _is_header


def test_is_header_rejects_standalone_time_token():
    assert not _is_header("14.00.")
    assert not _is_header("08:30")


def test_is_header_allows_legitimate_headers():
    header = "PERJANJIAN KERJA BERSAMA"
    cleaned = _clean_line(header)
    assert _is_header(cleaned)
