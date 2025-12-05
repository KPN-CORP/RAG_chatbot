import json
import sys
from pathlib import Path

import pytest

APPLICATION_DIR = Path(__file__).resolve().parents[1]
if str(APPLICATION_DIR) not in sys.path:
    sys.path.append(str(APPLICATION_DIR))

from application.modules.utils import extract_json_string


def test_extract_json_string_defaults_missing_sources_to_empty_list():
    text = '{"answer": "ok"}'

    result = extract_json_string(text)

    assert result is not None
    payload = json.loads(result)
    assert payload["answer"] == "ok"
    assert payload["sources"] == []


def test_extract_json_string_deduplicates_source_entries():
    text = """
    ```json
    {"answer": "hi", "sources": [{"id": 1}, {"id": 1}, {"id": 2}]}
    ```
    """

    result = extract_json_string(text)

    assert result is not None
    payload = json.loads(result)
    assert payload["sources"] == [{"id": 1}, {"id": 2}]
