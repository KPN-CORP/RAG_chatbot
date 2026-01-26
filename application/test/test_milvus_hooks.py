import sys
from pathlib import Path

APPLICATION_DIR = Path(__file__).resolve().parents[1]
if str(APPLICATION_DIR) not in sys.path:
    sys.path.append(str(APPLICATION_DIR))

from modules.milvus_hooks import SOURCE_METADATA_KEY, _build_delete_expr


def test_build_delete_expr_includes_basename_and_full_path():
    file_path = "/tmp/Knowledge Base/Policy A.pdf"

    expr, candidates = _build_delete_expr(file_path)

    assert "Policy A.pdf" in candidates
    assert "/tmp/Knowledge Base/Policy A.pdf" in candidates
    for key in [SOURCE_METADATA_KEY, "filename", "path", "file_name"]:
        assert f'metadata["{key}"] == "Policy A.pdf"' in expr
        assert f'metadata["{key}"] == "/tmp/Knowledge Base/Policy A.pdf"' in expr


def test_build_delete_expr_escapes_quotes_in_filename():
    file_path = '/tmp/Policy "A".pdf'

    expr, candidates = _build_delete_expr(file_path)

    assert 'Policy "A".pdf' in candidates
    assert '\\"A\\"' in expr
