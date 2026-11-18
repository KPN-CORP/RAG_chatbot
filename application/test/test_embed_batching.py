import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the ingest module with its heavy dependencies stubbed, then restore
# sys.modules so other test files can install their own stubs/real modules.
_STUBS = {
    "pymilvus": ("Collection", "CollectionSchema", "DataType", "FieldSchema", "connections", "utility"),
    "models.embbedings": ("get_gemini_embeddings", "get_huggingface_embeddings"),
    "ingest.chunking": ("build_chunks_from_pages",),
    "modules.utils": ("resolve_embedding_api_key",),
    "modules.chunking_utils": ("_checksum",),
}
_saved = dict(sys.modules)
for _name, _attrs in _STUBS.items():
    _mod = types.ModuleType(_name)
    for _attr in _attrs:
        setattr(_mod, _attr, object())
    sys.modules[_name] = _mod
sys.modules.pop("ingest.langchain_milvus_ingest", None)
import ingest.langchain_milvus_ingest as ing  # noqa: E402

sys.modules.clear()
sys.modules.update(_saved)


class FakeEmbeddings:
    def __init__(self, fail_times=0, fail_if_larger_than=None, error="504 Deadline Exceeded"):
        self.calls = []
        self.fail_times = fail_times
        self.fail_if_larger_than = fail_if_larger_than
        self.error = error

    def embed_documents(self, batch):
        self.calls.append(list(batch))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(f"Error embedding content: {self.error}")
        if self.fail_if_larger_than and len(batch) > self.fail_if_larger_than:
            raise RuntimeError(f"Error embedding content: {self.error}")
        return [[float(len(t))] for t in batch]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ing.time, "sleep", lambda s: None)


def test_batches_are_capped_by_total_characters():
    texts = ["x" * 3000] * 20  # 60k chars
    emb = FakeEmbeddings()
    vectors = ing.embed_batch_texts(texts, emb, batch_size=32)
    assert len(vectors) == 20
    assert all(sum(len(t) for t in call) <= ing.EMBED_MAX_BATCH_CHARS for call in emb.calls)
    assert len(emb.calls) > 1


def test_transient_error_is_retried():
    emb = FakeEmbeddings(fail_times=2)
    assert ing.embed_batch_texts(["a", "bb"], emb, batch_size=32) == [[1.0], [2.0]]
    assert len(emb.calls) == 3


def test_persistent_timeout_splits_batch_and_keeps_order():
    emb = FakeEmbeddings(fail_if_larger_than=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    assert ing.embed_batch_texts(texts, emb, batch_size=32) == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_non_transient_error_is_raised_immediately():
    emb = FakeEmbeddings(fail_times=5, error="400 API key not valid")
    with pytest.raises(RuntimeError, match="API key"):
        ing.embed_batch_texts(["a"], emb, batch_size=32)
    assert len(emb.calls) == 1
