import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

import application.ingest.chunking as chunking


def test_gemini_pages_feed_section_carry(monkeypatch):
    parse_calls: List[Dict[str, Any]] = []
    atomic_pages: List[int] = []

    def fake_parse_policy_to_nodes(text: str, section_hint=None, lines_with_meta=None):
        parse_calls.append({"text": text.strip(), "section_hint": section_hint})
        nodes = []
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned.endswith("HEADER"):
                nodes.append({"kind": "header", "line": cleaned, "path": [cleaned]})
        return nodes

    def fake_group_atomic_rules_semantic(nodes, embeddings=None, sim_threshold=None, use_percentile=None, percentile=None):
        return [{"text": n.get("line", ""), "kind": n.get("kind")} for n in nodes]

    def fake_semantic_refine_rules(rules, embeddings=None, enable=True, max_chars=None, breakpoint_threshold_type=None, breakpoint_threshold_amount=None):
        return rules

    def fake_atomic_rules_to_chunks(rules, doc_id: str, source: str, page: int):
        atomic_pages.append(page)
        return [
            {"chunk_id": f"{doc_id}::atomic::{page:04d}::{idx:04d}", "chunk_text": r.get("text", ""), "page": page}
            for idx, r in enumerate(rules, 1)
        ]

    monkeypatch.setattr(chunking, "parse_policy_to_nodes", fake_parse_policy_to_nodes)
    monkeypatch.setattr(chunking, "group_atomic_rules_semantic", fake_group_atomic_rules_semantic)
    monkeypatch.setattr(chunking, "semantic_refine_rules", fake_semantic_refine_rules)
    monkeypatch.setattr(chunking, "atomic_rules_to_chunks", fake_atomic_rules_to_chunks)

    pages = [
        {"page": 1, "narrative_text": "OLD HEADER\nFirst rule."},
        {
            "page": 2,
            # Gemini produced the only content on this page.
            "gemini_chunks": [{"chunk_text": "NEW HEADER\nGemini table output", "chunk_type": "narrative"}],
        },
        {"page": 3, "narrative_text": "Follow-on rule"},
    ]

    built = chunking.build_chunks_from_pages(
        pages,
        doc_id="doc-1",
        source="pdf",
        embeddings_for_semantic=None,
    )

    # Gemini chunks should still be included.
    gemini_ids = [c["chunk_id"] for c in built if "gemini" in c.get("chunk_id", "")]
    assert gemini_ids == ["doc-1::gemini::0002::0001"]

    # The semantic/atomic pipeline should process pages 1 and 3 and inspect page 2
    # so section carry-over is aware of Gemini-only pages.
    assert atomic_pages == [1, 3]
    assert [c["section_hint"] for c in parse_calls] == ["Old Header", "New Header", "NEW HEADER"]


def test_ocr_lines_drive_bullet_hierarchy_from_geometry():
    text = "SECTION HEADER\n• Parent bullet.\n• Child bullet."
    lines_with_meta = [
        {"text": "SECTION HEADER", "hanging_x": 100.0, "min_x": 100.0, "top": 50.0, "bottom": 80.0, "in_band": False, "trail_num_token": None},
        {"text": "• Parent bullet.", "hanging_x": 100.0, "min_x": 100.0, "top": 150.0, "bottom": 180.0, "in_band": False, "trail_num_token": None},
        {"text": "• Child bullet.", "hanging_x": 320.0, "min_x": 320.0, "top": 220.0, "bottom": 250.0, "in_band": False, "trail_num_token": None},
    ]

    nodes = chunking.parse_policy_to_nodes(text, lines_with_meta=lines_with_meta)

    bullet_nodes = [n for n in nodes if n["kind"] == "bullet"]
    assert [n["level"] for n in bullet_nodes] == [1, 2]
    assert [n["line"] for n in bullet_nodes] == ["Parent bullet.", "Child bullet."]


def test_build_chunks_from_pages_accepts_ocr_narrative_lines(monkeypatch):
    def fake_group_atomic_rules_semantic(nodes, embeddings=None, sim_threshold=None, use_percentile=None, percentile=None):
        return [{"text": n.get("line", "")} for n in nodes if n.get("kind") != "header"]

    def fake_semantic_refine_rules(rules, embeddings=None, enable=True, max_chars=None, breakpoint_threshold_type=None, breakpoint_threshold_amount=None):
        return rules

    def fake_atomic_rules_to_chunks(rules, doc_id: str, source: str, page: int):
        return [
            {"chunk_id": f"{doc_id}::atomic::{page:04d}::{idx:04d}", "chunk_text": r["text"], "page": page}
            for idx, r in enumerate(rules, 1)
        ]

    monkeypatch.setattr(chunking, "group_atomic_rules_semantic", fake_group_atomic_rules_semantic)
    monkeypatch.setattr(chunking, "semantic_refine_rules", fake_semantic_refine_rules)
    monkeypatch.setattr(chunking, "atomic_rules_to_chunks", fake_atomic_rules_to_chunks)

    pages = [
        {
            "page": 1,
            "narrative_text": "SECTION HEADER\n• Parent bullet.\n• Child bullet.",
            "narrative_lines": [
                {"text": "SECTION HEADER", "hanging_x": 100.0, "min_x": 100.0, "top": 50.0, "bottom": 80.0, "in_band": False, "trail_num_token": None},
                {"text": "• Parent bullet.", "hanging_x": 100.0, "min_x": 100.0, "top": 150.0, "bottom": 180.0, "in_band": False, "trail_num_token": None},
                {"text": "• Child bullet.", "hanging_x": 320.0, "min_x": 320.0, "top": 220.0, "bottom": 250.0, "in_band": False, "trail_num_token": None},
            ],
            "tables": [],
        }
    ]

    built = chunking.build_chunks_from_pages(
        pages,
        doc_id="doc-ocr",
        source="pdf",
        embeddings_for_semantic=None,
    )

    assert [chunk["chunk_text"] for chunk in built] == ["Parent bullet.", "Child bullet."]


def test_build_chunks_from_pages_uses_gemini_rules_and_updates_section_carry(monkeypatch):
    parse_calls = []

    def fake_parse_policy_to_nodes(text: str, section_hint=None, lines_with_meta=None):
        parse_calls.append(section_hint)
        return [{"kind": "para", "line": text.strip(), "path": [section_hint] if section_hint else []}]

    def fake_group_atomic_rules_semantic(nodes, embeddings=None, sim_threshold=None, use_percentile=None, percentile=None):
        return [{"path": n.get("path", []), "text": n.get("line", "")} for n in nodes]

    def fake_semantic_refine_rules(rules, embeddings=None, enable=True, max_chars=None, breakpoint_threshold_type=None, breakpoint_threshold_amount=None):
        return rules

    monkeypatch.setattr(chunking, "parse_policy_to_nodes", fake_parse_policy_to_nodes)
    monkeypatch.setattr(chunking, "group_atomic_rules_semantic", fake_group_atomic_rules_semantic)
    monkeypatch.setattr(chunking, "semantic_refine_rules", fake_semantic_refine_rules)

    pages = [
        {"page": 1, "narrative_text": "bullet carry page", "gemini_rules": [{"path": ["Bab 1", "Bagian A"], "text": "Butir hasil OCR"}]},
        {"page": 2, "narrative_text": "Lanjutan butir"},
    ]

    built = chunking.build_chunks_from_pages(
        pages,
        doc_id="doc-gemini-rules",
        source="pdf",
        embeddings_for_semantic=None,
    )

    assert built[0]["chunk_text"] == "Bab 1 > Bagian A — Butir hasil OCR"
    assert parse_calls == [None, "Bab 1"]


def test_apply_gemini_ocr_chunking_only_processes_textract_pages(monkeypatch):
    import importlib
    import types

    monkeypatch.setitem(sys.modules, "langchain", types.ModuleType("langchain"))
    output_parsers = types.ModuleType("langchain.output_parsers")
    output_parsers.PydanticOutputParser = object
    monkeypatch.setitem(sys.modules, "langchain.output_parsers", output_parsers)

    prompts = types.ModuleType("langchain_core.prompts")
    prompts.ChatPromptTemplate = object
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", prompts)

    google_genai = types.ModuleType("langchain_google_genai")
    google_genai.ChatGoogleGenerativeAI = object
    monkeypatch.setitem(sys.modules, "langchain_google_genai", google_genai)

    pydantic = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return self.__dict__.copy()

    def _field(default=None, default_factory=None, description=None):
        return default_factory() if default_factory is not None else default

    pydantic.BaseModel = _BaseModel
    pydantic.Field = _field
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)

    sys.modules.pop("application.ingest.ocr_gemini_chunking", None)
    ocr_gemini_chunking = importlib.import_module("application.ingest.ocr_gemini_chunking")

    calls = []

    def fake_chunk(page_text: str, previous_hierarchy=None, profile=""):
        calls.append({"page_text": page_text, "previous_hierarchy": previous_hierarchy})
        return ocr_gemini_chunking.OCRHierarchyPage(
            primary_language="id",
            rules=[ocr_gemini_chunking.OCRHierarchyRule(path=["Bab 1"], text=page_text)],
        )

    monkeypatch.setattr(ocr_gemini_chunking, "chunk_ocr_page_with_gemini", fake_chunk)

    pages = [
        {"page": 1, "narrative_text": "Textract page", "ocr_used": True, "ocr_engine": "textract", "tables": []},
        {"page": 2, "narrative_text": "Gemini OCR page", "ocr_used": True, "ocr_engine": "gemini", "tables": []},
        {"page": 3, "narrative_text": "Table page", "ocr_used": True, "ocr_engine": "textract", "tables": [{"table_index": 0}], "gemini_chunks": [{"chunk_text": "table"}]},
    ]

    result = ocr_gemini_chunking.apply_gemini_ocr_chunking(pages)

    assert [call["page_text"] for call in calls] == ["Textract page"]
    assert result[0]["gemini_rules"] == [{"path": ["Bab 1"], "text": "Textract page"}]
    assert "gemini_rules" not in result[1]
    assert result[2]["gemini_chunks"] == [{"chunk_text": "table"}]


def test_chunk_ocr_page_with_gemini_sends_page_text_as_human_content(monkeypatch):
    import importlib
    import types

    monkeypatch.setitem(sys.modules, "langchain", types.ModuleType("langchain"))

    parser_module = types.ModuleType("langchain.output_parsers")

    class FakeParser:
        def __init__(self, pydantic_object=None):
            self.pydantic_object = pydantic_object

        def get_format_instructions(self):
            return "schema"

        def invoke(self, value):
            return value

        def __ror__(self, other):
            return other

    parser_module.PydanticOutputParser = FakeParser
    monkeypatch.setitem(sys.modules, "langchain.output_parsers", parser_module)

    prompts_module = types.ModuleType("langchain_core.prompts")

    captured = {}

    class FakePromptTemplate:
        def __init__(self, messages):
            self.messages = messages

        @classmethod
        def from_messages(cls, messages):
            captured["messages"] = messages
            return cls(messages)

        def partial(self, **kwargs):
            captured["partial"] = kwargs
            return self

        def __or__(self, other):
            return other

    prompts_module.ChatPromptTemplate = FakePromptTemplate
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", prompts_module)

    google_module = types.ModuleType("langchain_google_genai")

    class FakeChain:
        def __init__(self, model=None, google_api_key=None, temperature=None):
            captured["model"] = model
            captured["google_api_key"] = google_api_key
            captured["temperature"] = temperature

        def __or__(self, other):
            return self

        def invoke(self, payload):
            captured["payload"] = payload
            class Result:
                primary_language = "id"
                rules = []
            return Result()

    google_module.ChatGoogleGenerativeAI = FakeChain
    monkeypatch.setitem(sys.modules, "langchain_google_genai", google_module)

    pydantic = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return self.__dict__.copy()

    def _field(default=None, default_factory=None, description=None):
        return default_factory() if default_factory is not None else default

    pydantic.BaseModel = _BaseModel
    pydantic.Field = _field
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)

    sys.modules.pop("application.ingest.ocr_gemini_chunking", None)
    ocr_gemini_chunking = importlib.import_module("application.ingest.ocr_gemini_chunking")
    monkeypatch.setattr(ocr_gemini_chunking, "load_config", lambda *_args, **_kwargs: types.SimpleNamespace(gemini_chat_model="gemini-test", google_api_key_name="GEMINI_API_KEY"))
    monkeypatch.setattr(ocr_gemini_chunking.os, "environ", {"GEMINI_API_KEY": "secret"})

    ocr_gemini_chunking.chunk_ocr_page_with_gemini("OCR body", previous_hierarchy=["Bab 1"])

    assert captured["messages"][0][0] == "system"
    assert captured["messages"][1][0] == "human"
    assert captured["partial"]["previous_hierarchy"] == "Bab 1"
    assert captured["payload"] == {"page_text": "OCR body"}
