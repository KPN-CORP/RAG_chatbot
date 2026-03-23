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
