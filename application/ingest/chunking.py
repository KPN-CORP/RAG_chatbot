# ingest/chunking.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from modules.chunking_utils import (
    BULLET_MARK_RE,
    TRAIL_NUM_RE,
    _checksum,
    _clean_line,
    _is_header,
    _norm_for_embed,
    _normalize_header,
    _looks_like_page_marker,
    _bullet_level_and_text,
    _push_label_to_path,
    _ends_sentence,
    _cluster_indent_buckets,      # NEW
    _indent_level_for_x,          # NEW
    boundary_join,
)

# -----------------------------
# Public API (what you import)
# -----------------------------
__all__ = [
    "build_chunks_from_pages",
    "parse_policy_to_nodes",
    "group_atomic_rules_semantic",
    "semantic_refine_rules",
    "atomic_rules_to_chunks",
    "table_cells_to_chunks",
]

# -----------------------------
# Parse to nodes (structure-first)
# -----------------------------

# --- CONTINUATION-FIRST LOGIC (prevents mid-sentence splits) ---
def _can_continue_prev_node(out: List[Dict[str, Any]], path: List[str], candidate_level: Optional[int]) -> bool:
    if not out:
        return False
    prev = out[-1]
    # must be bullet or paragraph to join
    if prev["kind"] not in {"bullet", "para"}:
        return False
    # same hierarchical path
    if prev.get("path", []) != path:
        return False
    # if both have levels, require same level
    if "level" in prev and candidate_level is not None:
        if prev["level"] != candidate_level:
            return False
    # only continue if previous line hasn't ended a sentence
    return not _ends_sentence(prev.get("line", ""))

def parse_policy_to_nodes(
    text: str,
    section_hint: Optional[str] = None,
    lines_with_meta: Optional[List[Dict[str, Any]]] = None,  # NEW
) -> List[Dict[str, Any]]:
    """
    Convert raw narrative text into nodes with breadcrumb 'path' and 'kind'.

    node = {'path': ['SECTION','Subsection'], 'kind':'header'|'bullet'|'para', 'line': str}

    If `lines_with_meta` is provided (as emitted by `get_page_elements`), we use
    indentation (hanging_x) to assign bullet levels robustly. Otherwise we fall
    back to glyph-based detection only.
    """
    path: List[str] = [section_hint] if section_hint else []
    out: List[Dict[str, Any]] = []

    pending_bullet_level: Optional[int] = None  # if previous line was just a bullet glyph
    last_kind: Optional[str] = None

    # Build indentation buckets if meta is available
    indent_buckets: List[Tuple[float, float]] = []
    meta_by_index: Dict[int, Dict[str, Any]] = {}

    if lines_with_meta:
        for i, m in enumerate(lines_with_meta):
            meta_by_index[i] = m
        # cluster using hanging_x (prefer) or min_x as fallback
        xs = [
            (m.get("hanging_x") if m.get("hanging_x") is not None else m.get("min_x"))
            for m in lines_with_meta
        ]
        xs = [x for x in xs if x is not None]
        indent_buckets = _cluster_indent_buckets(xs, gap=12.0)

    # iterate lines; if meta is provided, sync by index
    lines = (text or "").splitlines()
    # Track last physical line index that contributed text to the current node
    last_phys_idx_for_node = None
    for idx, raw in enumerate(lines):
        # 1) skip empties & universal page/line markers
        if not raw.strip():
            continue
        if _looks_like_page_marker(raw):
            continue

        line = _clean_line(raw)
        if not line:
            continue

        # 2) header?
        if _is_header(line):
            # extra safety: normalize header and clear pending bullet
            title = _normalize_header(line).title()
            path = [title]
            out.append({"path": path[:], "kind": "header", "line": title})
            pending_bullet_level = None
            last_kind = "header"
            continue

        # 3) bullet detection (glyph + indentation)
        glyph_level, content = _bullet_level_and_text(line)

        indent_level: Optional[int] = None
        if lines_with_meta:
            m = meta_by_index.get(idx, {})
            x = m.get("hanging_x", m.get("min_x"))
            if x is not None:
                indent_level = _indent_level_for_x(float(x), indent_buckets)

        # Honor pending glyph-only level if we saw a standalone bullet marker before
        if glyph_level is None and pending_bullet_level is not None:
            glyph_level = pending_bullet_level
            content = line
            pending_bullet_level = None
        
        # If there is *either* a glyph-level or an indentation-level, treat as bullet candidate
        level: Optional[int] = None
        if glyph_level is not None and indent_level is not None:
            # reconcile: prefer indentation to keep nesting stable across mixed glyphs
            level = indent_level
        else:
            level = glyph_level if glyph_level is not None else indent_level

        if level is not None:
            # Handle pure label lines like "Kuota Maksimal:" that should shape the path
            if content == "":
                # standalone glyph line (e.g., '●') → carry level to next line
                pending_bullet_level = level
                last_kind = None
                continue

            if content.endswith(":"):
                # label → push into path at that level
                label = content[:-1].strip()
                if not path:
                    path = [label]
                else:
                    path = _push_label_to_path(path, level, label)
                out.append({"path": path[:], "kind": "label", "line": label})
                last_kind = "label"
                continue

            # If previous node exists and we're mid-sentence, join instead of split
            if _can_continue_prev_node(out, path, level):
                prev_meta = meta_by_index.get(last_phys_idx_for_node or idx - 1, {})
                out[-1]["line"] = boundary_join(out[-1]["line"], content, prev_meta)
                last_kind = out[-1]["kind"]
                last_phys_idx_for_node = idx
                continue

            # Otherwise, start a fresh bullet node
            out.append({"path": path[:], "kind": "bullet", "line": content, "level": level})
            last_kind = "bullet"
            last_phys_idx_for_node = idx
            continue

        # 4) plain paragraph / continuation
        if last_kind in {"bullet", "para"} and _can_continue_prev_node(out, path, candidate_level=None):
            prev_meta = meta_by_index.get(last_phys_idx_for_node or idx - 1, {})
            out[-1]["line"] = boundary_join(out[-1]["line"], line, prev_meta)
            last_kind = out[-1]["kind"]
            last_phys_idx_for_node = idx
        else:
            out.append({"path": path[:], "kind": "para", "line": line})
            last_kind = "para"
            last_phys_idx_for_node = idx

    return out


# -----------------------------
# Semantic grouping utilities
# -----------------------------


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity without external deps."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def _embed_small_texts(texts: List[str], embeddings) -> List[List[float]]:
    """
    Batch-embed short strings for semantic grouping (not stored).
    Expect embeddings to implement .embed_documents([...]).
    """
    if not texts:
        return []
    return embeddings.embed_documents(texts)


# -----------------------------
# Group bullets/paras into Atomic Rules (semantic)
# -----------------------------


def group_atomic_rules_semantic(
    nodes: List[Dict[str, Any]],
    embeddings,  # required (e.g., GoogleGenerativeAIEmbeddings instance)
    sim_threshold: float = 0.80,  # tune 0.78–0.86
    use_percentile: bool = False,
    percentile: int = 90,
) -> List[Dict[str, Any]]:
    """
    Group bullets/paras into atomic rules by semantic continuity:
      - New header => flush.
      - Path change => flush.
      - Else: cosine similarity of consecutive items >= threshold => same rule.

    If use_percentile=True, we compute consecutive similarities across the page
    and set threshold to the given percentile(similarities).
    """
    # Collect sequence with headers marked
    items: List[Tuple[List[str], str]] = []  # (path, text)
    paths: List[List[str]] = []
    for n in nodes:
        if n["kind"] == "header":
            items.append((["__HEADER__"], n["line"].strip()))
            paths.append(["__HEADER__"])
        else:
            items.append((n["path"][:], n["line"].strip()))
            paths.append(n["path"][:])

    if not items:
        return []

    # Embed all non-header lines once (batched)
    idxs, to_embed = [], []
    for i, (_, txt) in enumerate(items):
        if paths[i] and paths[i][0] == "__HEADER__":
            continue
        norm = _norm_for_embed(txt)
        idxs.append(i)
        to_embed.append(norm)
    vecs = _embed_small_texts(to_embed, embeddings) if to_embed else []
    emb_by_idx: Dict[int, List[float]] = {i: v for i, v in zip(idxs, vecs)}

    # Compute consecutive sims for same path
    sims: List[float] = []
    for i in range(1, len(items)):
        if paths[i - 1] and paths[i - 1][0] == "__HEADER__":
            continue
        if paths[i] and paths[i][0] == "__HEADER__":
            continue
        if paths[i] != paths[i - 1]:
            continue
        if (i - 1) not in emb_by_idx or i not in emb_by_idx:
            continue
        sims.append(float(_cosine(emb_by_idx[i - 1], emb_by_idx[i])))

    # Adaptive threshold (optional)
    if use_percentile and sims:
        s = sorted(sims)
        k = min(max(int(len(s) * (percentile / 100.0)) - 1, 0), len(s) - 1)
        sim_threshold = s[k]

    # Grouping pass
    rules: List[Dict[str, Any]] = []
    buf: List[str] = []
    cur_path: Optional[List[str]] = None

    def flush():
        nonlocal buf, cur_path
        if buf and cur_path:
            rules.append({"path": cur_path[:], "text": " ".join(buf).strip()})
        buf = []

    for i, (p, txt) in enumerate(items):
        if p and p[0] == "__HEADER__":
            flush()
            continue

        if cur_path is not None and p != cur_path:
            flush()

        if cur_path is None:
            cur_path = p[:]
            buf = [txt]
            continue

        # continuity check with previous item if same path and both embedded
        if i - 1 >= 0 and paths[i - 1] == cur_path and (i - 1) in emb_by_idx and i in emb_by_idx:
            sim = _cosine(emb_by_idx[i - 1], emb_by_idx[i])
            if sim >= sim_threshold:
                buf.append(txt)
            else:
                flush()
                cur_path = p[:]
                buf = [txt]
        else:
            flush()
            cur_path = p[:]
            buf = [txt]

    if buf:
        flush()
    return rules


# -----------------------------
# Semantic fallback for long prose
# -----------------------------


def semantic_refine_rules(
    rules: List[Dict[str, Any]],
    embeddings=None,
    enable: bool = True,
    max_chars: int = 1000,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 95,
) -> List[Dict[str, Any]]:
    """
    Split only long rules using LangChain SemanticChunker (if available).
    Keeps same path for all splits. No-op if not enabled or module missing.
    """
    if not enable or embeddings is None:
        return rules
    try:
        from langchain_experimental.text_splitter import SemanticChunker  # lazy import
    except Exception:
        return rules

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    refined: List[Dict[str, Any]] = []
    for r in rules:
        t = r["text"]
        if len(t) <= max_chars:
            refined.append(r)
            continue
        docs = splitter.create_documents([t])
        for d in docs:
            refined.append({"path": r["path"], "text": d.page_content.strip()})
    return refined


# -----------------------------
# Convert atomic rules -> chunks
# -----------------------------


def atomic_rules_to_chunks(
    rules: List[Dict[str, Any]], doc_id: str, source: str, page: Optional[int]
) -> List[Dict[str, Any]]:
    """Make child chunks for atomic rules (embed these)."""
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rules, 1):
        breadcrumb = " > ".join(r["path"]) if r["path"] else ""
        breadcrumb = TRAIL_NUM_RE.sub("", breadcrumb).strip()
        text = f"{breadcrumb} — {r['text']}".strip() if breadcrumb else r["text"].strip()
        meta = {
            "source": source,
            "page": page,
            "element_type": "policy_rule",
            "breadcrumb": breadcrumb,
            "list_id": None,
            "table_id": None,
            "row_id": None,
            "col_name": None,
        }
        cid = f"{doc_id}::rule::{page or 0:04d}::{i:04d}"
        csum = _checksum(text, source=source, page=page, extra={"kind": "policy_rule", "i": i})
        out.append(
            {
                "chunk_id": cid,
                "chunk_text": text,
                "checksum": csum,
                **meta,
            }
        )
    return out


def table_cells_to_chunks(
    table_dict_rows: List[Dict[str, Any]],
    doc_id: str,
    source: str,
    page: int,
    table_index: int,
    table_title: Optional[str] = None,
    *,
    id_col: Optional[str] = None,  # optional: force an ID column, else auto-pick first column
    include_identifier_cell: bool = True,  # include the ID column as a cell-chunk too
) -> List[Dict[str, Any]]:
    """
    Emit exactly one chunk per (row, col) cell.
    - No summaries, no collapsing, no header/region parsing.
    - breadcrumb = table_title (page header carried by the builder).
    - Text format: "<breadcrumb> — Row[<id_label>=<id_val>] • Col[<col>] ⇒ <value>"
    """
    out: List[Dict[str, Any]] = []
    if not table_dict_rows:
        return out

    cols = list(table_dict_rows[0].keys())
    # auto-pick ID column if not provided: just use the first column
    auto_id_col = cols[0] if cols else None
    _id_col = id_col or auto_id_col

    table_id = f"{doc_id}::table::{page:04d}::{table_index:03d}"

    for r_idx, row in enumerate(table_dict_rows):
        row_id = f"{table_id}::row::{r_idx:04d}"
        id_label = _normalize_header(_id_col) if _id_col else "Row"
        id_val = str(row.get(_id_col, r_idx)) if _id_col in row else str(r_idx)

        for c_idx, col in enumerate(cols):
            # optionally skip the identifier column as a separate cell
            if not include_identifier_cell and _id_col and col == _id_col:
                continue

            raw_val = row.get(col, "")
            val = "" if raw_val is None else str(raw_val).strip()
            if val == "":
                continue  # skip truly empty cells

            col_name = _normalize_header(col)
            row_tag = f"Row[{id_label}={id_val}]" if _id_col else f"Row[{r_idx}]"
            txt = (
                f"{(table_title + ' — ') if table_title else ''}{row_tag} • Col[{col_name}] ⇒ {val}"
            )

            cid = f"{row_id}::cell::{c_idx:03d}"
            csum = _checksum(
                txt,
                source=source,
                page=page,
                extra={"table": table_id, "row": r_idx, "col": col_name},
            )

            out.append(
                {
                    "chunk_id": cid,
                    "chunk_text": txt,
                    "checksum": csum,
                    "source": source,
                    "page": page,
                    "element_type": "table_cell",
                    "table_id": table_id,
                    "row_id": row_id,
                    "col_name": col_name,
                    "breadcrumb": table_title or "",
                    "list_id": None,
                }
            )
    return out


# -----------------------------
# Page-level builder (with section carry)
# -----------------------------


def build_chunks_from_pages(
    pages: List[Dict[str, Any]],
    doc_id: str,
    source: str,
    embeddings_for_semantic,  # embeddings instance for grouping & optional fallback
    enable_semantic_fallback: bool = True,
    semantic_fallback_max_chars: int = 1000,
    sim_threshold: float = 0.82,
    use_percentile: bool = False,
    percentile: int = 90,
) -> List[Dict[str, Any]]:
    """
    pages: list of dicts like:
      {'page': int, 'narrative_text': str, 'tables': [{'table_index': int, 'table_dct': [ {col:val}, ... ]}, ...]}

    Behavior:
      - Carry last seen ALL-CAPS header across pages (no '/pagebreak' markers).
      - Narrative → nodes → semantic grouping → (optional) length fallback → policy_rule chunks.
      - Tables → cell-level chunks (one chunk per cell).
    """
    built: List[Dict[str, Any]] = []
    section_carry: Optional[str] = None
    pages_sorted = sorted(pages, key=lambda p: (p.get("page") is None, p.get("page")))

    for pg in pages_sorted:
        page_no = pg.get("page")
        nar = (pg.get("narrative_text") or "").strip()
        tables = pg.get("tables") or []
        lines_with_meta=pg.get("narrative_lines")

        # detect this page header (if any)
        section_hint = section_carry
        if nar:
            first_line = ""
            for raw in nar.splitlines():
                raw = raw.strip()
                if raw:
                    first_line = raw
                    break
            if first_line and _is_header(_clean_line(first_line)):
                section_hint = _clean_line(first_line).title()

        # 1) Narrative → atomic rules
        if nar:
            nodes = parse_policy_to_nodes(nar, section_hint=section_hint, lines_with_meta=lines_with_meta)

            # track carry-over header
            for n in nodes:
                if n.get("kind") == "header":
                    section_carry = n.get("line") or section_carry
            if section_carry is None and section_hint:
                section_carry = section_hint

            rules = group_atomic_rules_semantic(
                nodes,
                embeddings=embeddings_for_semantic,
                sim_threshold=sim_threshold,
                use_percentile=use_percentile,
                percentile=percentile,
            )
            rules = semantic_refine_rules(
                rules,
                embeddings=embeddings_for_semantic,
                enable=enable_semantic_fallback,
                max_chars=semantic_fallback_max_chars,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
            built.extend(atomic_rules_to_chunks(rules, doc_id=doc_id, source=source, page=page_no))

        # 2) Tables → cell chunks
        for t in tables:
            t_idx = t.get("table_index", 0)
            rows = t.get("table_dct")
            if isinstance(rows, dict) and "table_dct" in rows:
                rows = rows["table_dct"]
            if not isinstance(rows, list):
                continue

            table_title = section_carry
            cell_chunks = table_cells_to_chunks(
                table_dict_rows=rows,
                doc_id=doc_id,
                source=source,
                page=page_no if page_no is not None else -1,
                table_index=t_idx,
                table_title=table_title,
            )
            built.extend(cell_chunks)

    return built
