# ingest/chunking.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import hashlib, json, re, math
from modules.chunking_utils import (
    _clean_line, _is_header, _norm_for_embed, _checksum, _to_number, BULLET_MARK_RE
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
# Checksum / IDs
# -----------------------------



# -----------------------------
# Parse to nodes (structure-first)
# -----------------------------

def parse_policy_to_nodes(text: str, section_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert raw narrative text into nodes with breadcrumb 'path' and 'kind'.
    node = {'path': ['SECTION','Subsection'], 'kind':'header'|'bullet'|'para', 'line': str}
    - section_hint: carried header from previous page (if any).
    """
    path: List[str] = [section_hint] if section_hint else []
    out: List[Dict[str, Any]] = []

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue

        if _is_header(line):
            title = line.title()
            path = [title]
            out.append({'path': path[:], 'kind': 'header', 'line': title})
            continue

        if BULLET_MARK_RE.match(line):
            content = BULLET_MARK_RE.sub("", line).strip()
            if content.endswith(":"):
                label = content[:-1].strip()
                if path:
                    if len(path) == 1:
                        path = [path[0], label]
                    else:
                        path[-1] = label
                else:
                    path = [label]
                out.append({'path': path[:], 'kind': 'para', 'line': label})
            else:
                out.append({'path': path[:], 'kind': 'bullet', 'line': content})
        else:
            out.append({'path': path[:], 'kind': 'para', 'line': line})

    return out

# -----------------------------
# Semantic grouping utilities
# -----------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity without external deps."""
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-12
    nb = math.sqrt(sum(y*y for y in b)) or 1e-12
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
    embeddings,                           # required (e.g., GoogleGenerativeAIEmbeddings instance)
    sim_threshold: float = 0.82,          # tune 0.78–0.86
    use_percentile: bool = False,
    percentile: int = 90
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
    items: List[Tuple[List[str], str]] = []   # (path, text)
    paths: List[List[str]] = []
    for n in nodes:
        if n['kind'] == 'header':
            items.append((['__HEADER__'], n['line'].strip()))
            paths.append(['__HEADER__'])
        else:
            items.append((n['path'][:], n['line'].strip()))
            paths.append(n['path'][:])

    if not items:
        return []

    # Embed all non-header lines once (batched)
    idxs, to_embed = [], []
    for i, (_, txt) in enumerate(items):
        if paths[i] and paths[i][0] == '__HEADER__': 
            continue
        norm = _norm_for_embed(txt)
        idxs.append(i)
        to_embed.append(norm)
    vecs = _embed_small_texts(to_embed, embeddings) if to_embed else []
    emb_by_idx: Dict[int, List[float]] = {i: v for i, v in zip(idxs, vecs)}

    # Compute consecutive sims for same path
    sims: List[float] = []
    for i in range(1, len(items)):
        if paths[i-1] and paths[i-1][0] == '__HEADER__': 
            continue
        if paths[i] and paths[i][0] == '__HEADER__':
            continue
        if paths[i] != paths[i-1]:
            continue
        if (i-1) not in emb_by_idx or i not in emb_by_idx:
            continue
        sims.append(float(_cosine(emb_by_idx[i-1], emb_by_idx[i])))

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
            rules.append({'path': cur_path[:], 'text': " ".join(buf).strip()})
        buf = []

    for i, (p, txt) in enumerate(items):
        if p and p[0] == '__HEADER__':
            flush()
            continue

        if cur_path is not None and p != cur_path:
            flush()

        if cur_path is None:
            cur_path = p[:]
            buf = [txt]
            continue

        # continuity check with previous item if same path and both embedded
        if i-1 >= 0 and paths[i-1] == cur_path and (i-1) in emb_by_idx and i in emb_by_idx:
            sim = _cosine(emb_by_idx[i-1], emb_by_idx[i])
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
        t = r['text']
        if len(t) <= max_chars:
            refined.append(r)
            continue
        docs = splitter.create_documents([t])
        for d in docs:
            refined.append({'path': r['path'], 'text': d.page_content.strip()})
    return refined

# -----------------------------
# Convert atomic rules -> chunks
# -----------------------------

def atomic_rules_to_chunks(
    rules: List[Dict[str, Any]],
    doc_id: str,
    source: str,
    page: Optional[int]
) -> List[Dict[str, Any]]:
    """Make child chunks for atomic rules (embed these)."""
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rules, 1):
        breadcrumb = " > ".join(r['path']) if r['path'] else ""
        text = (f"{breadcrumb} — {r['text']}".strip() if breadcrumb else r['text'].strip())
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
        out.append({
            "chunk_id": cid,
            "chunk_text": text,
            "checksum": csum,
            **meta,
        })
    return out

# -----------------------------
# Table → cell-level chunks
# -----------------------------

def _normalize_header(h: str) -> str:
    return re.sub(r"\s+", " ", (h or "").strip())

def _cell_chunk_text(table_title: Optional[str], row_kv: Dict[str, Any], col_name: str, row_idx: int) -> str:
    row_keys = [k for k in row_kv.keys() if k.lower() in ("gol.", "gol", "grade", "id", "kode", "keterangan", "level", "kategori")]
    row_ident = None
    for k in row_keys:
        v = str(row_kv.get(k, "")).strip()
        if v:
            row_ident = f"{k}={v}"
            break
    if not row_ident:
        for k, v in row_kv.items():
            if str(v).strip():
                row_ident = f"{k}={v}"
                break
    title = f"{table_title} — " if table_title else ""
    value = str(row_kv.get(col_name, "")).strip()
    return f"{title}Row[{row_ident or row_idx}] • Col[{col_name}] ⇒ {value}"

def _guess_id_column(rows: List[Dict[str, Any]]) -> Optional[str]:
    """
    Universal ID-column guesser:
      - choose the column with the highest uniqueness ratio
      - break ties by: fewer empty values, then leftmost order in the first row
    Works for 'Gol.', 'ID', 'Code', 'Name', etc. without hardcoding.
    """
    if not rows:
        return None
    cols = list(rows[0].keys())
    n = len(rows)
    best: Tuple[float, int, str] = (-1.0, 10**9, "")  # (unique_ratio, empties, colname)
    for ci, c in enumerate(cols):
        vals = [str(r.get(c, "")).strip() for r in rows]
        empties = sum(1 for v in vals if not v)
        uniq = len(set(v for v in vals if v))
        unique_ratio = (uniq / max(n - empties, 1))
        score = (unique_ratio, -empties, -ci)  # maximize unique_ratio, minimize empties, prefer leftmost
        if (unique_ratio, empties, ci) > (best[0], -best[1], -best[2] if isinstance(best[2], int) else 0):
            best = (unique_ratio, empties, c)
    # small sanity: if uniqueness is extremely low, skip and let row index be the id
    return best[2] if isinstance(best[2], str) and best[0] >= 0.4 else None

def _row_is_constant(row: Dict[str, Any], id_col: Optional[str]) -> Optional[str]:
    """
    If all non-ID non-empty values in the row are the same string, return that text (to collapse).
    """
    vals = []
    for k, v in row.items():
        if id_col and k == id_col:
            continue
        s = str(v or "").strip()
        if s:
            vals.append(s)
    if not vals:
        return None
    return vals[0] if all(s == vals[0] for s in vals) else None


def _row_summary_text(title: Optional[str], id_label: str, id_val: str,
                      row: Dict[str, Any], id_col: Optional[str],
                      max_cols: int = 8) -> str:
    """
    Compact one-liner summary per row: "Title — IDLABEL=IDVAL — col1=val1; col2=val2; ..."
    Limits to max_cols for readability (adds “… +N more” if truncated).
    """
    prefix = f"{title} — " if title else ""
    parts = []
    shown = 0
    for k, v in row.items():
        if id_col and k == id_col:
            continue
        s = str(v or "").strip()
        if not s:
            continue
        parts.append(f"{_normalize_header(k)}={s}")
        shown += 1
        if shown >= max_cols:
            break
    remaining = sum(1 for k, v in row.items() if (not id_col or k != id_col) and str(v or "").strip()) - shown
    tail = f"; +{remaining} more" if remaining > 0 else ""
    return f"{prefix}{id_label} {id_val} — " + ("; ".join(parts) + tail if parts else "(empty)")

def table_cells_to_chunks(
    table_dict_rows: List[Dict[str, Any]],
    doc_id: str,
    source: str,
    page: int,
    table_index: int,
    table_title: Optional[str] = None,
    *,
    id_col: Optional[str] = None,
    emit_row_summary: bool = True,
    emit_cell_chunks: bool = True,
    collapse_constant_rows: bool = True,
    max_summary_cols: int = 8,
) -> List[Dict[str, Any]]:
    """
    Universal, minimal-assumption table chunking:
      - infers an ID column (or use the provided id_col)
      - emits ONE row summary chunk per row (optional)
      - collapses rows where all non-ID values are identical (optional)
      - emits per-cell chunks for non-empty, non-ID cells (optional)
      - keeps metadata small & compatible with your Milvus schema
    """
    out: List[Dict[str, Any]] = []
    if not table_dict_rows:
        return out

    # ID column: use provided or infer
    _id_col = id_col or _guess_id_column(table_dict_rows)
    table_id = f"{doc_id}::table::{page:04d}::{table_index:03d}"

    for r_idx, row in enumerate(table_dict_rows):
        id_label = _normalize_header(_id_col) if _id_col else "Row"
        id_val = str(row.get(_id_col, r_idx)) if _id_col else str(r_idx)
        row_id = f"{table_id}::row::{r_idx:04d}"

        # 1) Row-constant collapse (e.g., "Sesuai tagihan, ...")
        if collapse_constant_rows:
            const_text = _row_is_constant(row, _id_col)
            if const_text:
                # constant chunk
                txt = f"{table_title+' — ' if table_title else ''}{id_label} {id_val} — {const_text}"
                cid = f"{row_id}::constant"
                csum = _checksum(txt, source=source, page=page, extra={"table": table_id, "row": r_idx, "kind": "row_constant"})
                out.append({
                    "chunk_id": cid,
                    "chunk_text": txt,
                    "checksum": csum,
                    "source": source,
                    "page": page,
                    "element_type": "table_row_constant",
                    "table_id": table_id,
                    "row_id": row_id,
                    "col_name": None,
                    "breadcrumb": table_title or "",
                    "list_id": None,
                })
                # optional row summary
                if emit_row_summary:
                    rs = _row_summary_text(table_title, id_label, id_val, row, _id_col, max_cols=max_summary_cols)
                    rcid = f"{row_id}::summary"
                    rcsum = _checksum(rs, source=source, page=page, extra={"table": table_id, "row": r_idx, "kind": "row_summary"})
                    out.append({
                        "chunk_id": rcid,
                        "chunk_text": rs,
                        "checksum": rcsum,
                        "source": source,
                        "page": page,
                        "element_type": "table_row",
                        "table_id": table_id,
                        "row_id": row_id,
                        "col_name": None,
                        "breadcrumb": table_title or "",
                        "list_id": None,
                    })
                continue  # skip per-cell chunks for constant rows

        # 2) Row summary (compact recall anchor)
        if emit_row_summary:
            rs = _row_summary_text(table_title, id_label, id_val, row, _id_col, max_cols=max_summary_cols)
            rcid = f"{row_id}::summary"
            rcsum = _checksum(rs, source=source, page=page, extra={"table": table_id, "row": r_idx, "kind": "row_summary"})
            out.append({
                "chunk_id": rcid,
                "chunk_text": rs,
                "checksum": rcsum,
                "source": source,
                "page": page,
                "element_type": "table_row",
                "table_id": table_id,
                "row_id": row_id,
                "col_name": None,
                "breadcrumb": table_title or "",
                "list_id": None,
            })

        # 3) Per-cell chunks (non-ID, non-empty)
        if emit_cell_chunks:
            c_idx = -1
            for k, v in row.items():
                c_idx += 1
                if _id_col and k == _id_col:
                    continue
                s = str(v or "").strip()
                if not s:
                    continue
                col_name = _normalize_header(k)
                txt = f"{table_title+' — ' if table_title else ''}{id_label} {id_val} • {col_name}: {s}"
                cid = f"{row_id}::cell::{c_idx:03d}"
                csum = _checksum(txt, source=source, page=page, extra={"table": table_id, "row": r_idx, "col": col_name})
                # optional numeric value (tiny, helpful for range filters)
                val_num = _to_number(s)

                out.append({
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
                    # minimal, universal extra for numeric range queries
                    "value_num": val_num,
                })

    return out

# -----------------------------
# Page-level builder (with section carry)
# -----------------------------

def build_chunks_from_pages(
    pages: List[Dict[str, Any]],
    doc_id: str,
    source: str,
    embeddings_for_semantic,               # embeddings instance for grouping & optional fallback
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
            nodes = parse_policy_to_nodes(nar, section_hint=section_hint)

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
                percentile=percentile
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
