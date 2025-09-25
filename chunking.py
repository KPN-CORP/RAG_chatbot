from langchain.text_splitter import RecursiveCharacterTextSplitter

import hashlib
import unicodedata
import re
import uuid
import time
from typing import List, Dict, Any
from continuation_words import indo_continuations as _IND_CONT, english_continuation as _EN_CONT

# ---------- Config / heuristics ----------
# Strong sentence terminator
_STRONG_END = re.compile(r'[\.!\?…;:]$')
# Heading heuristic (short, many uppercase)
_HEADING_HEURISTIC = re.compile(r'^[A-Z0-9 \-\/]{1,80}$')
# Standalone page/line number
_STANDALONE_NUMBER = re.compile(r'^\d+\s*$')
# List start markers
_LIST_START = re.compile(r'^[\u2022\-\*\•\●]\s+|^\d+[\.\)]\s+')

# Combined connector set (lowercase)
_CONNECTORS = _IND_CONT.union(_EN_CONT)


def canonicalize_text(s: str, lower: bool = False) -> str:
    """Normalize text for stable checksum: unicode, collapse whitespace, optional lowercase."""
    if s is None:
        s = ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.strip()
    s = re.sub(r"\s+", " ", s).strip()
    # normalize smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", "'")
    # normalize common dash placeholders
    if s in ["-", "—", "–", "n/a", "N/A", "NA", "null", "None"]:
        return ""
    return s

def sha256_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def row_dict_to_text(row: Dict[str, Any]) -> str:
    """
    Convert a row dict into a stable, LLM- and vector-db-friendly text string.
    - Keeps the same signature and return type as your original function.
    - Produces natural-language sentences (one per non-empty cell) joined by newlines.
    - Deterministic ordering: keys sorted case-insensitively.
    - Canonicalizes whitespace and normalizes common "empty" tokens.
    - If a common primary key like 'Gol.'/'Gol'/'ID' exists, sentences are prefixed with it
      (e.g. "For Gol. 9, the value for 'Header' is ...").
    """

    if not isinstance(row, dict):
        return canonicalize_text(row)

    # deterministic key order
    keys = sorted(row.keys(), key=lambda k: str(k).lower())

    # detect a primary-key-like column if present
    pk_candidates = [k for k in keys if str(k).lower() in {"gol.", "gol", "id", "no", "nomor", "kode"}]
    primary_key = pk_candidates[0] if pk_candidates else None
    primary_val = canonicalize_text(row.get(primary_key)) if primary_key else ""

    sentences = []
    for k in keys:
        # canonicalize header and value
        header = canonicalize_text(k)
        value = canonicalize_text(row.get(k))
        # skip empties
        if value == "":
            continue
        # keep header readable but bounded
        if len(header) > 120:
            header = header[:80].rstrip() + " … " + header[-30:].lstrip()
        # create LLM-friendly sentence; prefer "For <PK>" when available and this isn't the PK cell itself
        if primary_key and k != primary_key and primary_val:
            sentence = f"For {primary_key} {primary_val}, the value for '{header}' is {value}."
        elif primary_key and k == primary_key:
            # describe the primary key itself
            sentence = f"{primary_key}: {value}."
        else:
            sentence = f"The value for '{header}' is {value}."
        sentences.append(sentence)

    # fallback: if everything was empty (or only primary present but skipped), return canonicalized row repr
    if not sentences:
        # produce compact "k: v | k2: v2" style as last resort (deterministic)
        parts = []
        for k in keys:
            v = canonicalize_text(row.get(k))
            parts.append(f"{k}: {v}" if v else f"{k}:")
        return " | ".join(parts)

    # join sentences with newline to keep single-string output while preserving chunk granularity
    return "\n".join(sentences)

# ---------- Helper functions (from earlier answers) ----------
_sentence_split_re = re.compile(r'(?<=[\.\?!])\s+|\n+')

def split_into_units(text: str) -> List[str]:
    t = text.replace('\r\n', '\n').strip()
    lines = t.split('\n')
    units = []
    buffer = []
    for line in lines:
        sline = line.strip()
        if not sline:
            if buffer:
                units.append(" ".join(buffer).strip())
                buffer = []
            continue

        # Heading heuristic
        if len(sline) <= 80 and re.match(r'^[A-Z0-9 \-\/]+$', sline) and sum(1 for c in sline if c.isupper())/max(1,len(sline)) > 0.4:
            if buffer:
                units.append(" ".join(buffer).strip())
                buffer = []
            units.append(sline)
            continue

        # Bullets
        if re.match(r'^[\u2022\u2023\u25E6\u2043\u2219\-\*\•\●]\s+', sline) or sline.startswith('●'):
            if buffer:
                units.append(" ".join(buffer).strip())
                buffer = []
            units.append(re.sub(r'^[\u2022\u2023\u25E6\u2043\u2219\-\*\•\●]\s*', '', sline))
            continue

        # Standalone numbers (like page markers)
        if re.match(r'^\d+\s*$', sline):
            if buffer:
                units.append(" ".join(buffer).strip())
                buffer = []
            units.append(sline)
            continue

        if re.match(r'^\d+[\.\)]\s+', sline):
            if buffer:
                units.append(" ".join(buffer).strip())
                buffer = []
            units.append(sline)
            continue

        buffer.append(sline)

    if buffer:
        units.append(" ".join(buffer).strip())

    # merge stray numeric separators with next if needed
    merged = []
    skip_next = False
    for i, u in enumerate(units):
        if skip_next:
            skip_next = False
            continue
        if re.match(r'^\d+\s*$', u) and i+1 < len(units):
            merged.append(f"{u} {units[i+1]}".strip())
            skip_next = True
        else:
            merged.append(u)
    return merged

def split_long_unit_by_sentence(unit: str, max_chars: int) -> List[str]:
    parts = [s.strip() for s in _sentence_split_re.split(unit) if s.strip()]
    groups = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip() if cur else p
        else:
            if cur:
                groups.append(cur)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    groups.append(p[i:i+max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        groups.append(cur)
    return groups

def semantic_chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    units = split_into_units(text)
    chunks = []
    cur = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > chunk_size:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            pieces = split_long_unit_by_sentence(unit, chunk_size)
            for p in pieces:
                if chunks and chunk_overlap > 0:
                    tail = chunks[-1][-chunk_overlap:]
                    chunks.append((tail + " " + p).strip())
                else:
                    chunks.append(p)
            continue

        if cur and (len(cur) + 1 + len(unit) > chunk_size):
            chunks.append(cur.strip())
            if chunk_overlap > 0:
                overlap_text = chunks[-1][-chunk_overlap:]
                cur = overlap_text + " " + unit
            else:
                cur = unit
        else:
            cur = (cur + " " + unit).strip() if cur else unit

    if cur:
        chunks.append(cur.strip())
    return chunks

# ---------- List parser (returns item dicts) ----------
_numbered_item_re = re.compile(r'^\s*(\d+)[\.\)]\s*(.*)', re.UNICODE)
_bullet_item_re   = re.compile(r'^\s*[\u2022\u2023\u25E6\u2043\u2219\-\*\•\●]\s*(.*)', re.UNICODE)

def parse_list_block(text_block: str, header_hint: str = None, list_id: str = None) -> List[Dict]:
    import uuid as _uuid
    if list_id is None:
        list_id = str(_uuid.uuid4())

    lines = [ln.rstrip() for ln in text_block.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    header = header_hint
    if not header and lines:
        if re.match(r'^[A-Z0-9 \-/]{3,80}$', lines[0].strip()):
            header = lines[0].strip()
            lines = lines[1:]

    items = []
    buffer = None
    current_pos = 0

    def flush_buffer():
        nonlocal buffer, items, current_pos
        if buffer:
            idx, txt = buffer
            chunk_text = (header + " | " + txt) if header else txt
            items.append({
                "list_id": list_id,
                "header": header,
                "item_index": idx,
                "text": txt,
                "chunk_text": chunk_text,
                "position": current_pos
            })
            current_pos += 1
            buffer = None

    for ln in lines:
        ln_stripped = ln.strip()
        m = _numbered_item_re.match(ln_stripped)
        if m:
            flush_buffer()
            idx = int(m.group(1))
            text_after = m.group(2).strip()
            buffer = (idx, text_after)
            continue

        m2 = _bullet_item_re.match(ln_stripped)
        if m2:
            flush_buffer()
            buffer = (None, m2.group(1).strip())
            continue

        if re.match(r'^\d+\s*$', ln_stripped):
            flush_buffer()
            continue

        if buffer:
            idx, prev = buffer
            buffer = (idx, prev + " " + ln_stripped)
        else:
            buffer = (None, ln_stripped)

    flush_buffer()
    return items

# ---------- Small helper: extract list blocks from a narrative ----------
def extract_list_blocks(text: str) -> List[Dict[str, str]]:
    """
    Naive line-scanner that extracts consecutive lines that look like a list block:
    A block starts when we see either:
      - a heading-like line (short ALLCAPS) followed by lines that contain bullets or numbered items
      - OR we encounter a line that starts with bullet/number and collect until blank or next big heading
    Returns list of dicts: {"block": <text>, "header": <maybe header>}
    """
    lines = text.replace('\r\n', '\n').split('\n')
    n = len(lines)
    i = 0
    blocks = []
    while i < n:
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue

        # detect if this line is a heading candidate and next lines show list markers
        header_candidate = None
        if len(ln) <= 80 and re.match(r'^[A-Z0-9 \-\/]{3,80}$', ln) and i+1 < n:
            # look ahead for bullet or numbered line in next 3 lines
            lookahead = "\n".join(lines[i+1:min(n, i+5)])
            if re.search(r'(^|\n)\s*[\u2022\-\*\•\●]|\n\s*\d+[\.\)]', lookahead):
                header_candidate = ln
                start = i + 1
                # collect list lines
                j = start
                while j < n and lines[j].strip() != "":
                    # stop if we reach another header-like line with no bullets following (conservative)
                    j += 1
                block_text = "\n".join(lines[i:j])
                blocks.append({"block": block_text, "header": header_candidate, "start": i, "end": j})
                i = j
                continue

        # detect direct list start (line starts with bullet or number)
        if re.match(r'^[\u2022\-\*\•\●]\s+|^\d+[\.\)]\s+', ln) or ln.startswith('●'):
            j = i
            while j < n and lines[j].strip() != "":
                j += 1
            block_text = "\n".join(lines[i:j])
            blocks.append({"block": block_text, "header": None, "start": i, "end": j})
            i = j
            continue

        i += 1

    return blocks


# ---------- Helper tests ----------
def _is_heading_like(line: str) -> bool:
    l = line.strip()
    if not l:
        return False
    if _HEADING_HEURISTIC.match(l):
        # require a reasonable fraction uppercase to be considered heading
        upper_frac = sum(1 for c in l if c.isupper()) / max(1, len(l))
        return upper_frac > 0.4
    return False

def _first_token_lower_or_connector(line: str) -> bool:
    ln = line.strip()
    if not ln:
        return False
    # first char lowercase is strong signal
    if ln[0].islower():
        return True
    # starts with punctuation commonly used mid-sentence
    if ln[0] in {',','('}:
        return True
    # first token connector
    first = re.split(r'\s+', ln)[0].strip(",(").lower()
    if first in _CONNECTORS:
        return True
    return False

# ---------- Main merge function with hyphenation ----------
def merge_across_page_breaks(
    pages: List[str],
    max_join_lines: int = 2
) -> List[str]:
    """
    Merge page narratives where a sentence is likely split across a page break.
    - pages: list of page narrative strings in order
    - max_join_lines: how many lines to borrow from next page when joining (default 2)
    Returns a new list of page narratives (same length) where some next-page lines
    have been stolen and appended to previous page when safe.
    """
    merged = []
    i = 0
    n = len(pages)
    # operate on a mutable copy to allow removing borrowed lines from next page
    pages_copy = list(pages)

    while i < n:
        cur = pages_copy[i] or ""
        # prepare last non-empty line check
        cur_lines = [ln for ln in cur.splitlines()]
        last_nonempty = ""
        for ln in reversed(cur_lines):
            if ln.strip():
                last_nonempty = ln.rstrip()
                break

        will_join = False
        force_hyphen_join = False
        # If last non-empty line ends with hyphen -> force join (high confidence)
        if last_nonempty.endswith('-'):
            force_hyphen_join = True
            will_join = True
        else:
            # If last line doesn't end with strong punctuation and is not heading/list/number -> candidate
            if last_nonempty and not _STRONG_END.search(last_nonempty) and not _is_heading_like(last_nonempty) \
               and not _STANDALONE_NUMBER.match(last_nonempty) and not _LIST_START.match(last_nonempty):
                will_join = True

        if i+1 < n and will_join:
            nxt = pages_copy[i+1] or ""
            nxt_lines = [ln for ln in nxt.splitlines()]
            # find first non-empty line on next page
            first_nonempty = ""
            first_idx = None
            for idx, ln in enumerate(nxt_lines):
                if ln.strip():
                    first_nonempty = ln.strip()
                    first_idx = idx
                    break

            # if no content on next page -> nothing to join
            if not first_nonempty:
                merged.append(cur)
                i += 1
                continue

            # do not join if next line is heading/list/standalone number
            if _is_heading_like(first_nonempty) or _LIST_START.match(first_nonempty) or _STANDALONE_NUMBER.match(first_nonempty):
                merged.append(cur)
                i += 1
                continue

            # Hyphenation case handling (force join): remove trailing hyphen and concatenate properly
            if force_hyphen_join:
                # remove trailing hyphen from last_nonempty
                # if next starts with lowercase, just concatenate; else add a space between
                first_part = first_nonempty
                # remove trailing '-' from cur
                # replace only the last occurrence
                cur = cur.rstrip()
                # drop trailing hyphen
                cur = cur[:-1]
                # decide whether to insert a space
                if first_part and first_part[0].isupper():
                    # if next part starts uppercase, add a space
                    cur = cur + " " + first_part
                else:
                    # lowercase likely continuation inside word, no extra space
                    cur = cur + first_part
                # remove the borrowed line(s) from next page
                remaining = nxt_lines[first_idx+1:] if first_idx is not None else nxt_lines
                pages_copy[i+1] = "\n".join(remaining).lstrip()
                merged.append(cur)
                i += 1
                continue

            # otherwise, continuation heuristics: check first token lowercase or connector
            if _first_token_lower_or_connector(first_nonempty):
                # collect up to max_join_lines from next page (but stop at headings/list markers)
                borrowed = []
                taken = 0
                for ln in nxt_lines[first_idx:]:
                    if ln.strip() == "":
                        break
                    # stop if the line looks like a heading or a list start (conservative)
                    if _is_heading_like(ln.strip()) or _LIST_START.match(ln.strip()):
                        break
                    borrowed.append(ln.strip())
                    taken += 1
                    if taken >= max_join_lines:
                        break

                if borrowed:
                    appended = " ".join(borrowed)
                    cur = cur.rstrip() + " " + appended
                    # remove borrowed lines from next page
                    remaining = nxt_lines[first_idx + taken:]
                    pages_copy[i+1] = "\n".join(remaining).lstrip()
                    merged.append(cur)
                    i += 1
                    continue

        # default: no join
        merged.append(cur)
        i += 1

    return merged

def pages_to_chunks(
    pdf_pages: List[Dict[str, Any]],
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    language: str = "id"
) -> List[Dict[str, Any]]:
    """
    Convert cleaned/formatted pdf_pages into chunk metadata ready for dedupe/embedding.
    Each returned item is a dict with fields including chunk_id, chunk_text, checksum and metadata.
    Upgraded pages_to_chunks:
     - Extracts list blocks and indexes them as item-level chunks
     - Removes list blocks from narrative before semantic chunking
    NOTE: This function does not persist or embed — it only returns canonical chunks.
   
    Upgraded: extracts list blocks -> item-level chunks; removes list blocks from narrative before semantic chunking. 
    """

    all_chunks: List[Dict[str, Any]] = []
    # table_index is local to each page (keeps original table_index); we'll also create unique table_id using source/page/table_index
     
    # --- 0) Ensure pages are sorted by page number (important) ---
    pdf_pages_sorted = sorted(pdf_pages, key=lambda p: int(p.get("page", 0)))
    
# --- 1) Build list of narrative strings in order ---
    narrative_texts = [p.get("narrative_text") or "" for p in pdf_pages_sorted]
     # --- 2) Merge across page breaks (hyphenation-aware) ---
    merged_texts = merge_across_page_breaks(narrative_texts, max_join_lines=2)

    # Safety: ensure merged_texts length matches pages; if not, pad/truncate conservatively
    if len(merged_texts) < len(pdf_pages_sorted):
        merged_texts += [""] * (len(pdf_pages_sorted) - len(merged_texts))
    elif len(merged_texts) > len(pdf_pages_sorted):
        merged_texts = merged_texts[: len(pdf_pages_sorted)]
    
    # merged_texts is a list of strings aligned with narrative_texts (same length)
    # write them back into page dicts (preserve other keys)
    for idx, page_dict in enumerate(pdf_pages_sorted):
        page_dict["narrative_text"] = merged_texts[idx]

    for page in pdf_pages_sorted:
        page_num = page.get("page")
        # 1) Narrative text -> chunked
        narrative = page.get("narrative_text") or ""
        if narrative and narrative.strip():
            # optional: canonicalize the narrative before splitting (keeps chunking deterministic)
            narrative_canon = canonicalize_text(narrative)
            # 1) find list blocks and parse them into item-level chunks
            list_blocks = extract_list_blocks(narrative_canon)
            # keep track of ranges to remove from narrative
            remove_spans = []
            for lb in list_blocks:
                block_text = lb["block"]
                header_hint = lb.get("header")
                items = parse_list_block(block_text, header_hint=header_hint)
                for it in items:
                    cid = uuid.uuid4().hex
                    checksum = sha256_checksum(it["chunk_text"]) if 'sha256_checksum' in globals() else ""  # replace with your checksum fn
                    meta = {
                        "chunk_id": cid,
                        "source": source,
                        "page": page_num,
                        "element_type": "list_item",
                        "list_id": it["list_id"],
                        "item_index": it["item_index"],
                        "position": it["position"],
                        "table_id": None,
                        "row_id": None, 
                        "language": language,
                        "chunk_text": it["chunk_text"],
                        "checksum": checksum,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    all_chunks.append(meta)

                # mark span for removal
                remove_spans.append((lb["start"], lb["end"]))

            # remove list blocks from narrative by line indices (to avoid duplication)
            if remove_spans:
                lines = narrative_canon.split('\n')
                keep_lines = []
                for idx, ln in enumerate(lines):
                    drop = False
                    for s, e in remove_spans:
                        if idx >= s and idx < e:
                            drop = True
                            break
                    if not drop:
                        keep_lines.append(ln)
                narrative_canon = "\n".join(keep_lines).strip()

            # 2) semantic-chunk the remaining narrative
            pieces = semantic_chunk(narrative_canon, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for piece in pieces:
                cid = uuid.uuid4().hex
                checksum = sha256_checksum(canonicalize_text(piece))
                meta = {
                    "chunk_id": cid,
                    "source": source,
                    "page": page_num,
                    "element_type": "narrative",
                    "table_id": None,
                    "row_id": None,
                    "list_id": None,
                    "bbox": None,
                    "language": language,
                    "chunk_text": piece,
                    "checksum": checksum,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                all_chunks.append(meta)

        # 2) Tables -> row level chunks (handle nested / different shapes)
        tables = page.get("tables") or []
        for t in tables:
            table_index = t.get("table_index")
            bbox = t.get("bbox")
            table_id = f"{source}_p{page_num}_t{table_index}"
            table_dct = t.get("table_dct") or []

            # If it's a list of dict rows (most common), handle row-by-row
            if isinstance(table_dct, list) and table_dct and all(isinstance(r, dict) for r in table_dct):
                for ridx, row in enumerate(table_dct):
                    text = row_dict_to_text(row)
                    cid = uuid.uuid4().hex
                    checksum = sha256_checksum(canonicalize_text(text))
                    meta = {
                        "chunk_id": cid,
                        "source": source,
                        "page": page_num,
                        "element_type": "table_row",
                        "table_id": table_id,
                        "table_index": table_index,
                        "row_id": ridx,
                        "list_id": None,
                        "bbox": bbox,
                        "language": language,
                        "chunk_text": text,
                        "checksum": checksum,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    all_chunks.append(meta)

            else:
                # Generic fallback: table_dct might be nested like your example.
                # Flatten reasonable nested patterns into readable text rows.
                # If table_dct is list but not list-of-dicts, attempt to extract inner dict lists.
                if isinstance(table_dct, list) and table_dct:
                    # try to find inner lists of dicts
                    found = False
                    for item in table_dct:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                                    # treat each inner dict as a row
                                    for ridx, row in enumerate(v):
                                        text = row_dict_to_text(row)
                                        cid = uuid.uuid4().hex
                                        checksum = sha256_checksum(canonicalize_text(text))
                                        meta = {
                                            "chunk_id": cid,
                                            "source": source,
                                            "page": page_num,
                                            "element_type": "table_row",
                                            "table_id": table_id,
                                            "table_index": table_index,
                                            "row_id": ridx,
                                            "list_id": None,
                                            "bbox": bbox,
                                            "language": language,
                                            "chunk_text": text,
                                            "checksum": checksum,
                                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                                        }
                                        all_chunks.append(meta)
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        # fallback: represent the table as a single textual chunk (serialized)
                        text = canonicalize_text(repr(table_dct))
                        cid = uuid.uuid4().hex
                        checksum = sha256_checksum(text)
                        meta = {
                            "chunk_id": cid,
                            "source": source,
                            "page": page_num,
                            "element_type": "table_text",
                            "table_id": table_id,
                            "table_index": table_index,
                            "row_id": None,
                            "list_id": None,
                            "bbox": bbox,
                            "language": language,
                            "chunk_text": text,
                            "checksum": checksum,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        }
                        all_chunks.append(meta)
                else:
                    # empty or unknown format -> skip or keep a representation
                    continue

    return all_chunks
