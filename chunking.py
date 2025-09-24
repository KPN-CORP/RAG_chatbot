from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    # preferred provider package name
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    try:
        # other community/provider naming possibilities
        from langchain.embeddings import GoogleGenerativeAIEmbeddings  # hypothetical fallback
    except Exception:
        raise ImportError(
            "Unable to import GoogleGenerativeAIEmbeddings. "
            "Install the LangChain Google GenAI provider (e.g. `pip install langchain-google-genai`) "
            "and ensure LangChain is up to date."
        )

import hashlib
import unicodedata
import re
import uuid
import time
from typing import List, Dict, Any


def pages_to_chunks_old(pdf_pages, source, chunk_size=500, chunk_overlap=50):
    '''convert pdf pages to chunks'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = []
    table_id = 0
    chunk_id = 0
    for page in pdf_pages:
        content = page['narrative_text']
        if content.strip():  # Check if content is not empty
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                chunk_id += 1
                chunk_metadata = {
                    'source':source,
                    'page_number': page['page'],
                    'element_type': 'narrative',
                    'table_id': None,
                    'chunk_id': chunk_id,
                    'chunk_text': chunk,
                    'checksum': chunk  # not yet coded
                }
                all_chunks.append(chunk_metadata)
        tables = page['tables']
        if tables:
            for t in tables:
                table_id += 1
                if t['table_dct']:
                    for row in t['table_dct']:
                        chunk = " | ".join([f"{k}: {v}" for k, v in row.items()])
                        chunk_metadata = {
                            'source':source,
                            'page_number': page['page'],
                            'element_type': 'table',
                            'table_id': table_id,
                            'chunk_id': chunk_id,
                            'chunk_text': chunk,
                            'checksum': chunk  # not yet coded
                        }
                        all_chunks.append(chunk_metadata)
    return all_chunks

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

def row_dict_to_text_old(row: Dict[str, Any]) -> str:
    """
    Convert a row dict into a stable text string for embedding/checksum.
    Sort keys for deterministic order and canonicalize values.
    """
    if not isinstance(row, dict):
        return canonicalize_text(str(row))
    keys = sorted(row.keys(), key=lambda k: str(k).lower())
    parts = []
    for k in keys:
        v = row.get(k, "")
        v = canonicalize_text(str(v))
        parts.append(f"{k}: {v}")
    return " | ".join(parts)

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
    NOTE: This function does not persist or embed — it only returns canonical chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks: List[Dict[str, Any]] = []
    # table_index is local to each page (keeps original table_index); we'll also create unique table_id using source/page/table_index
    for page in pdf_pages:
        page_num = page.get("page")
        # 1) Narrative text -> chunked
        narrative = page.get("narrative_text") or ""
        if narrative and narrative.strip():
            # optional: canonicalize the narrative before splitting (keeps chunking deterministic)
            narrative_canon = canonicalize_text(narrative)
            pieces = text_splitter.split_text(narrative_canon)
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
