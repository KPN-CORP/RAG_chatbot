"""Generic string parsing helpers shared across ingestion utilities."""

import json
import re

from logging_config import get_logger
from typing import Optional
from pathlib import Path
import hashlib
import os


logger = get_logger(__name__)


def extract_json_string(s) -> Optional[str]:
    """Return a JSON string extracted from ``s`` or ``None`` if not found/invalid."""
    if s is None:
        return None

    # Gracefully handle non-text inputs (e.g., dict, list, structured messages)
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="ignore")
    elif not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        matches = list(re.finditer(r"\{[\s\S]*\}", s))
        if not matches:
            logger.warning("No JSON found in text", extra={"preview": s[:200]})
            return None
        candidate = matches[-1].group(0).strip()

    try:
        # sanity check it’s valid JSON
        candidate_json = json.loads(candidate)
        
        candidate_json['sources'] = [dict(t) for t in {tuple(d.items()) for d in candidate_json['sources']}]
        candidate_string = json.dumps(candidate_json)
        return candidate_string
    
    except json.JSONDecodeError as e:
        logger.warning("Extracted text is not valid JSON", extra={"error": str(e), "preview": candidate[:200]})
        return None
    
# def extract_json(text_or_msg):
#     t = getattr(text_or_msg, "content", str(text_or_msg))

#     # Prefer fenced ```json blocks
#     m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
#     if m:
#         return m.group(1).strip()

#     # Fallback: last JSON object in the string
#     m = re.search(r"(\{[\s\S]*\})\s*$", t)
#     return m.group(1).strip() if m else t.strip()

def _build_document_identifier(filename: str, data: bytes) -> str:
    """Create a stable document identifier using filename and SHA-256 digest."""

    name = Path(filename).stem if filename else "document"
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"{safe_name or 'document'}-{digest}"

def unwrap_meta(meta: dict) -> dict:
    # Flatten one level of nested 'metadata'
    inner = (meta or {}).get("metadata") or {}
    flat = {k: v for k, v in (meta or {}).items() if k != "metadata"}
    flat.update(inner)
    return flat

def meta_filename(meta: dict) -> str:
    m = unwrap_meta(meta)
    src = m.get("filename") or m.get("source") or m.get("path") or m.get("file_name")
    if src:
        try: return Path(src).name
        except Exception: return str(src)
    return "N/A"

def meta_page(meta: dict):
    m = unwrap_meta(meta)
    return m.get("page") or m.get("page_number") or m.get("pagenum")

def resolve_embedding_api_key(google_api_key_env: str) -> str:
    """Fetch the Gemini embedding API key using known environment fallbacks."""

    candidates = [
        os.environ.get(google_api_key_env),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
    ]
    for value in candidates:
        if value:
            return value
    return ""