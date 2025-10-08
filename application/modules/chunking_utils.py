"""Utility helpers supporting deterministic document chunking."""

import hashlib

# from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Optional

# -----------------------------
# Checksum / IDs
# -----------------------------


def _checksum(
    text: str,
    source: Optional[str] = None,
    page: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable checksum for dedupe (text + context)."""
    m = hashlib.sha1()
    m.update(text.encode("utf-8"))
    if source:
        m.update(f"|src:{source}".encode("utf-8"))
    if page is not None:
        m.update(f"|pg:{page}".encode("utf-8"))
    if extra:
        m.update(
            f"|x:{json.dumps(extra, sort_keys=True, ensure_ascii=False)}".encode(
                "utf-8"
            )
        )
    return m.hexdigest()


# -----------------------------
# Text normalization / heuristics
# -----------------------------

# HDR_RE = re.compile(r"^[A-Z0-9 ./-]{6,}$")          # heuristic ALL-CAPS headers
HDR_RE = re.compile(r"^[A-Z0-9 ()/.,&\-’'–—:]{6,}$")
TRAIL_NUM_RE = re.compile(r"(\s\d{1,3})$")  # trailing page line nums like " 21"
BULLET_MARK_RE = re.compile(r"^(?:[•●\-–▪]|○|\d+\.)\s*")


def _clean_line(s: str) -> str:
    """Remove whitespace noise and trailing numbers from a line."""

    s = s.strip()
    s = TRAIL_NUM_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _is_header(line: str) -> bool:
    """Heuristically determine whether *line* looks like a header chunk."""

    return bool(HDR_RE.match(line)) and (" " in line or "-" in line or "/" in line)


def _norm_for_embed(s: str) -> str:
    """Normalise a string before creating embeddings."""

    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_number(s: str) -> Optional[float]:
    """Generic numeric parse: '1.200'->1200, '1200'->1200, '80'->80, else None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # 1) 1.200 -> 1200 (thousand separators with dots)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        try:
            return float(int(s.replace(".", "")))
        except Exception:
            return None
    # 2) plain int/float
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None
    return None
