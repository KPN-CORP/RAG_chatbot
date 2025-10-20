"""Utility helpers supporting deterministic document chunking."""

import hashlib

# from dataclasses import dataclass
import json
import re
from typing import List, Any, Dict, Optional

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
        m.update(f"|x:{json.dumps(extra, sort_keys=True, ensure_ascii=False)}".encode("utf-8"))
    return m.hexdigest()


# -----------------------------
# Text normalization / heuristics
# -----------------------------

# HDR_RE = re.compile(r"^[A-Z0-9 ./-]{6,}$")          # heuristic ALL-CAPS headers
HDR_RE = re.compile(r"^[A-Z0-9 ()/.,&\-’'–—:]{6,}$")

TRAIL_NUM_RE = re.compile(r"(\s\d{1,3})$")  # trailing page line nums like " 21"
# Broader trailing pagination crumbs at end of a line (allows dash + up to 5 digits)
TRAIL_NUM_RE = re.compile(r"(\s*[–—-]?\s*\d{1,5})$")

# Bullet markers (unchanged, extend if you meet new glyphs)
BULLET_MARK_RE = re.compile(r"^(?:[•●\-–▪]|○|\d+\.)\s*")

# Page-marker patterns in multiple languages (very permissive but short)
PAGE_TAG_RE = re.compile(r"(?i)^(page|hal(\.|aman)?|pg)\s*\d{1,5}$")
ROMAN_RE = re.compile(
    r"(?i)^(?=[MDCLXVI]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
DECOR_ONLY_RE = re.compile(r"^[\s\-\—\–_.•○●|·]+$")



def _clean_line(s: str) -> str:
    """Remove whitespace noise and trailing numbers from a line."""

    s = s.strip()
    s = TRAIL_NUM_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_header(h: str) -> str:
    return re.sub(r"\s+", " ", (h or "").strip())


def _is_header(line: str) -> bool:
    """Heuristically determine whether *line* looks like a header chunk."""

    return bool(HDR_RE.match(line)) and (" " in line or "-" in line or "/" in line)


def _norm_for_embed(s: str) -> str:
    """Normalise a string before creating embeddings."""

    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _mostly_digits_or_roman(s: str) -> bool:
    """
    True if the token (after stripping decorations) is short and either all digits
    or a valid Roman numeral. Handles '— 12 —', 'xii', '| 34 |', etc.
    """
    t = s.strip()
    if not t:
        return False
    core = re.sub(r"[\s\-\—\–\|_\.·•●○]+", "", t)
    if not core:
        return False
    if len(core) <= 6 and (core.isdigit() or ROMAN_RE.match(core)):
        return True
    return False

def _looks_like_page_marker(s: str) -> bool:
    """
    Universal text-only filter for page numbers / headers / footers:
    - '12', '— 12 —', 'xiii'
    - 'Page 3', 'Hal. 7', 'pg 10', 'p. 5'
    - very short decoration-only stubs
    """
    t = s.strip()
    if not t:
        return False
    if PAGE_TAG_RE.match(t):
        return True
    if re.match(r"(?i)^(p\.|pg)\s*\d{1,5}$", t):
        return True
    if DECOR_ONLY_RE.match(t) and len(t) <= 12:
        return True
    if _mostly_digits_or_roman(t):
        return True
    return False

def _bullet_level_and_text(line: str) -> tuple[Optional[int], str]:
    """
    Return (level, content) if line starts with a bullet marker, else (None, line).
    Levels:
      1 -> ● • ▪
      2 -> ○ – -  or  N.
    """
    m = BULLET_MARK_RE.match(line)
    if not m:
        return None, line

    token = m.group(0).strip()
    content = line[m.end():].strip()

    # Map marker to level
    if token and token[0] in {"●", "•", "▪"}:
        lvl = 1
    elif token and (token[0] in {"○", "–", "-"} or re.match(r"^\d+\.$", token)):
        lvl = 2
    else:
        lvl = 1
    return lvl, content

def _push_label_to_path(path_stack: List[str], level: int, label: str) -> List[str]:
    """
    Ensure path_stack has slots up to 'level' (1-based), then set/replace
    the entry at 'level' with label. Returns a copy.
    """
    if len(path_stack) >= level:
        path_stack = path_stack[:level-1]
    while len(path_stack) < level-1:
        path_stack.append("")  # unnamed parent (rare)
    path_stack.append(label)
    return path_stack




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
