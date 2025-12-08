"""Utility helpers supporting deterministic document chunking."""

from __future__ import annotations

import hashlib

# from dataclasses import dataclass
import json
import re
from typing import List, Any, Dict, Optional, Sequence, Tuple
from logging_config import get_logger


logger = get_logger(__name__)

# -----------------------------
# Text normalization / heuristics
# -----------------------------

# HDR_RE = re.compile(r"^[A-Z0-9 ./-]{6,}$")          # heuristic ALL-CAPS headers
HDR_RE = re.compile(r"^[A-Z0-9 ()/.,&\-’'–—:]{6,}$")

# very conservative: standalone 1–4 digits, not part of decimal/percent/ID
TRAIL_NUM_RE = re.compile(
    r"""
    (?<![A-Za-z0-9%/\-])     # not preceded by alnum, %, /, -
    (?:—\s*|–\s*|–\s*)?      # optional dash
    (\d{1,4})                # 1–4 digits (capture if you want to compare to page)
    (?<!\.\d)                # not decimal tail like ".5"
    \s*$                     # end of line
    """,
    re.VERBOSE
)

_WHITELIST_EO_LINE = re.compile(
    r"(Pasal|Bab|Tahun|Golongan|No\.?|Level|ISO|UU|PP|PerMen|Perdir|Q\d|%)\s*$",
    re.IGNORECASE,
)


# Bullet markers (unchanged, extend if you meet new glyphs)
TIME_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?P<hour>\d{1,2})
    [\.:]
    (?P<minute>\d{2})
    (?:\.?)(?=\s|$|[,;:])
    """,
    re.VERBOSE,
)

TIME_TOKEN_RE = re.compile(
    r"""
    ^\s*
    (?P<hour>\d{1,2})
    [\.:]
    (?P<minute>\d{2})
    (?:\.?)(?=\s*$)
    """,
    re.VERBOSE,
)


BULLET_MARK_RE = re.compile(r"^(?:[•●\-–▪]|○|\d+\.)\s*")

# Page-marker patterns in multiple languages (very permissive but short)
PAGE_TAG_RE = re.compile(r"(?i)^(page|hal(\.|aman)?|pg)\s*\d{1,5}$")
ROMAN_RE = re.compile(
    r"(?i)^(?=[MDCLXVI]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
DECOR_ONLY_RE = re.compile(r"^[\s\-\—\–_.•○●|·]+$")

# --- Sentence boundary helper (universal) ---
_END_SENTENCE_RE = re.compile(r'([.!?…])["”\']?\s*$')

def _ends_sentence(s: str) -> bool:
    """
    True if 's' looks like an end of sentence. Conservative: only ., !, ?, …
    Avoids false positives on abbreviations by requiring end-of-line punctuation.
    """
    if not s:
        return False
    s = s.strip()
    return bool(_END_SENTENCE_RE.search(s))


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
    stripped = line.strip()

    # Reject standalone time-like tokens (e.g., "14.00" or "08:30") to avoid
    # promoting clock values to headers.
    m = TIME_TOKEN_RE.match(stripped)
    if m:
        try:
            hour = int(m.group("hour"))
            minute = int(m.group("minute"))
        except (TypeError, ValueError):
            hour = minute = -1
        if 0 <= hour < 24 and 0 <= minute < 60:
            return False

    return bool(HDR_RE.match(stripped)) and (
        " " in stripped or "-" in stripped or "/" in stripped
    )


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
    stripped = line.lstrip()
    logger.debug(f"Checking line for bullet marker: {stripped}")
    # Guard against time-like prefixes (e.g. "14.00" or "08:30") being misread as
    # numbered bullets.  When detected, short-circuit and treat the line as plain
    # text.  Only accept plausible clock times where 0 <= hour < 24 and
    # 0 <= minute < 60.
    m_time = TIME_PREFIX_RE.match(stripped)
    if m_time:
        logger.debug(f"Detected time-like prefix: {m_time.group(0)} in line: {line}")
        try:
            hour = int(m_time.group("hour"))
            minute = int(m_time.group("minute"))
        except (TypeError, ValueError):
            hour = minute = -1
        if 0 <= hour < 24 and 0 <= minute < 60:
            return None, line

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


def should_strip_trailing_number(
    line: str,
    y_top: float,
    page_height: float,
    page_no: int | None = None,
    captured_token: str | None = None,
    top_bottom_band_ratio: float = 0.10,
) -> bool:
    """
    Return True if `line` likely ends with a pagination crumb that should be stripped.
    Uses position (top/bottom bands), whitelist guard, and optional page_no check.
    """
    if not line or page_height <= 0:
        return False

    # 1) Only consider top/bottom bands
    in_band = (y_top < top_bottom_band_ratio * page_height) or (
        y_top > (1.0 - top_bottom_band_ratio) * page_height
    )
    if not in_band:
        return False

    # 2) Do not strip if end-of-line has typical legal/HR tokens
    if _WHITELIST_EO_LINE.search(line):
        return False

    # 3) If we know page number, keep only if close to it
    if page_no is not None and captured_token is not None:
        try:
            if abs(int(captured_token) - int(page_no)) > 1:
                return False
        except ValueError:
            return False

    return True


def boundary_join(prev_text: str, next_text: str, prev_meta: dict) -> str:
    """
    Join prev_text and next_text with a space, but if prev_meta indicates
    a header/footer-band trailing number, drop that number before joining.
    """
    if prev_meta and prev_meta.get("in_band") and prev_meta.get("trail_num_token"):
        # remove only the trailing token at end-of-line
        prev_text = TRAIL_NUM_RE.sub("", prev_text).rstrip()
    return (prev_text + " " + next_text).strip()

# --- Indentation helpers for bullet levels ---------------------------------


def _cluster_indent_buckets(xs: Sequence[float], gap: float = 12.0) -> List[Tuple[float, float]]:
    """
    Given a sequence of x positions (e.g., hanging_x), return a list of buckets
    as [(min_x, max_x), ...], where a new bucket starts when the gap to the
    previous x exceeds `gap`. This is a simple 1D clustering that works well
    for indent levels.
    """
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return []
    buckets: List[Tuple[float, float]] = [(vals[0], vals[0])]
    for x in vals[1:]:
        lo, hi = buckets[-1]
        if x - hi > gap:
            buckets.append((x, x))
        else:
            buckets[-1] = (lo, max(hi, x))
    return buckets


def _indent_level_for_x(x: float, buckets: Sequence[Tuple[float, float]]) -> Optional[int]:
    """
    Map an x position to a 1-based level according to `buckets`.
    Returns None if no buckets exist.
    """
    if not buckets:
        return None
    for i, (lo, hi) in enumerate(buckets, start=1):
        if lo - 1e-6 <= x <= hi + 1e-6:
            return i
    # if x is left of the first bucket, treat as level 1; if far right, last
    if x < buckets[0][0]:
        return 1
    return len(buckets)


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
