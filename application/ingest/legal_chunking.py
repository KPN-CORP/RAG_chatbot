"""Article-level (``Pasal``) chunking for Indonesian legal documents.

Laws and regulations (UU, PP, Permen, Peraturan Perusahaan, PKB) are organised
as ``BAB > Bagian > Paragraf > Pasal > ayat (n) > huruf a.``. The generic
line/semantic chunker splits such documents into tiny fragments, especially
for scanned PDFs where Textract emits list markers (``a.``), ayat markers
(``(1)``) and even the article number (``Pasal`` / ``230``) as separate lines.
A question like "apa saja larangan PHK" then retrieves only the stub
"Pemberi Kerja dilarang melakukan Pemutusan Hubungan" without the list.

This module rebuilds the document as a continuous stream of lines, drops page
furniture (running headers, SK numbers, page numbers), repairs common OCR
mistakes in structural markers and emits ONE chunk per ``Pasal`` (split at
ayat boundaries only when the article is very long). Each chunk carries a
breadcrumb such as ``UU Nomor 27 Tahun 2026 > BAB V Pemutusan Hubungan Kerja >
Pasal 230`` so both retrieval and the answering LLM know the exact legal basis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modules.chunking_utils import _checksum

__all__ = [
    "build_legal_chunks",
    "count_pasal_headings",
    "is_legal_document",
]

# OCR frequently confuses digits with letters inside article numbers
# ("Pasal 23O", "Pasal 1O", "(l)").
_OCR_DIGIT_MAP = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})
_NUM_TOKEN = r"[0-9OoIl|]{1,4}"

# "Pasal 230", "Pasal23O", "Pasal 17..", "Pasal 230A" — the whole line, so
# in-text references like "Pasal 20 ayat (1) wajib ..." are not headings.
PASAL_HEAD_RE = re.compile(rf"^Pasal\s*({_NUM_TOKEN})\s*([A-Z])?\s*[.\s]*$")
PASAL_BARE_RE = re.compile(r"^Pasal\s*[.:]?\s*$")
NUM_ONLY_RE = re.compile(rf"^({_NUM_TOKEN})\s*([A-Z])?\s*[.\s]*$")
BAB_RE = re.compile(r"^BAB\s*([IVXLC]+|\d+)\b\.?\s*(.*)$")
BAGIAN_RE = re.compile(r"^Bagian\s*(Ke[a-z]+|Pertama)\b\s*(.*)$")
PARAGRAF_RE = re.compile(rf"^Paragraf\s*({_NUM_TOKEN})\b\.?\s*(.*)$")
PENJELASAN_RE = re.compile(r"^PENJELASAN\b")

# Page furniture: running headers/footers that OCR repeats on every page.
_FURNITURE_RES = [
    re.compile(r"^PRESIDEN\W*$"),
    # "REPUBLIK INDONESIA" and OCR variants ("Refublik Indonesia", "Refubuk Indonesi,a").
    re.compile(r"^r\w{4,8}\s*indone\w{0,4}\W*\w?$", re.IGNORECASE),
    re.compile(r"^SK\s*No\.?\s*\S+(\s*[A-Z])?$"),
    re.compile(r"^[-–—\s]*\d{1,4}[-–—\s]*$"),  # page numbers "- 95 -"
]

# Line starters that must begin a new physical line in the chunk body.
_AYAT_START_RE = re.compile(r"^\(\d{1,2}\)")
_ITEM_START_RE = re.compile(r"^([a-z]|\d{1,2})[.)]\s")
_MARKER_ONLY_RE = re.compile(r"^(\(\d{1,2}\)|[a-z][.)]|\d{1,2}[.)])$")
_JOIN_BREAK_RE = re.compile(r"[:;.]$")


def _fix_ocr_number(token: str) -> str:
    return token.translate(_OCR_DIGIT_MAP)


def _normalize_ayat_marker(line: str) -> str:
    """Repair OCR'd ayat markers: "(l) ..." → "(1) ...", "(21 Pemutusan" → "(2) Pemutusan"."""

    m = re.match(r"^\(([0-9OoIl|]{1,2})\)\s*", line)
    if m:
        return f"({_fix_ocr_number(m.group(1))}) " + line[m.end():]
    # A missing closing paren read as "1"/"l": "(21 Text" → "(2) Text".
    m = re.match(r"^\(([0-9])[1l]\s+(?=\S)", line)
    if m:
        return f"({m.group(1)}) " + line[m.end():]
    return line


def _is_furniture(line: str) -> bool:
    return any(rx.match(line) for rx in _FURNITURE_RES)


def _iter_lines(pages: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Optional[int], str]]:
    for pg in sorted(pages, key=lambda p: (p.get("page") is None, p.get("page"))):
        text = str(pg.get("narrative_text") or "")
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if line:
                yield pg.get("page"), line


def count_pasal_headings(pages: Iterable[Dict[str, Any]]) -> int:
    """Count ``Pasal N`` heading lines (including "Pasal" + number on the next line)."""

    return sum(
        1 for _, line in _iter_lines(pages) if PASAL_HEAD_RE.match(line) or PASAL_BARE_RE.match(line)
    )


def is_legal_document(pages: Iterable[Dict[str, Any]], min_articles: int = 5) -> bool:
    """True when the narrative looks like an article-structured legal document."""

    return count_pasal_headings(pages) >= min_articles


def _append_block(out: List[str], line: str) -> None:
    """Append a new ayat/item line, dropping a page-bottom catchword it repeats.

    Printed Indonesian regulations repeat the first words of the next page at the
    bottom of each page (e.g. "j. dalam" then, on the next page, "j. dalam keadaan ...").
    """

    if out and len(out[-1]) <= 60 and line != out[-1] and line.startswith(out[-1]):
        out[-1] = line
    else:
        out.append(line)


def _split_sentences(line: str, max_chars: int) -> List[str]:
    """Split a single over-long line at sentence (or clause) boundaries."""

    if len(line) <= max_chars:
        return [line]
    parts: List[str] = []
    buf = ""
    for sent in re.split(r"(?<=[.;])\s+", line):
        if buf and len(buf) + len(sent) + 1 > max_chars:
            parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def _join_body(lines: List[str]) -> str:
    """Join OCR lines into readable text, keeping one line per ayat / list item."""

    out: List[str] = []
    pending_marker: Optional[str] = None
    for line in lines:
        line = _normalize_ayat_marker(line).strip()
        if _MARKER_ONLY_RE.match(line):
            if pending_marker:
                out.append(pending_marker)
            pending_marker = _normalize_ayat_marker(line).strip()
            continue
        if pending_marker:
            line = f"{pending_marker} {line}"
            pending_marker = None
            _append_block(out, line)
            continue
        starts_block = bool(_AYAT_START_RE.match(line) or _ITEM_START_RE.match(line))
        if not out or starts_block:
            _append_block(out, line)
        elif out[-1].endswith("-") and not out[-1].endswith(" -"):
            out[-1] = out[-1] + line  # hyphenated word wrap: "terus-" + "menerus"
        else:
            out[-1] = f"{out[-1]} {line}"
    if pending_marker:
        out.append(pending_marker)
    return "\n".join(out).strip()


def _split_long(body: str, max_chars: int) -> List[str]:
    """Split an over-long article at ayat/item line boundaries."""

    if len(body) <= max_chars:
        return [body]
    pieces: List[str] = []
    buf: List[str] = []
    size = 0
    lines = [part for raw in body.split("\n") for part in _split_sentences(raw, max_chars)]
    for line in lines:
        # Prefer breaking right before a new ayat once the buffer is non-trivial.
        if buf and (size + len(line) > max_chars or (_AYAT_START_RE.match(line) and size > max_chars * 0.6)):
            pieces.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        pieces.append("\n".join(buf))
    return pieces


def _doc_title(source: str) -> str:
    stem = Path(source or "").stem
    stem = re.sub(r"\s*-\s*Copy(\s*\(\d+\))?$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[_]+", " ", stem).strip()


def build_legal_chunks(
    pages: List[Dict[str, Any]],
    doc_id: str,
    source: str,
    max_chars: int = 2500,
) -> List[Dict[str, Any]]:
    """Build one chunk per ``Pasal`` (plus preamble/heading-only sections).

    Args:
        pages: Page dicts with ``page`` and ``narrative_text``.
        doc_id: Document id used in chunk ids.
        source: Source filename/path (also used as the breadcrumb root).
        max_chars: Articles longer than this are split at ayat boundaries.
    """

    title = _doc_title(source)
    bab = bab_title = bagian = paragraf = None
    in_penjelasan = False
    pending_title: Optional[str] = None  # which heading still expects a title line
    awaiting_pasal_number = False

    units: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"pasal": None, "page": None, "lines": [], "path": []}

    def context_path() -> List[str]:
        path: List[str] = []
        if in_penjelasan:
            path.append("Penjelasan")
        if bab:
            path.append(f"{bab} {bab_title}".strip() if bab_title else bab)
        if bagian:
            path.append(bagian)
        if paragraf:
            path.append(paragraf)
        return path

    def start_unit(pasal: Optional[str], page: Optional[int]) -> None:
        nonlocal current
        if current["lines"]:
            units.append(current)
        current = {"pasal": pasal, "page": page, "lines": [], "path": context_path()}

    for page, line in _iter_lines(pages):
        if _is_furniture(line) and not (awaiting_pasal_number and NUM_ONLY_RE.match(line)):
            continue

        if awaiting_pasal_number:
            awaiting_pasal_number = False
            m = NUM_ONLY_RE.match(line)
            if m:
                current["pasal"] = _fix_ocr_number(m.group(1)) + (m.group(2) or "")
                continue

        m = PASAL_HEAD_RE.match(line)
        if m:
            pending_title = None
            start_unit(_fix_ocr_number(m.group(1)) + (m.group(2) or ""), page)
            continue
        if PASAL_BARE_RE.match(line):
            pending_title = None
            start_unit(None, page)
            current["pasal"] = "?"
            awaiting_pasal_number = True
            continue

        if PENJELASAN_RE.match(line):
            in_penjelasan = True
            bab = bab_title = bagian = paragraf = None
            pending_title = None
            start_unit(None, page)
            current["lines"].append(line)
            continue

        m = BAB_RE.match(line)
        if m:
            bab, bab_title = f"BAB {m.group(1)}", (m.group(2).strip() or None)
            bagian = paragraf = None
            pending_title = None if bab_title else "bab"
            start_unit(None, page)
            continue
        m = BAGIAN_RE.match(line)
        if m:
            bagian = f"Bagian {m.group(1)} {m.group(2)}".strip()
            paragraf = None
            pending_title = None if m.group(2).strip() else "bagian"
            start_unit(None, page)
            continue
        m = PARAGRAF_RE.match(line)
        if m:
            paragraf = f"Paragraf {_fix_ocr_number(m.group(1))} {m.group(2)}".strip()
            pending_title = None if m.group(2).strip() else "paragraf"
            start_unit(None, page)
            continue

        if pending_title and len(line) <= 120 and not _AYAT_START_RE.match(line):
            if pending_title == "bab":
                bab_title = line.title() if line.isupper() else line
            elif pending_title == "bagian":
                bagian = f"{bagian} {line}"
            else:
                paragraf = f"{paragraf} {line}"
            pending_title = None
            current["path"] = context_path()
            continue
        pending_title = None

        if current["page"] is None:
            current["page"] = page
        current["lines"].append(line)

    if current["lines"]:
        units.append(current)

    chunks: List[Dict[str, Any]] = []
    for idx, unit in enumerate(units, 1):
        body = _join_body(unit["lines"])
        if not body:
            continue
        pasal = unit["pasal"]
        head = f"Pasal {pasal}" if pasal and pasal != "?" else None
        path = ([title] if title else []) + unit["path"] + ([head] if head else [])
        breadcrumb = " > ".join(path)
        pieces = _split_long(body, max_chars)
        for part_no, piece in enumerate(pieces, 1):
            label = head or ""
            if head and len(pieces) > 1 and part_no > 1:
                label = f"{head} (lanjutan)"
            text_body = f"{label}\n{piece}" if label else piece
            text = f"{breadcrumb} — {text_body}" if breadcrumb else text_body
            page = unit["page"]
            cid = f"{doc_id}::pasal::{page or 0:04d}::{idx:04d}::{part_no:02d}"
            chunks.append(
                {
                    "chunk_id": cid,
                    "chunk_text": text,
                    "checksum": _checksum(
                        text, source=source, page=page, extra={"kind": "legal_article", "i": idx, "p": part_no}
                    ),
                    "source": source,
                    "page": page,
                    "element_type": "legal_article",
                    "breadcrumb": breadcrumb,
                    "list_id": None,
                    "table_id": None,
                    "row_id": None,
                    "col_name": None,
                }
            )
    return chunks
