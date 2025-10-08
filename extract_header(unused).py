import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import pdfplumber


# ----------------- Helpers -----------------
def bbox_intersects(
    b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]
) -> bool:
    # b: (x0, y0/top, x1, y1/bottom)
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2
    horiz = (x0a < x1b) and (x1a > x0b)
    vert = (y0a < y1b) and (y1a > y0b)
    return horiz and vert


_NOTE_HEADERS_RE = re.compile(r"^\s*(CATATAN|CATATAN:|NOTE|NOTES)\s*$", re.IGNORECASE)
_HEADING_UPPER_RE = re.compile(r"^[A-Z0-9 \-\/]{1,120}$")


def detect_page_header_candidate(
    words: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    From extract_words() items, group into near-horizontal lines and return
    a header candidate (text) if present and the produced sorted_lines list.
    sorted_lines: list of dicts {y, text, font_size, words}
    """
    if not words:
        return None, []

    # group by top coordinate (rounded)
    lines = {}
    for w in words:
        # 'top' is y0; use rounded bucket to group near-line words
        top = round(float(w.get("top", 0)) / 3.0) * 3.0
        lines.setdefault(top, []).append(w)

    sorted_lines = []
    for y in sorted(lines.keys()):
        words_on_line = sorted(lines[y], key=lambda x: float(x.get("x0", 0)))
        text = " ".join(w.get("text", "") for w in words_on_line).strip()
        # average font size (if available)
        sizes = [float(w.get("size", 0)) for w in words_on_line if w.get("size")]
        avg_font = (sum(sizes) / len(sizes)) if sizes else None
        sorted_lines.append({"y": y, "text": text, "font_size": avg_font, "words": words_on_line})

    # top candidates: first few physical lines
    candidate = None
    top_font = -1
    for ln in sorted_lines[:6]:
        txt = ln["text"]
        if not txt or re.match(r"^\d+\s*$", txt):  # skip page numbers
            continue
        # prefer larger font size if given, else prefer uppercase short lines
        fs = ln.get("font_size") or 0
        uc_ratio = sum(1 for c in txt if c.isupper()) / max(1, len(txt))
        score = fs + (uc_ratio * 2)
        if score > top_font:
            top_font = score
            candidate = ln

    header_text = None
    if candidate:
        txt = candidate["text"]
        # further sanity checks: not a long paragraph and has uppercase ratio or font size evidence
        uc_ratio = sum(1 for c in txt if c.isupper()) / max(1, len(txt))
        if (candidate.get("font_size") and candidate["font_size"] > 8) or uc_ratio > 0.3:
            header_text = txt

    return header_text, sorted_lines


def extract_notes_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Given page text (multi-line), find 'CATATAN' headers and extract following lines until blank or next header-like line.
    Returns list of {header, note_text, start_line, end_line}
    """
    notes = []
    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        ln = lines[i].strip()
        if _NOTE_HEADERS_RE.match(ln):
            header = ln
            j = i + 1
            collected = []
            while j < n:
                nxt = lines[j].rstrip()
                if not nxt.strip():
                    break
                # stop if next is a header-like ALLCAPS short line
                if (
                    len(nxt.strip()) <= 120
                    and _HEADING_UPPER_RE.match(nxt.strip())
                    and sum(1 for c in nxt if c.isupper()) / max(1, len(nxt.strip())) > 0.4
                ):
                    break
                collected.append(nxt)
                j += 1
            note_text = "\n".join(collected).strip()
            notes.append(
                {
                    "header": header,
                    "note_text": note_text,
                    "start_line": i,
                    "end_line": j,
                }
            )
            i = j
        else:
            i += 1
    return notes


# ----------------- Combined reader -----------------
def get_page_elements_with_header_notes(
    pdf_path: str, strip_repeated_headers: bool = True
) -> List[Dict[str, Any]]:
    """
    Returns pages_output: list of dicts per page:
      {
        "page": pidx,
        "narrative_text": narrative_text,      # text with table words removed (and headers possibly stripped)
        "tables": [ {table_index, bbox, table_dct}, ... ],
        "header_text": header_text_or_None,    # candidate header for this page
        "header_repeated": bool,
        "notes": [ {header, note_text, start_line, end_line}, ... ],
        "lines": sorted_lines                   # lines with text+font info (useful for debugging/provenance)
      }
    """
    # First pass: collect per-page words, table bboxes/rows, header candidate + sorted_lines, and raw narrative lines
    per_page_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            # table detection (same as your code)
            table_bboxes = []
            tables = []
            try:
                table_objs = page.find_tables() or []
            except Exception:
                table_objs = []
            for t_idx, t in enumerate(table_objs):
                tbbox = t.bbox  # (x0, top, x1, bottom)
                table_bboxes.append({"bbox": tbbox, "table_index": t_idx})
                try:
                    rows = t.extract()
                except Exception:
                    rows = None
                tables.append({"table_index": t_idx, "bbox": tbbox, "rows": rows})

            # extract words and keep only those not inside any table bbox
            words = (
                page.extract_words(
                    extra_attrs=["fontname", "size", "adv", "top", "bottom", "x0", "x1"]
                )
                or []
            )
            narrative_words = []
            for w in words:
                wbox = (
                    float(w.get("x0", 0)),
                    float(w.get("top", 0)),
                    float(w.get("x1", 0)),
                    float(w.get("bottom", 0)),
                )
                in_table = any(bbox_intersects(wbox, tb["bbox"]) for tb in table_bboxes)
                if not in_table:
                    narrative_words.append(w)

            # build lines grouped by top coordinate (tolerance)
            lines_map = {}
            tol = 3.0
            for w in narrative_words:
                top = round(float(w.get("top", 0)) / tol) * tol
                lines_map.setdefault(top, []).append((float(w.get("x0", 0)), w))

            sorted_lines = []
            for y in sorted(lines_map.keys()):
                words_on_line = [t[1] for t in sorted(lines_map[y], key=lambda x: x[0])]
                text = " ".join(w.get("text", "") for w in words_on_line).strip()
                sizes = [float(w.get("size", 0)) for w in words_on_line if w.get("size")]
                avg_font = (sum(sizes) / len(sizes)) if sizes else None
                sorted_lines.append(
                    {
                        "y": y,
                        "text": text,
                        "font_size": avg_font,
                        "words": words_on_line,
                    }
                )

            # reconstruct narrative_text (raw; header stripping happens later optionally)
            narrative_text = "\n".join([ln["text"] for ln in sorted_lines]).strip()

            # detect header candidate from narrative_words (use positional/font info)
            header_candidate, header_lines = detect_page_header_candidate(narrative_words)

            # extract notes using the raw narrative_text
            notes = extract_notes_from_text(narrative_text)

            per_page_data.append(
                {
                    "page": pidx,
                    "tables": tables,
                    "table_bboxes": table_bboxes,
                    "words": narrative_words,
                    "sorted_lines": sorted_lines,
                    "raw_narrative": narrative_text,
                    "header_candidate": header_candidate,
                    "notes": notes,
                }
            )

    # Second pass: detect repeated headers across pages
    header_counts = Counter([p["header_candidate"] for p in per_page_data if p["header_candidate"]])
    repeated_headers = set()
    total_pages = len(per_page_data)
    if strip_repeated_headers and total_pages > 0:
        # threshold: appears on more than 50% pages or at least on 3 pages
        for h, cnt in header_counts.items():
            if cnt >= max(3, int(total_pages * 0.5)):
                repeated_headers.add(h)

    # Build final pages_output, optionally stripping repeated headers from narrative text and marking header_repeated flag
    pages_output = []
    for pdata in per_page_data:
        pnum = pdata["page"]
        raw = pdata["raw_narrative"]
        header = pdata["header_candidate"]
        header_repeated = header in repeated_headers if header else False

        # If header is repeated -> remove lines equal to header (exact match) from top of narrative
        narrative_lines = raw.splitlines()
        if header_repeated:
            # remove leading lines that exactly equal the header text (some PDFs might have slight spacing)
            new_lines = []
            header_norm = header.strip() if header else None
            # removed_any = False
            for idx, ln in enumerate(narrative_lines):
                if idx < 6 and header_norm and ln.strip() and ln.strip() == header_norm:
                    # removed_any = True
                    continue
                new_lines.append(ln)
            narrative_text = "\n".join(new_lines).strip()
        else:
            narrative_text = raw

        # attach the tables as you did, convert to table_records with table_dct preserved
        table_records = []
        for t in pdata["tables"]:
            table_dct = t.get("rows") if t.get("rows") else None
            table_records.append(
                {
                    "table_index": t["table_index"],
                    "bbox": t["bbox"],
                    "table_dct": table_dct,
                }
            )

        pages_output.append(
            {
                "page": pnum,
                "narrative_text": narrative_text,
                "tables": table_records,
                "header_text": header,
                "header_repeated": header_repeated,
                "notes": pdata["notes"],
                "lines": pdata["sorted_lines"],  # optional: useful for provenance/debugging
            }
        )

    return pages_output
