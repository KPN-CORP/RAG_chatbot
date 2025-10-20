import re

# Bullet tokens we commonly see; extend if your docs use others
BULLET_TOKENS = {"●", "•", "▪", "○", "-", "–", "—"}
NUM_BULLET_RE = re.compile(r"^\d+\.$")  # "1." / "23." style bullets

# Page-number patterns in many languages (broad but not brittle)
PAGE_TAG_RE = re.compile(r"(?i)^(page|hal(\.|aman)?|pg)\s*\d{1,5}$")
ROMAN_RE = re.compile(r"(?i)^(?=[MDCLXVI]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _strip_bullet(text: str):
    """
    Remove a leading bullet glyph or numeric '1.' and return (content, had_bullet).
    Keeps text otherwise untouched.
    """
    t = text.strip()
    if t and t[0] in BULLET_TOKENS:
        return t[1:].lstrip(), True
    first = t.split(" ", 1)[0]
    if NUM_BULLET_RE.match(first):
        return re.sub(r"^\d+\.\s*", "", t), True
    return t, False

def _same_line(a, b, tol=2.5):
    return abs(float(a["top"]) - float(b["top"])) <= tol

def _mostly_digits_or_roman(s: str) -> bool:
    """
    True if line is short and composed mostly of digits, separators, or roman numerals.
    Robust to '— 12 —', '12', 'xii', '| 34 |', etc.
    """
    t = s.strip()
    if not t:
        return False
    # Remove common decoration chars
    core = re.sub(r"[\s\-\—\–\|_\.·•●○]+", "", t)
    if not core:
        return False
    # Very short, likely a number token
    if len(core) <= 6:
        # all digits?
        if core.isdigit():
            return True
        # roman numeral?
        if ROMAN_RE.match(core):
            return True
    return False

def _looks_like_page_marker(s: str) -> bool:
    """
    True if the text explicitly looks like a page marker ('Page 12', 'Hal. 5', etc.)
    or a short numeric/roman token with decorations.
    """
    t = s.strip()
    if not t:
        return False
    if PAGE_TAG_RE.match(t):
        return True
    # Common forms like 'p. 3', 'pg 10'
    if re.match(r"(?i)^(p\.|pg)\s*\d{1,5}$", t):
        return True
    # Plain or decorated number/roman (— 12 —, xii)
    if _mostly_digits_or_roman(t):
        return True
    return False

def _is_probable_header_footer(line_text: str, top: float, page_height: float) -> bool:
    """
    Heuristic: short page markers near the very top/bottom bands are likely headers/footers.
    Bands are ~7% of page height by default.
    """
    band = max(24.0, page_height * 0.07)  # at least 24px, or 7% of page height
    near_top = top <= band
    near_bottom = top >= (page_height - band)
    if not (near_top or near_bottom):
        return False

    # short and looks like page marker or mostly punctuation/digits
    t = line_text.strip()
    if len(t) <= 20 and _looks_like_page_marker(t):
        return True

    # also filter very short decoration-only lines in the bands: '— — —', '.....'
    if len(t) <= 12 and not re.search(r"[A-Za-z]", t) and re.search(r"[-—–_.•○●|]+", t):
        return True

    return False