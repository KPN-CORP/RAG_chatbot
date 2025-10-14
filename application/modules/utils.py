"""Generic string parsing helpers shared across ingestion utilities."""

import json
import re

from logging_config import get_logger
from typing import Optional, Dict, Any
# from typing import List


logger = get_logger(__name__)


def extract_json_string(s: str) -> Optional[str]:
    """Return a JSON string extracted from s, or None if not found/invalid."""
    if s is None:
        return None

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
        json.loads(candidate)
        return candidate
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
