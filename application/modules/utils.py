"""Generic string parsing helpers shared across ingestion utilities."""

import re
import json
# from typing import List

from logging_config import get_logger

logger = get_logger(__name__)


def extract_and_convert_dictionary_string(s: str) -> dict | None:
    """Extract JSON-like content from *s* and return it as a dictionary if possible."""
    # 1. Extraction/Filtering using regex
    # Pattern: \{ (literal '{') .* (any characters) \} (literal '}')
    match = re.search(r'\{.*\}', s, re.DOTALL)

    if not match:
        logger.warning("Could not find content enclosed in braces", extra={"value": s})
        return None

    dict_string = match.group(0)

    # 2. Conversion
    try:
        data = json.loads(dict_string)
        if isinstance(data, dict):
            return data
        else:
            logger.warning(
                "Extracted JSON was not a dictionary",
                extra={"value_type": type(data).__name__},
            )
            return None
    except json.JSONDecodeError as e:
        logger.error("Extracted content is not valid JSON", extra={"error": str(e)})
        return None


