"""Gemini-assisted hierarchical chunking for OCR-derived page text.

This helper is only used for Textract-backed scanned/OCR pages. It asks the
configured Gemini chat model to convert each OCR page into the same structured
hierarchy used by manual chunking: a list of breadcrumb paths plus atomic rule
text. It does not run for the separate Gemini image/table extraction flow.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config_loader import load_config
from logging_config import get_logger

logger = get_logger(__name__)


class OCRHierarchyRule(BaseModel):
    """One OCR chunk/rule enriched with its hierarchy path."""

    path: List[str] = Field(
        default_factory=list,
        description="Hierarchy path for the rule, e.g. ['Section', 'Subsection'].",
    )
    text: str = Field(..., description="Atomic rule or paragraph text for retrieval.")


class OCRHierarchyPage(BaseModel):
    """Structured Gemini response for a single OCR page."""

    primary_language: str = Field(
        default="id",
        description="Dominant language kept from a bilingual page: 'id' or 'en'.",
    )
    rules: List[OCRHierarchyRule] = Field(
        default_factory=list,
        description="Atomic OCR chunks with inherited hierarchy paths.",
    )


DEFAULT_OCR_CHUNKING_PROMPT = textwrap.dedent(
    """
    You are a document chunking assistant for OCR output in a RAG ingestion pipeline.

    You receive exactly one OCR page at a time.
    Your job is to convert it into structured hierarchical chunks.

    Rules:
    1. Split only within this page. Never merge content from multiple pages.
    2. Return a hierarchy path for every chunk, similar to document headings, subheadings, and list context.
    3. If the page starts with a literal bullet or numbered-list marker, inherit the previous page hierarchy path when it is relevant.
    4. If the page is bilingual, keep only one language:
       - prefer Indonesian first,
       - otherwise use English,
       - discard the other language from the output.
    5. Keep wording faithful to the OCR text. Do not hallucinate or summarize beyond light cleanup.
    6. Each output item must be semantically atomic and useful for retrieval.
    7. Do not output heading-only items unless they are needed as hierarchy context for a rule.

    Previous page hierarchy path: {previous_hierarchy}

    OCR page text:
    ---
    {page_text}
    ---

    {format_instructions}
    """
).strip()


def _normalize_previous_hierarchy(previous_hierarchy: Optional[List[str]]) -> str:
    """Render the previous hierarchy as a compact prompt string."""

    if not previous_hierarchy:
        return "<none>"
    return " > ".join(str(part).strip() for part in previous_hierarchy if str(part).strip()) or "<none>"


def chunk_ocr_page_with_gemini(
    page_text: str,
    *,
    previous_hierarchy: Optional[List[str]] = None,
    profile: str = "",
) -> OCRHierarchyPage:
    """Convert one OCR page into hierarchical rules using Gemini.

    Args:
        page_text: OCR text for a single page.
        previous_hierarchy: Most recent hierarchy path from the previous page.
        profile: Optional config profile passed to :func:`load_config`.

    Returns:
        Parsed Gemini output as :class:`OCRHierarchyPage`.
    """

    config = load_config(profile)
    parser = PydanticOutputParser(pydantic_object=OCRHierarchyPage)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise OCR chunking assistant. Return only JSON that matches the provided schema.",
            ),
            (
                "human",
                DEFAULT_OCR_CHUNKING_PROMPT,
            ),
        ]
    )
    api_key = os.environ.get(getattr(config, "google_api_key_name", "GEMINI_API_KEY"))
    chat_model = ChatGoogleGenerativeAI(
        model=getattr(config, "gemini_chat_model", None),
        google_api_key=api_key,
        temperature=0,
    )
    prompt_chain = prompt.partial(
        previous_hierarchy=_normalize_previous_hierarchy(previous_hierarchy),
        format_instructions=parser.get_format_instructions(),
    )
    chain = prompt_chain | chat_model | parser
    return chain.invoke({"page_text": page_text})


def apply_gemini_ocr_chunking(
    pages: List[Dict[str, Any]],
    *,
    profile: str = "",
) -> List[Dict[str, Any]]:
    """Populate Textract OCR pages with Gemini hierarchy rules when enabled.

    The function walks pages in order so bullet-first pages can inherit the most
    recent hierarchy path discovered on a previous page. Only pages explicitly
    marked as Textract OCR output are processed, which keeps the existing
    Gemini-as-vision/table-extractor path and any non-Textract chunking
    fallbacks unchanged.
    """

    previous_hierarchy: List[str] = []
    for page in sorted(pages, key=lambda item: (item.get("page") is None, item.get("page"))):
        if not page.get("ocr_used"):
            continue
        if page.get("ocr_engine") != "textract":
            continue
        if page.get("tables") or page.get("gemini_chunks") or page.get("gemini_rules"):
            continue

        page_text = str(page.get("narrative_text") or "").strip()
        if not page_text:
            continue

        try:
            parsed = chunk_ocr_page_with_gemini(
                page_text,
                previous_hierarchy=previous_hierarchy,
                profile=profile,
            )
        except Exception as exc:  # pragma: no cover - network/model variability
            logger.warning(
                "Gemini OCR chunking failed; leaving OCR page on manual chunker fallback",
                extra={"page": page.get("page")},
                exc_info=exc,
            )
            continue

        page["gemini_rules"] = [rule.model_dump() for rule in parsed.rules if rule.text.strip()]
        page["ocr_primary_language"] = parsed.primary_language
        if parsed.rules:
            last_path = parsed.rules[-1].path
            if last_path:
                previous_hierarchy = last_path[:]

    return pages
