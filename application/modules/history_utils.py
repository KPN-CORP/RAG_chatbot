"""Utilities for normalising chat history structures before LLM calls."""

import json
from typing import Iterable, List, Sequence

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from logging_config import get_logger

logger = get_logger(__name__)


def build_chat_history_messages(
    raw: Iterable | Sequence | str | None,
) -> List[BaseMessage]:
    """Normalise various history encodings into a list of LangChain messages."""

    # Supported input:
    #   - List of (user_text, assistant_text) tuples/lists
    #   - List of dicts like {"role":"user","content":"..."}
    #   - Already a list of BaseMessage (returned unchanged)
    #   - If a string is passed, try to JSON-decode to list; otherwise ignore and warn
    msgs = []
    if not raw:
        # no history
        return msgs

    # If already list of BaseMessage, return as-is (safety check)
    if isinstance(raw, list) and all(isinstance(x, BaseMessage) for x in raw):
        return raw

    # If raw is a JSON string representing history, try to parse it
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed
        except Exception:
            logger.warning("chat_history is a plain string and not valid JSON. Ignoring.")
            return []

    # Now raw should be a list (hopefully)
    if isinstance(raw, list):
        for item in raw:
            # tuple/list pair: (user, assistant)
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                user_text = item[0]
                assistant_text = item[1] if len(item) > 1 else None
                if user_text:
                    msgs.append(HumanMessage(content=str(user_text)))
                if assistant_text:
                    msgs.append(AIMessage(content=str(assistant_text)))
                continue

            # dict form: {"role": "user"/"assistant", "content": "..."}
            if isinstance(item, dict) and "role" in item and "content" in item:
                role = item.get("role")
                content = item.get("content", "")
                if role and role.lower().startswith("user"):
                    msgs.append(HumanMessage(content=str(content)))
                elif role and role.lower().startswith("assistant"):
                    msgs.append(AIMessage(content=str(content)))
                elif role and role.lower().startswith("system"):
                    msgs.append(SystemMessage(content=str(content)))
                else:
                    # default to human
                    msgs.append(HumanMessage(content=str(content)))
                continue

            # Already a BaseMessage object?
            if isinstance(item, BaseMessage):
                msgs.append(item)
                continue

            # fallback: item is some raw string
            if isinstance(item, str):
                msgs.append(HumanMessage(content=item))
                continue

            # unknown type: log and skip
            logger.warning("Unknown chat_history item type: %s — skipping", type(item))
    else:
        logger.warning("chat_history is not a list; type=%s", type(raw))

    # Defensive normalization & debug logging — ensure it's a list of BaseMessage
    chat_history_messages = msgs
    if not isinstance(chat_history_messages, list):
        logger.warning(
            "chat_history_messages is not a list (type=%s). Forcing to list.",
            type(chat_history_messages),
        )
        chat_history_messages = [HumanMessage(content=str(chat_history_messages))]

    # Ensure every item is a BaseMessage — if not, coerce to HumanMessage
    fixed_msgs = []
    for i, item in enumerate(chat_history_messages):
        if isinstance(item, BaseMessage):
            fixed_msgs.append(item)
        else:
            logger.warning(
                "chat_history item %s is not BaseMessage (type=%s). Converting to HumanMessage.",
                i,
                type(item),
            )
            fixed_msgs.append(HumanMessage(content=str(item)))

    chat_history_messages = fixed_msgs

    # Debug print: types and short content preview
    logger.debug(
        "Passing chat_history_messages to conv_chain (len=%d)",
        len(chat_history_messages),
    )
    for i, m in enumerate(chat_history_messages):
        # content may not exist on some custom objects, use getattr safely
        content = getattr(m, "content", None)
        logger.debug("  [%d] %s — content preview: %s", i, type(m).__name__, repr(content)[:200])

    return chat_history_messages
