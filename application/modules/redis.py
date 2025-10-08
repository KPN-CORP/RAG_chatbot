"""Helpers for storing and retrieving chat history from Redis."""

import json
from typing import List, Optional, Tuple

import redis

from config import MAX_HISTORY_MESSAGES, REDIS_URL, SESSION_TTL_SECONDS
from logging_config import get_logger

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
logger = get_logger(__name__)


# ----- Helpers for Redis history -----
# Key patterns:
# "session:{session_id}:history" -> Redis list of JSON items: [{"role":"user","text":...}, {"role":"assistant","text":..., "sources":[...]}, ...]
def _history_key(session_id: str) -> str:
    """Return the canonical Redis key that stores history for *session_id*."""

    return f"session:{session_id}:history"


def append_history_item(
    session_id: str,
    role: str,
    text: str,
    sources: Optional[List[dict]] = None,
) -> None:
    """Append a single message to the Redis history list."""

    item = {"role": role, "text": text}
    if sources:
        item["sources"] = sources
    payload = json.dumps(item)
    r.rpush(_history_key(session_id), payload)
    r.expire(_history_key(session_id), SESSION_TTL_SECONDS)  # optional
    logger.debug(
        "Appended history item",
        extra={
            "session_id": session_id,
            "role": role,
            "has_sources": bool(sources),
        },
    )


def load_recent_history(session_id: str, n: int = MAX_HISTORY_MESSAGES) -> List[dict]:
    """Return up to *n* most recent history entries for *session_id*."""

    key = _history_key(session_id)
    raw = r.lrange(key, -n, -1) if r.exists(key) else []
    items = [json.loads(x) for x in raw]
    logger.debug(
        "Loaded recent history",
        extra={"session_id": session_id, "items": len(items)},
    )
    return items


# Redis-backed "memory-like" helper (mimics old ConversationBufferMemory enough for drop-in)
class RedisConversationMemory:
    """Lightweight conversation buffer that persists turns in Redis."""

    def __init__(
        self, session_id: str, max_messages: int = MAX_HISTORY_MESSAGES
    ) -> None:
        """Create a memory helper for *session_id* limited to *max_messages*."""

        self.session_id = session_id
        self.max_messages = max_messages
        logger.debug(
            "Initialised RedisConversationMemory",
            extra={"session_id": session_id, "max_messages": max_messages},
        )

    def load(self) -> List[dict]:
        """Return the raw history payload for the configured session."""

        history = load_recent_history(self.session_id, n=self.max_messages)
        logger.debug(
            "Loaded history for conversation",
            extra={"session_id": self.session_id, "items": len(history)},
        )
        return history

    def get_chat_history_pairs(self) -> List[Tuple[str, str]]:
        """Return normalized ``(user, assistant)`` message tuples for LangChain."""

        # If there are stray messages they will be handled reasonably:
        #   - user followed by assistant -> paired
        #   - user without assistant -> (user, "")
        #   - assistant without preceding user -> ("", assistant)
        recent = self.load()
        pairs = []
        i = 0
        n = len(recent)
        while i < n:
            if recent[i]["role"] == "user":
                user_text = recent[i]["text"]
                assistant_text = ""
                # find next assistant message immediately following (not skipping other users)
                if i + 1 < n and recent[i + 1]["role"] == "assistant":
                    assistant_text = recent[i + 1]["text"]
                    i += 2
                else:
                    i += 1
                pairs.append((user_text, assistant_text))
            else:
                # assistant without preceding user -> preserve it as ("", assistant)
                pairs.append(("", recent[i]["text"]))
                i += 1
        return pairs

    def add_user_message(self, text: str) -> None:
        """Persist a user utterance and refresh the session TTL."""

        append_history_item(self.session_id, "user", text)
        logger.info(
            "Stored user message",
            extra={"session_id": self.session_id, "preview": text[:100]},
        )

    def add_assistant_message(
        self,
        text: str,
        sources: Optional[List[dict]] = None,
    ) -> None:
        """Persist an assistant message optionally enriched with sources."""

        append_history_item(self.session_id, "assistant", text, sources=sources)
        logger.info(
            "Stored assistant message",
            extra={
                "session_id": self.session_id,
                "preview": text[:100],
                "has_sources": bool(sources),
            },
        )
