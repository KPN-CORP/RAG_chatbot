from typing import List, Optional, Tuple

import json
import redis
from logging_config import get_logger

logger = get_logger(__name__)


def build_redis_client(redis_url: str) -> redis.Redis:
    """Create a Redis client for the supplied URL."""

    return redis.Redis.from_url(redis_url, decode_responses=True)


def _history_key(session_id: str) -> str:
    """Return the canonical Redis key that stores history for *session_id*."""

    return f"session:{session_id}:history"


def append_history_item(
    redis_client: redis.Redis,
    session_ttl_seconds: int,
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
    redis_client.rpush(_history_key(session_id), payload)
    redis_client.expire(_history_key(session_id), session_ttl_seconds)
    logger.debug(
        "Appended history item",
        extra={
            "session_id": session_id,
            "role": role,
            "has_sources": bool(sources),
        },
    )


def load_recent_history(
    redis_client: redis.Redis, session_id: str, n: int
) -> List[dict]:
    """Return up to *n* most recent history entries for *session_id*."""

    key = _history_key(session_id)
    raw = redis_client.lrange(key, -n, -1) if redis_client.exists(key) else []
    items = [json.loads(x) for x in raw]
    logger.debug(
        "Loaded recent history",
        extra={"session_id": session_id, "items": len(items)},
    )
    return items


def session_exists(redis_client: redis.Redis, session_id: str) -> bool:
    """Check whether a chat session already exists in Redis."""

    return redis_client.exists(f"session:{session_id}:meta") == 1


class RedisConversationMemory:
    """Lightweight conversation buffer that persists turns in Redis."""

    def __init__(
        self,
        session_id: str,
        redis_client: redis.Redis,
        max_messages: int,
        session_ttl_seconds: int,
    ) -> None:
        """Create a memory helper for *session_id* limited to *max_messages*."""

        self.session_id = session_id
        self.max_messages = max_messages
        self.redis_client = redis_client
        self.session_ttl_seconds = session_ttl_seconds
        logger.debug(
            "Initialised RedisConversationMemory",
            extra={"session_id": session_id, "max_messages": max_messages},
        )

    def load(self) -> List[dict]:
        """Return the raw history payload for the configured session."""

        history = load_recent_history(self.redis_client, self.session_id, n=self.max_messages)
        logger.debug(
            "Loaded history for conversation",
            extra={"session_id": self.session_id, "items": len(history)},
        )
        return history

    def get_chat_history_pairs(self) -> List[Tuple[str, str]]:
        """Return normalized ``(user, assistant)`` message tuples for LangChain."""

        recent = self.load()
        pairs = []
        i = 0
        n = len(recent)
        while i < n:
            if recent[i]["role"] == "user":
                user_text = recent[i]["text"]
                assistant_text = ""
                if i + 1 < n and recent[i + 1]["role"] == "assistant":
                    assistant_text = recent[i + 1]["text"]
                    i += 2
                else:
                    i += 1
                pairs.append((user_text, assistant_text))
            else:
                pairs.append(("", recent[i]["text"]))
                i += 1
        return pairs

    def add_user_message(self, text: str) -> None:
        """Persist a user utterance and refresh the session TTL."""

        append_history_item(
            self.redis_client,
            self.session_ttl_seconds,
            self.session_id,
            "user",
            text,
        )
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

        append_history_item(
            self.redis_client,
            self.session_ttl_seconds,
            self.session_id,
            "assistant",
            text,
            sources=sources,
        )
        logger.info(
            "Stored assistant message",
            extra={
                "session_id": self.session_id,
                "preview": text[:100],
                "has_sources": bool(sources),
            },
        )


__all__ = [
    "RedisConversationMemory",
    "append_history_item",
    "build_redis_client",
    "load_recent_history",
    "session_exists",
]
