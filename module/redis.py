import json
from typing import List, Tuple
import redis

from config import REDIS_URL, SESSION_TTL_SECONDS, MAX_HISTORY_MESSAGES

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ----- Helpers for Redis history -----
# Key patterns:
# "session:{session_id}:history" -> Redis list of JSON items: [{"role":"user","text":...}, {"role":"assistant","text":..., "sources":[...]}, ...]
def _history_key(session_id: str) -> str:
    return f"session:{session_id}:history"

def append_history_item(session_id: str, role: str, text: str, sources: List[dict] = None):
    item = {"role": role, "text": text}
    if sources:
        item["sources"] = sources
    r.rpush(_history_key(session_id), json.dumps(item))
    r.expire(_history_key(session_id), SESSION_TTL_SECONDS) # optional

def load_recent_history(session_id: str, n: int = MAX_HISTORY_MESSAGES):
    key = _history_key(session_id)
    raw = r.lrange(key, -n, -1) if r.exists(key) else []
    items = [json.loads(x) for x in raw]
    # return as list of (role, text) tuples for LangChain
    return items

# Redis-backed "memory-like" helper (mimics old ConversationBufferMemory enough for drop-in)
class RedisConversationMemory:
    def __init__(self, session_id: str, max_messages: int = MAX_HISTORY_MESSAGES):
        self.session_id = session_id
        self.max_messages = max_messages

    def load(self) -> List[dict]:
        return load_recent_history(self.session_id, n=self.max_messages)

    def get_chat_history_pairs(self) -> List[Tuple[str, str]]:
        """
        Return list of (user_text, assistant_text) tuples expected by many ConversationalRetrievalChains.
        If there are stray messages they will be handled reasonably:
          - user followed by assistant -> paired
          - user without assistant -> (user, "")
          - assistant without preceding user -> ("", assistant)
        """
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

    def add_user_message(self, text: str):
        append_history_item(self.session_id, "user", text)

    def add_assistant_message(self, text: str, sources: List[dict] = None):
        append_history_item(self.session_id, "assistant", text, sources=sources)
