from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
import json
import logging

# ... inside chat() endpoint:

def build_chat_history_messages(raw):
    """
    Normalize various history formats into List[BaseMessage].
    Supported input:
      - List of (user_text, assistant_text) tuples/lists
      - List of dicts like {"role":"user","content":"..."} or {"role":"assistant","content":"..."}
      - Already a list of BaseMessage (returned unchanged)
      - If a string is passed, try to JSON-decode to list; otherwise ignore and warn
    """
    msgs = []
    if raw is None:
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
            logging.warning("chat_history is a plain string and not valid JSON. Ignoring.")
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
            logging.warning("Unknown chat_history item type: %s — skipping", type(item))
    else:
        logging.warning("chat_history is not a list; type=%s", type(raw))

    return msgs

