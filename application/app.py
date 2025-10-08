"""FastAPI application entry point exposing chat and session endpoints."""

import os
import uuid

import redis
from fastapi import FastAPI, HTTPException
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain imports (use provider wrapper)
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    MAX_HISTORY_MESSAGES,
    REDIS_URL,
    SESSION_TTL_SECONDS,
    gemini_chat_model,
    google_api_key_name,
    qa_system_prompt,
)
from logging_config import configure_logging, get_logger
from modules.history_utils import build_chat_history_messages
from modules.redis import RedisConversationMemory
from schemas.api import ChatReq, ChatResp, CreateSessionResp
from schemas.rag_outputs import AnswerWithSources, format_docs_with_metadata
from vectorstore.milvus_vectorstore import get_vectorstore

# Configure logging once when the module is imported.
configure_logging()
logger = get_logger(__name__)

# ----- CONFIG -----
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
app = FastAPI()
pydantic_parser = PydanticOutputParser(pydantic_object=AnswerWithSources)

# Your already-created vectorstore (LangChain Milvus wrapper)
vectorstore = get_vectorstore()


# ----- Helpers (optional) -----
def _session_meta_key(sid: str) -> str:
    """Return the Redis metadata key for the given session identifier."""

    return f"session:{sid}:meta"


def session_exists(sid: str) -> bool:
    """Check whether a chat session already exists in Redis."""

    return r.exists(_session_meta_key(sid)) == 1


# ----- Endpoints -----
@app.post("/session", response_model=CreateSessionResp)
def create_session():
    """Create a new chat session backed by Redis."""

    sid = str(uuid.uuid4())
    r.hset(
        _session_meta_key(sid), mapping={"created_at": str(__import__("time").time())}
    )
    r.expire(_session_meta_key(sid), SESSION_TTL_SECONDS)
    logger.info("Created new chat session", extra={"session_id": sid})
    return {"session_id": sid}


@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    """Answer a user question using the retrieval augmented generation chain."""

    logger.info("Received chat request", extra={"session_id": req.session_id})

    # 1) Preconditions
    if not vectorstore:
        logger.error("Vectorstore is not initialised")
        raise HTTPException(status_code=500, detail="Vectorstore not wired in server.")
    if not session_exists(req.session_id):
        logger.warning(
            "Chat request for unknown session", extra={"session_id": req.session_id}
        )
        raise HTTPException(
            status_code=404, detail="Session not found. Create a session first."
        )

    # 2) Create session-scoped memory helper (Redis)
    mem = RedisConversationMemory(
        session_id=req.session_id, max_messages=MAX_HISTORY_MESSAGES
    )

    # 3) Build retriever from vectorstore (and optionally wrap with a sanitizer)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    # If you have a safe SanitizingRetriever (no LLM calls inside), use it; otherwise keep the plain retriever:
    # sanitized_retriever = SanitizingRetriever(retriever)

    # 4) Build the LLM client (safe fetch of API key)
    api_key_value = os.environ.get(
        google_api_key_name
    )  # ensure `google_api_key_name` is the env var name string from config
    if not api_key_value:
        logger.error(
            "Missing Google API key in environment",
            extra={"env_var": google_api_key_name},
        )
        raise HTTPException(status_code=500, detail="LLM API key not configured.")
    llm = ChatGoogleGenerativeAI(model=gemini_chat_model, google_api_key=api_key_value)

    # 5) Build the prompt — USE a plain string {history} placeholder (avoid MessagesPlaceholder)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # inject conversation history here
            # ("system", "Use the following extracted documents to answer the user's question:\n\n{context}"),
            ("human", "{question}"),
        ]
    )

    # 6) Build the RAG chain (components + flow)
    rag_chain = (
        {
            # The retriever runs first, using the original question.
            # Its output is passed to format_docs and becomes the 'context'.
            "context": (lambda x: x["question"])
            | retriever
            | format_docs_with_metadata,
            # The original 'question' is passed through unchanged.
            "question": (lambda x: x["question"]),
            # The 'chat_history' is passed through unchanged.
            "chat_history": (lambda x: x["chat_history"]),
        }
        | qa_prompt
        | llm
        | pydantic_parser
    )

    # 7) Load raw history from Redis and normalize to BaseMessage list (helper)
    raw_history = mem.get_chat_history_pairs()  # whatever your Redis class returns
    chat_history_messages = build_chat_history_messages(
        raw_history, req.question
    )  # returns List[BaseMessage]

    # Debug print to inspect normalized messages (types and content preview)
    # print("chat_history_messages in app.py:", chat_history_messages)

    # 8) Invoke the chain; pass 'history' (string) and let the chain fetch context from retriever
    try:
        # The invoke call now passes a single dictionary with all the required keys
        result = rag_chain.invoke(
            {"question": req.question, "chat_history": chat_history_messages}
        )

    except Exception as e:
        # print full exception with context for easier debugging in logs
        logger.exception(
            "Error invoking chain",
            extra={"session_id": req.session_id, "history": raw_history},
        )
        # Log/raise more helpful error in production
        raise HTTPException(status_code=500, detail=f"LLM/chain error: {e}")

    # 9) Debug print retrieved docs and extract answer + sources
    # print_retrieved_docs(result)

    # Pull assistant answer and source documents
    assistant_text = result.answer or result.result or ""
    sources_meta = result.sources or []

    # 10) Save the turn into Redis after successful call: user then assistant
    mem.add_user_message(req.question)
    mem.add_assistant_message(assistant_text, sources=sources_meta)
    logger.info(
        "Chat response generated",
        extra={
            "session_id": req.session_id,
            "has_sources": bool(sources_meta),
            "answer_preview": assistant_text[:100],
        },
    )

    # 11) Ensure return types match Pydantic model
    return {"answer": assistant_text, "sources": sources_meta}


def print_retrieved_docs(result: dict) -> None:
    """Pretty-print retrieved source documents for ad-hoc debugging."""

    if "source_documents" in result:
        print("\n🔎 Retrieved Documents:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"--- Document {i} ---")
            print("Content:", doc.page_content[:500])  # limit to 500 chars
            print("Metadata:", doc.metadata)


# to start: uvicorn app:app --reload --host 0.0.0.0 --port 8000
