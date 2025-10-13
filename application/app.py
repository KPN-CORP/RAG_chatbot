"""FastAPI application entry point exposing chat and session endpoints."""

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

import redis
from config import (
    MAX_HISTORY_MESSAGES,
    REDIS_URL,
    SESSION_TTL_SECONDS,
    collection_name,
    collection_description,
    embedding_model_name,
    gemini_chat_model,
    google_api_key_name,
    milvus_host,
    milvus_port,
    qa_system_prompt,
)
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from ingest.langchain_milvus_ingest import ingest_pages_to_milvus
from ingest.pdf_with_tables import extract_pdf_content
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain imports (use provider wrapper)
from langchain_google_genai import ChatGoogleGenerativeAI
from logging_config import configure_logging, get_logger
from modules.history_utils import build_chat_history_messages
from modules.redis import RedisConversationMemory
from schemas.api import ChatReq, ChatResp, CreateSessionResp
from schemas.rag_outputs import AnswerWithSources, format_docs_with_metadata
from vectorstore.milvus_vectorstore import get_vectorstore

from dotenv import load_dotenv

load_dotenv()

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


def _resolve_embedding_api_key() -> str:
    """Fetch the Gemini embedding API key using known environment fallbacks."""

    candidates = [
        os.environ.get(google_api_key_name),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
    ]
    for value in candidates:
        if value:
            return value
    return ""


def _expected_ingest_token() -> str:
    """Return the configured ingestion upload token from the environment."""

    return os.environ.get("INGEST_UPLOAD_TOKEN", "")


def _build_document_identifier(filename: str, data: bytes) -> str:
    """Create a stable document identifier using filename and SHA-256 digest."""

    name = Path(filename).stem if filename else "document"
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"{safe_name or 'document'}-{digest}"


# ----- Endpoints -----
@app.post("/session", response_model=CreateSessionResp)
def create_session():
    """Create a new chat session backed by Redis."""

    sid = str(uuid.uuid4())
    r.hset(_session_meta_key(sid), mapping={"created_at": str(__import__("time").time())})
    r.expire(_session_meta_key(sid), SESSION_TTL_SECONDS)
    logger.info("Created new chat session", extra={"session_id": sid})
    return {"session_id": sid}


@app.post("/ingest/upload")
async def upload_document(
    file: UploadFile = File(...),
    ingest_token: str = Header(..., alias="X-Ingest-Token"),
):
    """Ingest a user-provided PDF document into the Milvus knowledge base."""

    expected_token = _expected_ingest_token()
    if not expected_token:
        logger.error("Missing ingestion token configuration")
        raise HTTPException(status_code=500, detail="Ingestion token is not configured.")

    if ingest_token != expected_token:
        logger.warning(
            "Invalid ingestion token supplied", extra={"doc_header_present": bool(ingest_token)}
        )
        raise HTTPException(status_code=401, detail="Invalid ingestion token.")

    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id = _build_document_identifier(file.filename or "document.pdf", data)
    logger.info(
        "Received document upload",
        extra={"filename": file.filename, "doc_id": doc_id, "size_bytes": len(data)},
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        pages = extract_pdf_content(tmp_path)
    except Exception as exc:  # pragma: no cover - pdf parsing issues
        logger.exception(
            "Failed to extract PDF content",
            extra={"filename": file.filename, "doc_id": doc_id},
        )
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logger.debug("Temporary upload file already removed", extra={"path": tmp_path})

    if not pages:
        raise HTTPException(status_code=400, detail="No readable content found in PDF.")

    api_key = _resolve_embedding_api_key()

    try:
        skipped, inserted = ingest_pages_to_milvus(
            pages=pages,
            doc_id=doc_id,
            source=file.filename or doc_id,
            collection_name=collection_name,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            embedding_model_name=embedding_model_name,
            api_key=api_key or None,
            collection_description=collection_description
        )
    except Exception as exc:  # pragma: no cover - ingestion pipeline errors
        logger.exception(
            "Failed to ingest document",
            extra={"filename": file.filename, "doc_id": doc_id},
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    logger.info(
        "Document ingestion completed",
        extra={
            "doc_id": doc_id,
            "filename": file.filename,
            "skipped_chunks": skipped,
            "inserted_chunks": inserted,
        },
    )

    return {
        "doc_id": doc_id,
        "filename": file.filename or doc_id,
        "skipped_chunks": skipped,
        "inserted_chunks": inserted,
    }


@app.post("/ask", response_model=ChatResp)
def chat(req: ChatReq):
    """Answer a user question using the retrieval augmented generation chain."""

    logger.info("Received chat request", extra={"session_id": req.session_id})

    # 1) Preconditions
    if not vectorstore:
        logger.error("Vectorstore is not initialised")
        raise HTTPException(status_code=500, detail="Vectorstore not wired in server.")
    if not session_exists(req.session_id):
        logger.warning("Chat request for unknown session", extra={"session_id": req.session_id})
        raise HTTPException(status_code=404, detail="Session not found. Create a session first.")

    # 2) Create session-scoped memory helper (Redis)
    mem = RedisConversationMemory(session_id=req.session_id, max_messages=MAX_HISTORY_MESSAGES)

    # 3) Build retriever from vectorstore (and optionally wrap with a sanitizer)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    # If you have a safe SanitizingRetriever (no LLM calls inside), use it; otherwise keep the plain retriever:
    # sanitized_retriever = SanitizingRetriever(retriever)

    debug_retrieval = bool(getattr(req, "debug_retrieval", False))
    retrieved_docs = []
    formatted_context = None

    if debug_retrieval:
        # Fetch documents once so they can be reused for both the prompt context and
        # any downstream logging/inspection in this function.
        try:
            retrieved_docs = retriever.invoke(req.question) or []
        except Exception:
            logger.exception("Failed to retrieve documents", extra={"session_id": req.session_id})
            raise HTTPException(status_code=500, detail="Retriever error while fetching documents.")

        # Prepare the formatted context string once so the chain does not need to
        # re-run document formatting internally.
        formatted_context = format_docs_with_metadata(retrieved_docs)

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
            MessagesPlaceholder(variable_name="chat_history"),  # inject conversation history here
            # ("system", "Use the following extracted documents to answer the user's question:\n\n{context}"),
            ("human", "{question}"),
        ]
    )

    # 6) Build the RAG chain (components + flow)
    if debug_retrieval:
        rag_chain = (
            {
                # The retrieved docs are pre-fetched and formatted to a context string above.
                "context": (lambda x: x["context"]),
                # The original 'question' is passed through unchanged.
                "question": (lambda x: x["question"]),
                # The 'chat_history' is passed through unchanged.
                "chat_history": (lambda x: x["chat_history"]),
            }
            | qa_prompt
            | llm
            | pydantic_parser
        )
    else:
        rag_chain = (
            {
                # The retriever runs first, using the original question.
                # Its output is passed to format_docs and becomes the 'context'.
                "context": (lambda x: x["question"]) | retriever | format_docs_with_metadata,
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
    chat_history_messages = build_chat_history_messages(raw_history)  # returns List[BaseMessage]

    # Debug print to inspect normalized messages (types and content preview)
    # print("chat_history_messages in app.py:", chat_history_messages)

    # 8) Invoke the chain; pass 'history' (string) and let the chain fetch context from retriever
    invoke_payload = {
        "question": req.question,
        "chat_history": chat_history_messages,
    }
    if debug_retrieval:
        invoke_payload["context"] = formatted_context

    try:
        # The invoke call now passes a single dictionary with all the required keys
        result = rag_chain.invoke(invoke_payload)
        if not result.get("sources"):
            logger.warning("No sources found for the answer", extra={"session_id": req.session_id})
            not_found = True
        else:
            not_found = False

    except Exception as e:
        # print full exception with context for easier debugging in logs
        logger.exception(
            "Error invoking chain",
            extra={"session_id": req.session_id, "history": raw_history},
        )
        # Log/raise more helpful error in production
        raise HTTPException(status_code=500, detail=f"LLM/chain error: {e}")

    # 9) Debug logging helper: inspect retrieved docs from this request
    if debug_retrieval:
        if retrieved_docs:
            logger.debug(
                "Retrieved documents",
                extra={
                    "session_id": req.session_id,
                    "retrieved_docs": [
                        {
                            "filename": doc.metadata.get("filename", "N/A"),
                            "page": doc.metadata.get("page", "N/A"),
                        }
                        for doc in retrieved_docs
                    ],
                },
            )
        else:
            logger.debug(
                "No documents retrieved for question",
                extra={"session_id": req.session_id, "question": req.question},
            )

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
    return {"answer": assistant_text, "sources": sources_meta, "not_found": not_found}


def print_retrieved_docs(result: dict) -> None:
    """Pretty-print retrieved source documents for ad-hoc debugging."""

    if "source_documents" in result:
        print("\n🔎 Retrieved Documents:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"--- Document {i} ---")
            print("Content:", doc.page_content[:500])  # limit to 500 chars
            print("Metadata:", doc.metadata)


# to start: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# to start: uvicorn app:app --reload --host 127.0.0.1 --port 5002
