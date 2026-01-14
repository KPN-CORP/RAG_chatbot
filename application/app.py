"""FastAPI application entry point exposing chat and session endpoints."""

import os
import tempfile
import uuid
import time
import json

import redis
from functools import lru_cache
from typing import Optional
from config import BUSINESS_UNITS
from config_loader import load_config
from modules.utils import (
    extract_json_string,
    meta_filename,
    meta_page,
    _build_document_identifier,
    sanitize_collection_component,
)

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from ingest.langchain_milvus_ingest import ingest_pages_to_milvus, ingest_text_document
from ingest.pdf_with_tables import extract_pdf_content
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

# LangChain imports (use provider wrapper)
from langchain_google_genai import ChatGoogleGenerativeAI
from logging_config import configure_logging, get_logger
from modules.history_utils import build_chat_history_messages
from modules.redis import (
    RedisConversationMemory,
    build_redis_client,
    refresh_session_ttl,
    session_exists,
)
from schemas.api import ChatReq, ChatResp, CreateSessionResp
from schemas.rag_outputs import AnswerWithSources, format_docs_with_metadata
from vectorstore.milvus_vectorstore import get_vectorstore

from dotenv import load_dotenv

load_dotenv()

# Configure logging using the default profile at import time; profile-specific
# calls may further tweak the level at runtime if needed.
default_config = load_config()
configure_logging(getattr(default_config, "LOG_LEVEL", None))
logger = get_logger(__name__)

app = FastAPI()
pydantic_parser = PydanticOutputParser(pydantic_object=AnswerWithSources)


@lru_cache(maxsize=None)
def _redis_client(redis_url: str) -> redis.Redis:
    """Return a cached Redis client for the provided URL."""

    return build_redis_client(redis_url)


def get_runtime_config(profile: str):
    """Load the config for the requested profile and refresh logging level.

    The profile maps to ``config_{PROFILE}.py`` (defaulting to ``config.py``)
    so each request can use its own chatbot persona/configuration.
    """

    config = load_config(profile)
    configure_logging(getattr(config, "LOG_LEVEL", None))
    return config


def get_redis_for_config(config):
    """Return a Redis client bound to the active profile configuration."""

    return _redis_client(config.REDIS_URL)


# ----- Helpers (optional) -----
def _session_meta_key(sid: str) -> str:
    """Return the Redis metadata key for the given session identifier."""

    return f"session:{sid}:meta"


def session_exists_for_client(redis_client, sid: str, session_ttl_seconds: int) -> bool:
    """Check whether a chat session exists and keep it alive if it does."""

    exists = session_exists(redis_client, sid)
    if exists:
        refresh_session_ttl(redis_client, sid, session_ttl_seconds)
    return exists

def _expected_ingest_token() -> str:
    """Return the configured ingestion upload token from the environment."""

    return os.environ.get("INGEST_UPLOAD_TOKEN", "")

def extractor_wrapper(msg):
    # logger.debug("Extracting JSON from LLM message", extra={"message": msg})
    logger.debug("Extracting JSON from LLM message", extra={"llm_message": msg})
    default_json = json.dumps({"answer": "", "sources": []}, ensure_ascii=False)
    # msg may be a string or a ChatMessage; normalize to text
    text = getattr(msg, "content", msg)
    return extract_json_string(text) or default_json

def _parse_ttl(value) -> Optional[int]:
    """Return a valid TTL int (>0) or None."""
    if value is None:
        return None
    try:
        ttl = int(str(value).strip())
        return ttl if ttl > 0 else None
    except (TypeError, ValueError):
        return None


def _resolve_collection_name(config, business_unit: Optional[str]) -> str:
    """Return the Milvus collection name for the request, optionally scoped by unit."""

    if not business_unit:
        return config.collection_name
    if business_unit not in BUSINESS_UNITS:
        raise HTTPException(
            status_code=400,
            detail=f"business_unit must be one of: {', '.join(BUSINESS_UNITS)}",
        )
    suffix = sanitize_collection_component(business_unit)
    return f"{config.collection_name}_{suffix}" if suffix else config.collection_name


def _ingest_pdf_upload(
    *, data: bytes, upload: UploadFile, doc_id: str, config, collection_name: str
) -> tuple[int, int]:
    """Persist PDF upload to disk, extract pages, and push them to Milvus."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        pages = extract_pdf_content(tmp_path)
    except Exception as exc:  # pragma: no cover - pdf parsing issues
        logger.exception(
            "Failed to extract PDF content",
            extra={"upload_filename": upload.filename, "doc_id": doc_id},
        )
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logger.debug("Temporary upload file already removed", extra={"path": tmp_path})

    if not pages:
        raise HTTPException(status_code=400, detail="No readable content found in PDF.")

    try:
        return ingest_pages_to_milvus(
            pages=pages,
            doc_id=doc_id,
            source=upload.filename or doc_id,
            collection_name=collection_name,
            config=config,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            collection_description=config.collection_description,
        )
    except Exception as exc:  # pragma: no cover - ingestion pipeline errors
        logger.exception(
            "Failed to ingest document",
            extra={"upload_filename": upload.filename, "doc_id": doc_id},
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


def _ingest_text_upload(
    *, data: bytes, upload: UploadFile, doc_id: str, config, collection_name: str
) -> tuple[int, int]:
    """Decode TXT uploads (with fallback) and ingest their contents."""

    try:
        text_content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text_content = data.decode("latin-1")
        except UnicodeDecodeError as exc:
            logger.exception(
                "Failed to decode text upload",
                extra={"upload_filename": upload.filename, "doc_id": doc_id},
            )
            raise HTTPException(
                status_code=400,
                detail="Unable to decode TXT file. Please upload UTF-8 encoded text.",
            ) from exc

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="No readable content found in TXT file.")

    try:
        return ingest_text_document(
            text=text_content,
            doc_id=doc_id,
            source=upload.filename or doc_id,
            collection_name=collection_name,
            config=config,
            collection_description=config.collection_description,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
        )
    except Exception as exc:  # pragma: no cover - ingestion pipeline errors
        logger.exception(
            "Failed to ingest text document",
            extra={"upload_filename": upload.filename, "doc_id": doc_id},
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


# ----- Endpoints -----
@app.post("/session", response_model=CreateSessionResp)
def create_session(profile: str = ""):
    """Create a new chat session backed by Redis."""
    config = get_runtime_config(profile)  # empty profile → application/config.py
    redis_client = get_redis_for_config(config)
    sid = str(uuid.uuid4())
    key = _session_meta_key(sid)
    created_at = int(time.time())
    ttl = _parse_ttl(config.SESSION_TTL_SECONDS)

    try:
        pipe = redis_client.pipeline()
        pipe.hset(key, mapping={"created_at": str(created_at)})
        if ttl is not None:
            pipe.expire(key, ttl)
        pipe.execute()
    except Exception:
        logger.exception("Failed to create chat session", extra={"session_id": sid})
        raise HTTPException(status_code=500, detail="Failed to create session")

    logger.info(
        "Created new chat session",
        extra={"session_id": sid, "ttl": ttl, "profile": profile or "default"},
    )
    return {"session_id": sid}


@app.post("/ingest/upload")
async def upload_document(
    file: UploadFile = File(...),
    ingest_token: str = Header(..., alias="X-Ingest-Token"),
    profile: str = "",
    business_unit: Optional[str] = None,
):
    """Ingest a user-provided PDF or TXT document into the Milvus knowledge base.

    Optionally scopes ingestion to a business-unit-specific Milvus collection.
    """

    config = get_runtime_config(profile)  # empty profile → application/config.py
    collection_name = _resolve_collection_name(config, business_unit)
    expected_token = _expected_ingest_token()
    if not expected_token:
        logger.error("Missing ingestion token configuration")
        raise HTTPException(status_code=500, detail="Ingestion token is not configured.")

    if ingest_token != expected_token:
        logger.warning(
            "Invalid ingestion token supplied", extra={"doc_header_present": bool(ingest_token)}
        )
        raise HTTPException(status_code=401, detail="Invalid ingestion token.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id = _build_document_identifier(file.filename or "document.pdf", data)
    logger.info(
        "Received document upload",
        extra={
            "upload_filename": file.filename,
            "content_type": file.content_type,
            "business_unit": business_unit,
        }
    )

    filename_lower = (file.filename or "").lower()
    is_pdf_upload = file.content_type in {"application/pdf", "application/x-pdf"} or filename_lower.endswith(".pdf")
    is_text_upload = file.content_type in {"text/plain"} or filename_lower.endswith(".txt")

    if not is_pdf_upload and not is_text_upload:
        raise HTTPException(status_code=400, detail="Only PDF or TXT uploads are supported.")


    if is_pdf_upload:
        skipped, inserted = _ingest_pdf_upload(
            data=data,
            upload=file,
            doc_id=doc_id,
            config=config,
            collection_name=collection_name,
        )
    else:
        skipped, inserted = _ingest_text_upload(
            data=data,
            upload=file,
            doc_id=doc_id,
            config=config,
            collection_name=collection_name,
        )

    logger.info(
        "Document ingestion completed",
        extra={
            "doc_id": doc_id,
            "upload_filename": file.filename,
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
def chat(req: ChatReq, profile: str = ""):
    """Answer a user question using the retrieval augmented generation chain."""
    config = get_runtime_config(profile)  # empty profile → application/config.py
    redis_client = get_redis_for_config(config)
    collection_name = _resolve_collection_name(config, req.business_unit)
    vectorstore = get_vectorstore(config, collection_name=collection_name)
    session_ttl = _parse_ttl(config.SESSION_TTL_SECONDS)

    logger.info(
        "Received chat request",
        extra={"session_id": req.session_id, "profile": profile or "default"},
    )

    if not vectorstore:
        logger.error("Vectorstore is not initialised")
        raise HTTPException(status_code=500, detail="Vectorstore not wired in server.")
    if not session_exists_for_client(redis_client, req.session_id, session_ttl):
        logger.warning(
            "Chat request for unknown session", extra={"session_id": req.session_id}
        )
        raise HTTPException(status_code=404, detail="Session not found. Create a session first.")

    mem = RedisConversationMemory(
        session_id=req.session_id,
        redis_client=redis_client,
        max_messages=config.MAX_HISTORY_MESSAGES,
        session_ttl_seconds=session_ttl or 0,
    )

    raw_history = mem.get_chat_history_pairs()  # whatever your Redis class returns
    chat_history_messages = build_chat_history_messages(raw_history)  # returns List[BaseMessage]
    history_for_rewrite = "\n".join(
        [
            f"User: {m.content}" if getattr(m, "type", "") == "human" else f"Assistant: {m.content}"
            for m in chat_history_messages
        ]
    )

    # 3) Build retriever from vectorstore (and optionally wrap with a sanitizer)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": config.top_k_retrieval}
    )
    # If you have a safe SanitizingRetriever (no LLM calls inside), use it; otherwise keep the plain retriever:
    # sanitized_retriever = SanitizingRetriever(retriever)

    debug_retrieval = bool(getattr(req, "debug_retrieval", False))
    retrieved_docs = []
    formatted_context = None

    # 4) Build the LLM client (safe fetch of API key)
    api_key_value = os.environ.get(
        config.google_api_key_name
    )  # ensure `google_api_key_name` is the env var name string from config
    if not api_key_value:
        logger.error(
            "Missing Google API key in environment",
            extra={"env_var": config.google_api_key_name},
        )
        raise HTTPException(status_code=500, detail="LLM API key not configured.")
    llm = ChatGoogleGenerativeAI(model=config.gemini_chat_model, google_api_key=api_key_value)

    contextualize_chain = (
        {
            "chat_history": (lambda x: x["history_for_rewrite"]),
            "input": (lambda x: x["question"]),
        }
        | config.CONTEXTUALIZE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    )

    try:
        standalone_question = contextualize_chain.invoke(
            {
                "question": req.question,
                "chat_history": chat_history_messages,
                "history_for_rewrite": history_for_rewrite,
            }
        )
        if debug_retrieval:
            logger.debug(
                "Contextualized question generated",
                extra={
                    "session_id": req.session_id,
                    "original_question": req.question,
                    "standalone_question": standalone_question,
                },
            )
    except Exception:
        logger.exception(
            "Failed to contextualize question", extra={"session_id": req.session_id}
        )
        raise HTTPException(status_code=500, detail="Failed to rewrite question for retrieval.")

    if debug_retrieval:
        # Fetch documents once so they can be reused for both the prompt context and
        # any downstream logging/inspection in this function.
        try:
            retrieved_docs = retriever.invoke(standalone_question) or []
            for i, d in enumerate(retrieved_docs, 1):
                logger.debug(
                    {
                        "i": i,
                        "chunk": d.page_content,
                        "filename": meta_filename(d.metadata),
                        "page": meta_page(d.metadata),
                        "keys": list(d.metadata.keys()),
                    }
                )
        except Exception:
            logger.exception("Failed to retrieve documents", extra={"session_id": req.session_id})
            raise HTTPException(status_code=500, detail="Retriever error while fetching documents.")

        # Prepare the formatted context string once so the chain does not need to
        # re-run document formatting internally.
        formatted_context = format_docs_with_metadata(retrieved_docs)

    # 5) Build the prompt — USE a plain string {history} placeholder (avoid MessagesPlaceholder)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.qa_system_prompt),
            (
                "system",
                (
                    "Gunakan konteks berikut untuk menjawab pertanyaan karyawan:\n{context}\n\n"
                    "Ikuti instruksi format berikut agar output bisa diparsing secara otomatis:\n"
                    "{format_instructions}"
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),  # inject conversation history here
            ("human", "{question}"),
        ]
    ).partial(format_instructions=pydantic_parser.get_format_instructions())

    # 6) Build the RAG chain (components + flow)
    if debug_retrieval:
        rag_chain = (
            {
                # The retrieved docs are pre-fetched and formatted to a context string above.
                "context": (lambda x: x["context"]),
                # The rewritten question is passed through unchanged.
                "question": (lambda x: x["standalone_question"]),
                # The 'chat_history' is passed through unchanged.
                "chat_history": (lambda x: x["chat_history"]),
            }
            | qa_prompt
            | llm
            | RunnableLambda(extractor_wrapper)
            | pydantic_parser
        )
    else:
        rag_chain = (
            {
                # The retriever runs first, using the rewritten standalone question.
                # Its output is passed to format_docs and becomes the 'context'.
                "context": (lambda x: x["standalone_question"]) | retriever | format_docs_with_metadata,
                # The rewritten question is passed through unchanged.
                "question": (lambda x: x["standalone_question"]),
                # The 'chat_history' is passed through unchanged.
                "chat_history": (lambda x: x["chat_history"]),
            }
            | qa_prompt
            | llm
            | RunnableLambda(extractor_wrapper)
            | pydantic_parser
        )

    # 8) Invoke the chain; pass 'history' (string) and let the chain fetch context from retriever
    invoke_payload = {
        "question": req.question,
        "standalone_question": standalone_question,
        "chat_history": chat_history_messages,
    }
    if debug_retrieval:
        invoke_payload["context"] = formatted_context

    try:
        # The invoke call now passes a single dictionary with all the required keys
        result = rag_chain.invoke(invoke_payload)
        logger.debug(f"RAG chain invoked. Result: {result}", extra={"session_id": req.session_id})
        try:
            if not result.sources:
                logger.warning("No sources found for the answer", extra={"session_id": req.session_id})
                not_found = True
            else:
                not_found = False
        except:
            not_found = True

    except Exception as e:
        # print full exception with context for easier debugging in logs
        logger.exception(
            "Error invoking chain",
            extra={"session_id": req.session_id, "history": raw_history},
        )
        not_found = True
        # Log/raise more helpful error in production
        raise HTTPException(status_code=500, detail=f"LLM/chain error: {e}")

    # 9) Debug logging helper: inspect retrieved docs from this request
    if debug_retrieval:
        if retrieved_docs:
            logger.debug(
                f"Retrieved documents{formatted_context}"
            )
        else:
            logger.debug(
                "No documents retrieved for question",
                extra={"session_id": req.session_id, "question": req.question},
            )

    # Pull assistant answer and source documents
    try:
        assistant_text = result.answer or ""
        sources_meta = result.sources or []

        # 10) Save the turn into Redis after successful call: user then assistant
        mem.add_user_message(req.question)
        sources_dicts = [s.model_dump() for s in sources_meta]   
        mem.add_assistant_message(assistant_text, sources=sources_dicts)
        logger.info(
            "Chat response generated",
            extra={
                "session_id": req.session_id,
                "has_sources": bool(sources_meta),
                "answer_preview": assistant_text[:100],
            },
        )
    except Exception as exc:  # pragma: no cover - redis save errors
        assistant_text = ''
        sources_meta = []
        logger.exception(
            f"Failed to save chat turn to Redis. Result: {result}",
            extra={"session_id": req.session_id, "error": str(exc)},
        )

    # 11) Ensure return types match Pydantic model. IRIS has a slimmer contract
    # (no sources) to satisfy upstream expectations, while other profiles keep
    # the full payload with citations for downstream consumers.
    if profile and profile.upper() == "IRIS":
        return {"answer": assistant_text, "not_found": not_found}

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
