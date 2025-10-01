# app.py
import os
import uuid

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis

from modules.redis import RedisConversationMemory
# from modules.retriever_wrappper import SanitizingRetriever
from modules.history_utils import build_chat_history_messages

from config import (REDIS_URL, SESSION_TTL_SECONDS, MAX_HISTORY_MESSAGES, 
                    gemini_chat_model, api_key_name, system_prompt
)
from vectorstore.milvus_vectorstore import get_vectorstore

import logging

# LangChain imports (use provider wrapper)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# ----- CONFIG -----
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
app = FastAPI()

# Your already-created vectorstore (LangChain Milvus wrapper)
vectorstore = get_vectorstore()

# ----- Pydantic models -----
class CreateSessionResp(BaseModel):
    session_id: str

class ChatReq(BaseModel):
    session_id: str
    question: str

class ChatResp(BaseModel):
    answer: str
    sources: List[dict]

# ----- Helpers (optional) -----
def _session_meta_key(sid: str) -> str:
    return f"session:{sid}:meta"

def session_exists(sid: str) -> bool:
    return r.exists(_session_meta_key(sid)) == 1

# ----- Endpoints -----
@app.post("/session", response_model=CreateSessionResp)
def create_session():
    sid = str(uuid.uuid4())
    r.hset(_session_meta_key(sid), mapping={"created_at": str(__import__("time").time())})
    r.expire(_session_meta_key(sid), SESSION_TTL_SECONDS)
    return {"session_id": sid}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    # 1) Preconditions
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore not wired in server.")
    if not session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Create a session first.")

    # 2) Create session-scoped memory helper (Redis)
    mem = RedisConversationMemory(session_id=req.session_id, max_messages=MAX_HISTORY_MESSAGES)

    # 3) Build retriever from vectorstore (and optionally wrap with a sanitizer)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    # If you have a safe SanitizingRetriever (no LLM calls inside), use it; otherwise keep the plain retriever:
    # sanitized_retriever = SanitizingRetriever(retriever)
    sanitized_retriever = retriever

    # 4) Build the LLM client (safe fetch of API key)
    api_key_value = os.environ.get(api_key_name)  # ensure `api_key_name` is the env var name string from config
    llm = ChatGoogleGenerativeAI(model=gemini_chat_model, google_api_key=api_key_value)

    # 5) Build the prompt — USE a plain string {history} placeholder (avoid MessagesPlaceholder)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),            # inject conversation history here
        ("system", "Use the following extracted documents to answer the user's question:\n\n{context}"),
        ("human", "{question}")
    ])

    # 6) Build the ConversationalRetrievalChain (memory=None is fine since we handle history manually)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=sanitized_retriever,
        memory=None,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    # 7) Load raw history from Redis and normalize to BaseMessage list (helper)
    raw_history = mem.get_chat_history_pairs()  # whatever your Redis class returns
    chat_history_messages = build_chat_history_messages(raw_history, req.question)  # returns List[BaseMessage]

    # Debug print to inspect normalized messages (types and content preview)
    print("chat_history_messages in app.py:", chat_history_messages)

    # 8) Invoke the chain; pass 'history' (string) and let the chain fetch context from retriever
    try:
        # deprecated
        # result = conv_chain({"question": req.question, "chat_history": chat_history_pairs})
        
        # synchronous call
        result = conv_chain.invoke({"question": req.question, "chat_history": chat_history_messages})
        # asynchronous call
        # result = await conv_chain.ainvoke({"question": req.question, "chat_history": chat_history_pairs})

    except Exception as e:
        # print full exception with context for easier debugging in logs
        logging.exception("Error invoking chain. Raw history repr: %s", repr(raw_history))
        # Log/raise more helpful error in production
        raise HTTPException(status_code=500, detail=f"LLM/chain error: {e}")

    # 9) Debug print retrieved docs and extract answer + sources
    print_retrieved_docs(result)

    # Pull assistant answer and source documents
    assistant_text = result.get("answer") or result.get("result") or ""
    source_docs = result.get("source_documents", []) or []
    sources_meta = [d.metadata for d in source_docs]

    # 10) Save the turn into Redis after successful call: user then assistant
    mem.add_user_message(req.question)
    mem.add_assistant_message(assistant_text, sources=sources_meta)

    # 11) Ensure return types match Pydantic model
    return {"answer": assistant_text, "sources": sources_meta}


def print_retrieved_docs(result: dict):
    # Retrieved docs
    if "source_documents" in result:
        print("\n🔎 Retrieved Documents:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"--- Document {i} ---")
            print("Content:", doc.page_content[:500])   # limit to 500 chars
            print("Metadata:", doc.metadata)


# to start: uvicorn app:app --reload --host 0.0.0.0 --port 8000
