# app.py
import os
import uuid
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
from module.redis import RedisConversationMemory

from config import REDIS_URL, SESSION_TTL_SECONDS, MAX_HISTORY_MESSAGES, gemini_chat_model, API_KEY
from milvus_vectorstore import get_vectorstore

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
# assume you already have your Milvus vectorstore instance created (vs)
# from langchain.vectorstores import Milvus
# vs = Milvus(...)

# ----- CONFIG -----
# initialize
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
app = FastAPI()

# Replace with your initialized components
# Example placeholders (you already created the vectorstore)
vectorstore = get_vectorstore()   # <-- set your Milvus vectorstore instance here

# ----- Pydantic models -----
class CreateSessionResp(BaseModel):
    session_id: str

class ChatReq(BaseModel):
    session_id: str
    question: str

class ChatResp(BaseModel):
    answer: str
    sources: List[dict]

# ----- Endpoints -----
@app.post("/session", response_model=CreateSessionResp)
def create_session():
    sid = str(uuid.uuid4())
    # optionally create metadata in Redis
    r.hset(f"session:{sid}:meta", mapping={"created_at": str(__import__("time").time())})
    r.expire(f"session:{sid}:meta", SESSION_TTL_SECONDS)
    return {"session_id": sid}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore not wired in server.")
    # Validate session exists (optional)
    # Build retriever from vectorstore (tune k, score threshold):
    retriever = vectorstore.as_retriever(search_type="similarity",
                                        search_kwargs={"k": 6})  # tune k

    # Build LLM and chain
    llm = ChatGoogleGenerativeAI(model=gemini_chat_model, google_api_key=os.environ[API_KEY])  # your Gemini chat model
    # We'll use a ConversationBufferMemory and populate it from Redis
    # create memory helper for this session
    mem = RedisConversationMemory(session_id=req.session_id, max_messages=MAX_HISTORY_MESSAGES)

    # Save the turn into Redis: user then assistant (include source docs metadata)
    mem.add_user_message(req.question)

    # Build retriever & LLM as you did previously (assuming 'retriever' and 'llm' exist)
    # conv_chain WITHOUT ConversationBufferMemory
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,  # do not pass deprecated memory object
        return_source_documents=True,
    )

    # Build chat_history pairs and call chain
    chat_history_pairs = mem.get_chat_history_pairs()  # list of tuples: [(user, assistant), ...]
    result = conv_chain({"question": req.question, "chat_history": chat_history_pairs})
    
    # Pull assistant answer and sources from result (LangChain keys may vary: "answer" or "result")
    assistant_text = result.get("answer") or result.get("result") or ""
    source_docs = result.get("source_documents", [])
    sources_meta = [d.metadata for d in source_docs] if source_docs else None
    mem.add_assistant_message(assistant_text, sources=sources_meta)

    # result still contains 'answer' and 'source_documents' for your response
    return {"answer": assistant_text, "sources": sources_meta}