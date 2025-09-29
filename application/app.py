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

from config import REDIS_URL, SESSION_TTL_SECONDS, MAX_HISTORY_MESSAGES, gemini_chat_model, API_KEY
from vectorstore.milvus_vectorstore import get_vectorstore

# LangChain imports (use provider wrapper)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


system_prompt = """Nama kamu adalah Nastari. Kamu adalah assistant di bidang AI di perusahaan KPN-Corp yang berada di Kuningan, Jakarta.
    Bersikaplah seperti HR profesional yang membantu karyawan dengan pertanyaan terkait kebijakan perusahaan dan masalah karyawan.
    Jika kamu tidak mengetahui jawabannya, katakan "Maaf, saya tidak tahu." Jangan membuat-buat jawaban, namun kamu bisa menanyakan lebih detail dan tawarkan untuk menghubungi tim HRD asli atau arahkan user untuk membaca dokumen yang relevan dari sumber yang diberikan jika ada.
    Berikan sumber dokumen yang relevan jika kamu memberikan jawaban berdasarkan dokumen.
    """


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
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vectorstore not wired in server.")

    # Optional: validate session exists
    if not session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Create a session first.")

    # Build retriever from vectorstore
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Wrap retriever to sanitize Documents
    # sanitized_retriever = SanitizingRetriever(retriever)
    sanitized_retriever = retriever

    # Build LLM (use env var safely)
    api_key_value = os.environ.get(API_KEY)  # API_KEY in config should be the env var name string
    llm = ChatGoogleGenerativeAI(model=gemini_chat_model, google_api_key=api_key_value)

    # Create Redis-backed memory helper for this session
    mem = RedisConversationMemory(session_id=req.session_id, max_messages=MAX_HISTORY_MESSAGES)

    # debug
    # docs = retriever.get_relevant_documents(req.question)
    # for i, d in enumerate(docs, 1):
    #     print(f"Doc {i} — page_content type:", type(d.page_content))
    #     # if it's a dict, print its keys or a short preview
    #     if not isinstance(d.page_content, str):
    #         import json
    #         print("page_content (as JSON):", json.dumps(d.page_content)[:500])
    #     print("metadata keys:", d.metadata.keys())
    #     print("----")
    # end debug

    # Build prompt with system + history + human
    # Important: include a placeholder for the retrieved docs variable: {context}
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),            # inject conversation history here
        ("system", "Use the following extracted documents to answer the user's question:\n\n{context}"),
        ("human", "{question}")
    ])

    # Create conv chain WITHOUT ConversationBufferMemory
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=sanitized_retriever,
        memory=None,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},  # 👈 inject your system prompt
    )


    # Build chat_history pairs (from Redis) and call chain
    chat_history_pairs = mem.get_chat_history_pairs()  # [(user, assistant), ...]
    try:
        # deprecated
        # result = conv_chain({"question": req.question, "chat_history": chat_history_pairs})
        
        # synchronous call
        result = conv_chain.invoke({"question": req.question, "chat_history": chat_history_pairs})
        # asynchronous call
        # result = await conv_chain.ainvoke({"question": req.question, "chat_history": chat_history_pairs})


    except Exception as e:
        # Log/raise more helpful error in production
        raise HTTPException(status_code=500, detail=f"LLM/chain error: {e}")

    print_retrieved_docs(result)

    # Pull assistant answer and source documents
    assistant_text = result.get("answer") or result.get("result") or ""
    source_docs = result.get("source_documents", []) or []
    sources_meta = [d.metadata for d in source_docs]

    # Save the turn into Redis after successful call: user then assistant
    mem.add_user_message(req.question)
    mem.add_assistant_message(assistant_text, sources=sources_meta)

    # Ensure return types match Pydantic model
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
