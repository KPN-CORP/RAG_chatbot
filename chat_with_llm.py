# rag_gemini_milvus_chat.py
import os
from typing import Dict, List

import google.generativeai as genai
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.base import Embeddings

# LangChain core
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory

# Vectorstore (Milvus) wrappers - try common package names
try:
    from langchain_milvus import Milvus as LangchainMilvus
except Exception:
    try:
        # community wrapper fallback
        from langchain_community.vectorstores.milvus import Milvus as LangchainMilvus
    except Exception:
        LangchainMilvus = None

from datetime import datetime

# SQLAlchemy for simple history persistence
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

############################
# CONFIG
############################
# 1) Set GOOGLE_API_KEY as env var or assign here for quick testing (not recommended in prod)
# export GOOGLE_API_KEY="your_api_key_here" (Linux/mac) or set in Windows environment
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", None)
if not GOOGLE_API_KEY:
    raise RuntimeError("Set environment variable GOOGLE_API_KEY first.")

genai.configure(api_key=GOOGLE_API_KEY)

# Gemini models
GEMINI_LLM_MODEL = "gemini-2.5-flash"  # choose appropriate model
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# Milvus connection args (local Docker / Milvus Lite default)
MILVUS_CONN = {"host": "127.0.0.1", "port": "19530"}
MILVUS_COLLECTION = "milvus_langchain_test"

# Chat history DB (SQLite for demo). For prod, use PostgreSQL: e.g. postgresql://user:pass@host/db
HISTORY_DB_URL = "sqlite:///chat_history.db"

############################
# SIMPLE SQLALCHEMY MODEL
############################
Base = declarative_base()


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), index=True)
    role = Column(String(16))  # "user" or "assistant" or "system"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_engine(HISTORY_DB_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


def save_message(session_id: str, role: str, content: str):
    db = SessionLocal()
    msg = ChatMessage(
        session_id=session_id, role=role, content=content, created_at=datetime.utcnow()
    )
    db.add(msg)
    db.commit()
    db.close()


def load_messages(session_id: str) -> List[Dict]:
    db = SessionLocal()
    rows = (
        db.query(ChatMessage)
        .filter_by(session_id=session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    out = [{"role": r.role, "content": r.content, "created_at": r.created_at} for r in rows]
    db.close()
    return out


############################
# LANGCHAIN LLM & EMBEDDINGS WRAPPERS (calls google.generativeai)
############################
class GeminiLangChainLLM(LLM):
    """LangChain LLM wrapper that calls google.generativeai (Gemini)."""

    # max_output_tokens = None  # optional

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        # Note: pick the correct API call shape for your google.generativeai SDK version.
        # We use GenerativeModel.generate_content as shown earlier examples.
        model = genai.GenerativeModel(GEMINI_LLM_MODEL)
        # You can pass temperature, max_output_tokens, etc. as kwargs.
        response = model.generate_content(prompt)
        # response.text is typical; inspect your SDK if different
        return response.text

    @property
    def _identifying_params(self):
        return {"model": GEMINI_LLM_MODEL}

    @property
    def _llm_type(self):
        return "gemini"


class GeminiLangChainEmbeddings(Embeddings):
    """LangChain embeddings wrapper around google.generativeai embed_content"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # genai.embed_content may return a nested structure; adapt if needed.
        resp = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=texts)
        # Typical shape: {'data': [{'embedding': [...]}, ...]} or resp['data'] depending on SDK.
        # Try to extract robustly:
        vectors = []
        if isinstance(resp, dict) and "data" in resp:
            for item in resp["data"]:
                vectors.append(item["embedding"])
        elif isinstance(resp, list):
            for item in resp:
                # if each item is {'embedding': ...}
                if isinstance(item, dict) and "embedding" in item:
                    vectors.append(item["embedding"])
                else:
                    raise RuntimeError("Unexpected embed response item format: " + str(item)[:200])
        else:
            # last fallback: try resp['predictions'] etc.
            raise RuntimeError(
                f"Unexpected embedding response format: {type(resp)} - inspect and adapt."
            )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vecs = self.embed_documents([text])
        return vecs[0]


############################
# CONNECT / LOAD MILVUS via LangChain wrapper
############################
if LangchainMilvus is None:
    raise RuntimeError(
        "Could not import LangChain Milvus wrapper. Install langchain-milvus or adjust imports."
    )

# Create embedding object for LangChain wrapper
embeddings = GeminiLangChainEmbeddings()

# Create or load the vectorstore wrapper which will call embed when needed
# from_documents can create the collection; to load existing collection you can connect via collection_name only
# Here we assume the collection already exists and was previously filled; LangChain wrapper will use existing collection.
vectorstore = LangchainMilvus(
    collection_name=MILVUS_COLLECTION,
    embedding_function=embeddings,
    connection_args=MILVUS_CONN,
)

# Build a retriever from the vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


############################
# Prepare memory (load persisted history and convert to LangChain messages)
############################
def build_memory_from_db(session_id: str) -> ConversationBufferMemory:
    # We'll create a ConversationBufferMemory that returns messages (return_messages=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    msgs = load_messages(session_id)
    # Convert loaded messages to langchain message objects and append to memory.chat_memory.messages
    for m in msgs:
        role = m["role"]
        content = m["content"]
        if role == "user":
            memory.chat_memory.add_user_message(content)
        elif role == "assistant":
            memory.chat_memory.add_ai_message(content)
        else:
            # system or other roles could be stored as system messages in beginning
            memory.chat_memory.add_system_message(content)
    return memory


############################
# Create conversational chain factory
############################
def create_conv_chain(session_id: str) -> ConversationalRetrievalChain:
    llm = GeminiLangChainLLM()
    memory = build_memory_from_db(session_id)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # helpful for provenance
    )
    return chain


############################
# Chat function (single turn)
############################
def chat(session_id: str, user_text: str) -> Dict:
    # 1) Save user message first
    save_message(session_id, "user", user_text)

    # 2) Create chain (loads conversation history into memory)
    chain = create_conv_chain(session_id)

    # 3) Ask the chain. The key depends on chain interface; ConversationalRetrievalChain expects {"question":..., "chat_history":...}
    result = chain({"question": user_text})

    # 4) Get assistant answer
    answer = result.get("answer") or result.get("output_text") or ""

    # 5) Save assistant message
    save_message(session_id, "assistant", answer)

    # 6) Return answer plus sources for debugging/provenance
    return {"answer": answer, "sources": result.get("source_documents", [])}


############################
# Simple interactive REPL for local testing
############################
if __name__ == "__main__":
    print("RAG chat with Gemini + Milvus (history persisted to SQLite).")
    sid = input("Enter session id (any string, e.g. user123): ").strip()
    print("Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = chat(sid, q)
        print("\nAssistant:\n", out["answer"])
        if out["sources"]:
            print("\nSources (top results):")
            for s in out["sources"][:3]:
                # each source is a langchain Document with .page_content and .metadata
                meta = getattr(s, "metadata", {})
                print(
                    "-",
                    (meta.get("source") or meta.get("source_id") or ""),
                    ":",
                    (s.page_content[:200] + ("..." if len(s.page_content) > 200 else "")),
                )
