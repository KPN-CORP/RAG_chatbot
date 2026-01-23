import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

LOG_LEVEL = "INFO" # options: DEBUG, INFO, WARNING, ERROR, CRITICAL

load_dotenv()

redis_password = os.environ.get("REDIS_PASSWORD")
redis_password =  redis_password+"@" if redis_password else ''

milvus_host = "127.0.0.1"
milvus_port = "19530"
collection_name = "internal_document"  # collection name in Milvus
collection_description = "Internal Document for Rose's project"  # optional description

REDIS_URL = f"redis://{redis_password}localhost:6379/1"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # optional TTL for session keys
MAX_HISTORY_MESSAGES = 20  # how many recent messages to load into memory


gemini_chat_model = "models/gemini-2.5-flash-lite"  # pick the chat model you want
gemini_image_model = "models/gemini-2.5-flash" 
embedding_model_name = "models/gemini-embedding-001"  # embedding model for Gemini
semantic_embedding_model_name = "LazarusNLP/all-indo-e5-small-v4"  # embedding model for semantic search

top_k_retrieval = 7

google_api_key_name = "GEMINI_API_KEY_IRIS"  # environment variable name for Gemini/Google API key

WATCH_DIR = r"/home/hcispanel/kpn-iris/storage/app/private/knowledge-base/"  # change this
VALID_EXTS = {".pdf", ".docx", ".txt"}  # adapt to your project

def _escape_prompt_braces(prompt: str, *, allow: set[str]) -> str:
    """Escape all braces except those explicitly allowed as template vars."""

    escaped = prompt.replace("{", "{{").replace("}", "}}")
    for key in allow:
        escaped = escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}")
    return escaped


_qa_system_prompt_template = """Anda adalah IRIS, asisten AI untuk KPN Corp. Tugas Anda adalah menjawab pertanyaan karyawan dengan jelas dan mudah dibaca.
ATURAN PENTING:
1. JAWAB HANYA berdasarkan informasi dari \"DOKUMEN INTERNAL\" yang saya berikan. Jangan gunakan pengetahuan dari luar:\n{context}.
2. Selalu format jawaban Anda menggunakan Markdown (seperti heading, list, bold).
3. PENTING: Jangan pernah menggunakan format tabel Markdown (| --- |). Jika Anda menemukan informasi dalam bentuk tabel, jelaskan perbandingannya dalam bentuk daftar (bullet points) atau paragraf naratif.
4. Jika topik pertanyaan sama sekali tidak ada di dalam dokumen, jawab dengan: 'Maaf, informasi tersebut tidak ditemukan dalam knowledge base saya

Format output WAJIB:
- Berikan output **HANYA** dalam format JSON valid, tanpa teks tambahan.
- Struktur JSON yang wajib digunakan:
{
  "answer": "Teks jawaban ringkas dan profesional; Gunakan numbering untuk merinci poin-poin utama agar mudah dibaca; Gunakan formatting Markdown (seperti _italic_ atau *bold*) untuk menekankan informasi penting; Jika ada catatan atau syarat penting, tambahkan bagian \"Catatan:\" di akhir.",
  "sources": [
    {
      "filename": "NamaFile.pdf",
      "page": 3
    }
  ]
}
"""

qa_system_prompt = _escape_prompt_braces(_qa_system_prompt_template, allow={"context"})

CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a query rewriter for an HR assistant.\n"
                "Your job is to transform an incomplete, short, or ambiguous user question "
                "into a standalone, well-specified question that an HR knowledge base search "
                "engine can use.\n\n"
                "Domain: Human Capital / HR policies, procedures, benefits, and services at KPN Corp.\n"
                "- Use the same language as the user (Indonesian or English).\n"
                "- Use chat history ONLY to resolve references or ellipsis such as:\n"
                "  'itu', 'yang tadi', 'seperti sebelumnya', 'bagaimana jika dua hari', "
                "  'kalau lebih lama?', 'yang cuti itu', etc.\n"
                "- If the latest question is a follow-up that depends on the previous topic, "
                "  expand it into a full standalone question including that topic.\n"
                "- Do NOT answer the question.\n"
                "- Return exactly ONE rewritten question, concise but specific.\n"
            ),
        ),
        (
            "human",
            (
                "Chat history:\n{chat_history}\n\n"
                "Latest user question:\n{input}\n\n"
                "Rewrite the latest user question as ONE standalone search query:"
            ),
        ),
    ]
)
