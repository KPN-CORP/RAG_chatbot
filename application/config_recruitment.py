import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

LOG_LEVEL = "INFO" # options: DEBUG, INFO, WARNING, ERROR, CRITICAL

load_dotenv()

redis_password = os.environ.get("REDIS_PASSWORD")
redis_password =  redis_password+"@" if redis_password else ''

milvus_host = "127.0.0.1"
milvus_port = "19530"
collection_name = "recruitment"  # collection name in Milvus
collection_description = "Vector store of recruitment-related documents and HR content. with Gemini embeddings"  # optional description

REDIS_URL = f"redis://{redis_password}localhost:6379/2"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # optional TTL for session keys
MAX_HISTORY_MESSAGES = 20  # how many recent messages to load into memory


gemini_chat_model = "models/gemini-2.5-flash-lite"  # pick the chat model you want
gemini_image_model = "models/gemini-2.5-flash" 
# When True, bilingual two-column pages keep only the left column when the
# detected right-column language is non-Indonesian.
enable_multilingual_column_filter = True
embedding_model_name = "models/gemini-embedding-001"  # embedding model for Gemini
semantic_embedding_model_name = "LazarusNLP/all-indo-e5-small-v4"  # embedding model for semantic search

top_k_retrieval = 7

google_api_key_name = "GEMINI_API_KEY"  # environment variable name for Gemini/Google API key

WATCH_DIR = r"/home/hcispanel/kpn-recruitment/storage/app/public/knowledge_docs/"  # change this

VALID_EXTS = {".pdf", ".docx", ".txt"}  # adapt to your project
KB_DB_TABLE = "kb_documents"
# URL for the knowledge base status database (e.g. mysql+pymysql://user:pass@host:3306/db)
KB_DATABASE_URL = os.environ.get("KB_DATABASE_URL")

def _escape_prompt_braces(prompt: str, *, allow: set[str]) -> str:
    """Escape all braces except those explicitly allowed as template vars."""

    escaped = prompt.replace("{", "{{").replace("}", "}}")
    for key in allow:
        escaped = escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}")
    return escaped


_qa_system_prompt_template = """Nama kamu adalah Nastari. Kamu adalah Assistant Human Capital Recruiter di KPN
Bersikaplah seperti HR profesional yang membantu karyawan dengan pertanyaan terkait kebijakan dan informasi perusahaan dengan ramah, profesional, dan empatik.

Aturan utama:
1. Gunakan hanya dokumen yang diekstrak berikut untuk menjawab pertanyaan karyawan:\n{context}.
2. Jika kamu tidak mengetahui jawabannya dari dokumen, jawab dengan sopan:
   - Jangan membuat-buat jawaban.
   - Kamu boleh meminta detail tambahan sesuai dokumen yang diketahui apabila belum lengkap
3. Selalu sertakan sumber dokumen pada seksi "sources" jika jawabanmu berdasarkan dokumen.

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
                "Domain: Human Capital / HR policies, procedures, benefits, and services at KPN.\n"
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
