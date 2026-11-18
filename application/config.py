import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

LOG_LEVEL = "INFO" # options: DEBUG, INFO, WARNING, ERROR, CRITICAL

load_dotenv()

redis_password = os.environ.get("REDIS_PASSWORD")
# The password goes after ":" — "redis://secret@host" would send it as a USERNAME.
redis_password = f":{redis_password}@" if redis_password else ""

milvus_host = "127.0.0.1"
milvus_port = "19530"
collection_name = "company_policies"  # collection name in Milvus
collection_description = "Vector store of company policy documents and HR-related content. with Gemini embeddings"  # optional description

REDIS_URL = os.environ.get("REDIS_URL") or f"redis://{redis_password}localhost:6379/0"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # optional TTL for session keys
MAX_HISTORY_MESSAGES = 20  # how many recent messages to load into memory


gemini_chat_model = "models/gemini-flash-latest"  # pick the chat model you want
gemini_image_model = "models/gemini-flash-latest"
# Sampling temperature for the chat LLM (question rewriting + answering).
# 0 = deterministic and precise, which is what HR policy Q&A needs (consistent
# answers, no hedging). Read by app.py via getattr so every profile inherits it.
llm_temperature = 0
# When True, OCR-derived pages are hierarchically chunked with Gemini before ingest.
use_gemini_chunking = True
# When True, bilingual two-column pages keep only the left column when the
# detected right-column language is non-Indonesian.
enable_multilingual_column_filter = True
embedding_model_name = "models/gemini-embedding-001"  # embedding model for Gemini
semantic_embedding_model_name = "LazarusNLP/all-indo-e5-small-v4"  # embedding model for semantic search

# Number of chunks retrieved per query. Retrieval runs for both the original and
# the rewritten question and merges unique results, so effective context can be
# larger. Kept moderate to balance recall against token cost.
top_k_retrieval = 10

# --- Chunking parameters (semantic grouping of policy rules) ---
# Cosine-similarity threshold for merging consecutive lines into one atomic rule.
# Higher (→1.0) = more fragmentation (smaller chunks); lower = more merging.
chunk_sim_threshold = 0.82
# Split any atomic rule longer than this many characters via SemanticChunker.
chunk_semantic_fallback_max_chars = 1000
# Enable the long-rule SemanticChunker fallback.
chunk_enable_semantic_fallback = True
# Use an adaptive per-page percentile threshold instead of the fixed one above.
chunk_use_percentile = False
chunk_percentile = 90

google_api_key_name = "GEMINI_API_KEY"  # environment variable name for Gemini/Google API key

WATCH_DIR = os.environ.get("WATCH_DIR") or r"/home/hcispanel/chatbot-nastari/storage/app/pdf-file/"  # override via WATCH_DIR env

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


def build_qa_system_prompt(business_unit: Optional[str] = None) -> str:
    """Build the QA system prompt for the active business unit.

    The /ask endpoint passes ``req.business_unit`` into this helper so the
    assistant persona can mention the caller's business unit directly in the
    system prompt while still preserving the ``{context}`` LangChain variable.
    When the request omits the field, we keep the legacy default persona text.
    """

    cleaned_business_unit = (business_unit or "").strip()
    company_scope = f"di KPN BU {cleaned_business_unit}" 
    prompt_template = _qa_system_prompt_template.replace("{company_scope}", company_scope)
    return _escape_prompt_braces(prompt_template, allow={"context"})


_qa_system_prompt_template = """Nama kamu adalah Nastari, Assistant Human Capital Officer {company_scope}.
Tugasmu menjawab pertanyaan karyawan seputar kebijakan HR secara PRESISI dan dapat dipercaya.

PRINSIP UTAMA (WAJIB):
1. Jawab HANYA berdasarkan dokumen pada konteks berikut:
{context}
   - DILARANG mengarang, menebak, atau memakai pengetahuan umum di luar dokumen.
   - DILARANG berasumsi. Jangan menyimpulkan hal yang tidak tertulis eksplisit di dokumen.
2. Jika dokumen mendukung, jawab LANGSUNG, TEGAS, dan SPESIFIK:
   - Sebutkan angka, durasi, nominal, syarat, dan nomor kebijakan/pasal secara persis seperti tertulis.
   - Jangan menunda atau bertanya balik jika informasinya sudah tersedia di dokumen.
3. Jika pertanyaan ambigu TETAPI dokumen memuat informasi relevan:
   - Tetap berikan jawaban terbaik berdasarkan dokumen, dan sebutkan kondisi/asumsinya secara eksplisit.
   - Boleh menambahkan SATU pertanyaan klarifikasi singkat di akhir sebagai pelengkap, BUKAN pengganti jawaban.
4. Jika informasi TIDAK ADA di dokumen:
   - Nyatakan dengan jelas: "Hal ini tidak diatur dalam dokumen kebijakan yang tersedia."
   - Sarankan karyawan menghubungi tim HR terkait. Jangan menebak atau melengkapi sendiri.
5. Jika ada perbedaan aturan antar dokumen atau antar unit bisnis:
   - Sebutkan perbedaannya beserta sumber masing-masing. Jangan menggabungkannya menjadi satu jawaban.
6. Sampaikan HANYA isi kebijakan sebagaimana tertulis. DILARANG memberi nasihat atau interpretasi hukum.
7. Selalu sertakan sumber (filename dan page) untuk setiap klaim pada seksi "sources".
8. Gunakan bahasa yang sama dengan penanya, bersikap profesional dan empatik, namun utamakan ketepatan.

Format output WAJIB:
- Berikan output **HANYA** dalam format JSON valid, tanpa teks tambahan.
- Struktur JSON yang wajib digunakan:
{
  "answer": "Jawaban ringkas, tegas, dan spesifik. Gunakan numbering untuk merinci poin-poin utama. Gunakan formatting Markdown (seperti _italic_ atau *bold*) untuk menekankan angka/syarat penting. Jika ada syarat atau catatan penting, tambahkan bagian \"Catatan:\" di akhir.",
  "sources": [
    {
      "filename": "NamaFile.pdf",
      "page": 3
    }
  ]
}
"""

# Default prompt used by flows that do not inject a request-scoped business unit.
qa_system_prompt = build_qa_system_prompt()

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

BUSINESS_UNITS = [
    "KPN Corporation",
    "Cement",
    "Downstream",
    "Plantations",
    "Property",
    "KPN Sugar"
]
