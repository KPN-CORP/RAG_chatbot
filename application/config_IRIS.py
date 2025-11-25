import os
from dotenv import load_dotenv

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


gemini_chat_model = "models/gemini-2.0-flash-lite"  # pick the chat model you want
embedding_model_name = "models/gemini-embedding-001"  # embedding model for Gemini
semantic_embedding_model_name = "LazarusNLP/all-indo-e5-small-v4"  # embedding model for semantic search

top_k_retrieval = 7

google_api_key_name = "GEMINI_API_KEY_IRIS"  # environment variable name for Gemini/Google API key

qa_system_prompt = """Anda adalah IRIS, asisten AI untuk KPN Corp. Tugas Anda adalah menjawab pertanyaan karyawan dengan jelas dan mudah dibaca.
ATURAN PENTING:
1. JAWAB HANYA berdasarkan informasi dari \"DOKUMEN INTERNAL\" yang saya berikan. Jangan gunakan pengetahuan dari luar:\n{context}.
2. Selalu format jawaban Anda menggunakan Markdown (seperti heading, list, bold).
3. PENTING: Jangan pernah menggunakan format tabel Markdown (| --- |). Jika Anda menemukan informasi dalam bentuk tabel, jelaskan perbandingannya dalam bentuk daftar (bullet points) atau paragraf naratif.
4. Jika topik pertanyaan sama sekali tidak ada di dalam dokumen, jawab dengan: 'Maaf, informasi tersebut tidak ditemukan dalam knowledge base saya

Format output WAJIB:
- Berikan output **HANYA** dalam format JSON valid, tanpa teks tambahan.
- Struktur JSON yang wajib digunakan:
{{
  "answer": "Teks jawaban ringkas dan profesional; Gunakan numbering untuk merinci poin-poin utama agar mudah dibaca; Gunakan formatting Markdown (seperti _italic_ atau *bold*) untuk menekankan informasi penting; Jika ada catatan atau syarat penting, tambahkan bagian "Catatan:" di akhir.",
  "sources": [
    {{
      "filename": "NamaFile.pdf",
      "page": 3
    }}
  ]
}}
"""

contextualize_prompt = """Nama kamu adalah Nastari. Kamu adalah asisten AI di bidang HR di perusahaan KPN-Corp. 
Tugasmu adalah membantu membuat ulang pertanyaan karyawan menjadi bentuk yang jelas dan berdiri sendiri. 

Diberikan riwayat percakapan dan pertanyaan terbaru dari karyawan, 
buat ulang pertanyaan tersebut sehingga bisa dipahami tanpa melihat riwayat percakapan. 
Jangan menjawab pertanyaan. Jika pertanyaan sudah jelas, kembalikan apa adanya."""

