import os
from dotenv import load_dotenv

LOG_LEVEL = "INFO" # options: DEBUG, INFO, WARNING, ERROR, CRITICAL

load_dotenv()

redis_password = os.environ.get("REDIS_PASSWORD")
redis_password =  redis_password+"@" if redis_password else ''

milvus_host = "127.0.0.1"
milvus_port = "19530"
collection_name = "company_policies"  # collection name in Milvus
collection_description = "Vector store of company policy documents and HR-related content. with Gemini embeddings"  # optional description

REDIS_URL = f"redis://{redis_password}localhost:6379/0"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # optional TTL for session keys
MAX_HISTORY_MESSAGES = 20  # how many recent messages to load into memory


gemini_chat_model = "models/gemini-2.5-flash"  # pick the chat model you want
embedding_model_name = "models/gemini-embedding-001"  # embedding model for Gemini
semantic_embedding_model_name = "LazarusNLP/all-indo-e5-small-v4"  # embedding model for semantic search

top_k_retrieval = 7

google_api_key_name = "GEMINI_API_KEY"  # environment variable name for Gemini/Google API key

qa_system_prompt = """Nama kamu adalah Nastari. Kamu adalah Assistant Human Capital Officer di KPN Corp
Bersikaplah seperti HR profesional yang membantu karyawan dengan pertanyaan terkait kebijakan perusahaan dan masalah karyawan dengan ramah, profesional, dan empatik. 

Aturan utama:
1. Gunakan hanya dokumen yang diekstrak berikut untuk menjawab pertanyaan karyawan:\n{context}.
2. Jika kamu tidak mengetahui jawabannya dari dokumen, jawab dengan sopan: 
   - Katakan maaf apabila tidak yakin mengetahui jawaban dari pertanyaan user.
   - Jangan membuat-buat jawaban.
   - Kamu boleh meminta detail tambahan sesuai dokumen yang diketahui apabila belum lengkap
3. Selalu sertakan sumber dokumen pada seksi "sources" jika jawabanmu berdasarkan dokumen.

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

