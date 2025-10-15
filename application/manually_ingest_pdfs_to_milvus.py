# manually_ingest_pdfs_to_milvus.py 
from ingest.langchain_milvus_ingest import ingest_pages_to_milvus
from ingest.pdf_with_tables import extract_pdf_content
from modules.utils import _build_document_identifier
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
folder_path = Path('../assets')
from config import (
    milvus_host,
    milvus_port,
    embedding_model_name,
    semantic_embedding_model_name,  
    collection_description,
    google_api_key_name,
    collection_name
)
api_key = os.getenv(google_api_key_name)  


for pdf_path in folder_path.glob("*.pdf"):
    # 1) Read file bytes (this replaces `data = await file.read()` from your API route)
    data = pdf_path.read_bytes()
    print("Processing", pdf_path.name)
    # 2) Build a stable document ID the same way your API does
    doc_id = _build_document_identifier(pdf_path.name or "document.pdf", data)

    # 3) Extract pages (adjust if your extractor supports bytes instead of path)
    pages = extract_pdf_content(str(pdf_path))

    # 4) Ingest
    skipped, inserted = ingest_pages_to_milvus(
        pages=pages,
        doc_id=doc_id,
        source=pdf_path.name or doc_id,
        collection_name=collection_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        embedding_model_name=embedding_model_name,
        semantic_embedding_model_name=semantic_embedding_model_name,
        api_key=api_key or None,
        collection_description=collection_description,
    )

    print(f"[{pdf_path.name}] inserted={inserted} skipped={skipped}")
    