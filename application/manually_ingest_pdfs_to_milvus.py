# manually_ingest_pdfs_to_milvus.py
from pathlib import Path

from ingest.langchain_milvus_ingest import ingest_pages_to_milvus
from ingest.pdf_with_tables import extract_pdf_content
from modules.utils import _build_document_identifier
from config_loader import load_config

folder_path = Path("../assets")
config = load_config()

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
        collection_name=config.collection_name,
        config=config,
        milvus_host=config.milvus_host,
        milvus_port=config.milvus_port,
        collection_description=config.collection_description,
    )

    print(f"[{pdf_path.name}] inserted={inserted} skipped={skipped}")
