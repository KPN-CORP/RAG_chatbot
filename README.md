# RAG Chatbot Application Guide

This repository contains experimentation notebooks and a production-ready FastAPI service under [`application/`](application/) for a Retrieval Augmented Generation (RAG) HR assistant named **Nastari**. This guide explains how to run, stop, and use the service exposed by the `application/` package.

## 1. Prerequisites

1. **Python environment**
   - Python 3.10+
   - The recommended workflow is to create a Conda environment from the provided [`environment.yml`](environment.yml) file at the repository root.

   ```bash
   conda env create -f environment.yml
   conda activate rag-chatbot
   ```

   - If you prefer to manage packages manually (e.g., in an existing Conda env), install dependencies according to the imports in `application/` (FastAPI, Uvicorn, LangChain, Redis, pymilvus, etc.).

2. **External services**
   - **Redis** running and accessible at the URL configured in [`application/config.py`](application/config.py) (default: `redis://localhost:6379/0`).
   - **Milvus** vector database reachable using `milvus_host` and `milvus_port` from the same config file (default: `127.0.0.1:19530`).

3. **Environment variables**
   - `GEMINI_API_KEY` (or the name specified by `google_api_key_name` in `config.py`) – used by LangChain's Gemini chat & embedding models.
   - `INGEST_UPLOAD_TOKEN` – shared secret required when calling the document upload endpoint.
   - Optionally, override other settings by exporting variables consumed in `application/config.py`.

## 2. Running the FastAPI server

1. Navigate into the `application/` directory.

   ```bash
   cd application
   ```

2. Start the API using Uvicorn (development hot-reload shown below).

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

   The service exposes the following key endpoints:
   - `POST /session` – create a new Redis-backed chat session.
   - `POST /chat` – run the RAG pipeline for a question.
   - `POST /ingest/upload` – add a PDF to Milvus for retrieval.

3. Confirm the server is running by visiting the interactive docs: `http://localhost:8000/docs`.

### Profiles and config selection

- All chatbot endpoints accept an optional `profile` query/header parameter (for example, `POST /ask?profile=IRIS`). Leaving it empty keeps the legacy default behaviour and loads `application/config.py`.
- A profile name picks up a matching config module named `config_{PROFILE}.py` (e.g., `config_IRIS.py`). Missing profiles log a warning and fall back to the default config so existing integrations keep working.
- The `IRIS` profile returns a slimmed down chat payload (`{"answer": ..., "not_found": ...}`) without the `sources` key, while other profiles continue returning `sources` alongside the answer.
- To add a new persona/configuration, drop a file like `application/config_FOOBOT.py` alongside the existing config files using the same variable names.

## 3. Stopping the server

- Press `Ctrl+C` in the terminal where Uvicorn is running to gracefully stop the API.
- If running as a background process (e.g., via systemd, Docker, or `nohup`), stop the process using the respective tooling (`systemctl stop`, `docker stop`, `kill <pid>`, etc.).

## 4. Using the API

Below is a minimal sequence using `curl`. Replace placeholders with your own values.

1. **Create a session**

   ```bash
   curl -X POST http://localhost:8000/session
   # => {"session_id": "<uuid>"}
   ```

2. **Ingest a PDF** (optional but recommended before asking questions)

   ```bash
   curl -X POST "http://localhost:8000/ingest/upload" \
     -H "X-Ingest-Token: $INGEST_UPLOAD_TOKEN" \
     -F "file=@/path/to/document.pdf"
   ```

   The service extracts tables and text using `ingest/pdf_with_tables.py`, generates Gemini embeddings, and stores chunks in Milvus. Duplicate chunks (same checksum) are skipped automatically.

  - Tables with three-header rows **or nested tables inside cells** are treated
    as *complex* and will automatically be re-extracted via the Gemini image
    flow when a Gemini API key is configured. Simpler layouts continue to use
    the built-in parser. One-column, paragraph-like detections are explicitly
    left to the manual parser to avoid unnecessary Gemini calls. Gemini image
    extraction is memoized per page so multiple complex tables on the same
    page do not trigger repeated model calls. You can still call the helper
    directly if you want to fine-tune the prompt:

    - Single-row heading banners that only contain a title are reclassified as
      narrative text so they remain searchable even when `pdfplumber` reports
      them as tiny tables.

    - Paragraph-heavy rows (e.g., long multi-line definition blocks) are not
      treated as stacked headers, so a short title row above a prose block no
      longer triggers the complex-table/Gemini path.

    - The Gemini extractor now guards against mixed text/JSON responses and
      normalizes rows to the same list-of-dicts shape returned by the manual
      parser so downstream chunking never breaks. Temporary images created via
      `pdf2image` are cleaned up immediately after parsing.

  If a table contains complex multi-level headers that render poorly, you can
  render the page to an image and ask Gemini to re-extract the table JSON via
  [`ingest/table_image_extractor.py`](application/ingest/table_image_extractor.py):

   ```bash
   python - <<'PY'
   from application.ingest.table_image_extractor import extract_table_with_gemini

  result = extract_table_with_gemini(
      pdf_path="/path/to/document.pdf",
      page_number=6,
      prompt=(
          "Return the table as a JSON array of rows with every header column. "
          "If headers span multiple rows, combine them with ' - ' (e.g., "
          "'Allowance - Travel - Amount'). If a cell contains a nested table, "
          "capture that subtable as multi-line text inside the cell value."
      ),
      api_key="${GOOGLE_API_KEY}",
  )
   # result["rows"] is validated by a Pydantic schema and ready to ingest
   print(result["rows"])
   PY
   ```

3. **Ask a question**

   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "<uuid-from-session-endpoint>",
       "question": "Apa kebijakan cuti tahunan karyawan?",
       "debug_retrieval": false
     }'
   ```

   The service uses Redis conversation memory, retrieves similar chunks from Milvus, feeds context to Gemini (`models/gemini-2.5-flash` by default), and returns a JSON answer with document sources.

4. **Session expiry**

   - Sessions are stored in Redis with a TTL (`SESSION_TTL_SECONDS` in `config.py`). After expiration, create a new session before sending further questions.

## 5. Maintenance scripts

The `application/` directory also contains utilities that can be invoked manually:

- `clean_project.py` – housekeeping tasks such as clearing caches.
- `modules/redis.py` – helper for managing Redis conversation memory.
- `vectorstore/milvus_vectorstore.py` – wrapper to instantiate the Milvus vector store via LangChain.
- `milvus_inspector.py` – CLI helper to list Milvus collections and preview the ingested document sources/chunks. For example:

  ```bash
  # Inspect every available collection using the default config
  python application/milvus_inspector.py

  # Look at a single collection with a custom profile and fetch up to 5 rows
  python application/milvus_inspector.py --profile IRIS --collection company_policies --sample-limit 5
  ```

Refer to the module docstrings for advanced usage.

## 6. Folder watcher for automatic Milvus sync

The watcher under [`application/modules/watcher.py`](application/modules/watcher.py) observes the `WATCH_DIR` configured in
`application/config.py` (default: `assets/`). Its responsibilities are deliberately narrow:

- detect file system changes (create/modify/delete/move) for supported extensions in `VALID_EXTS`;
- call two hook functions, `upsert_file_to_milvus(file_path)` and `delete_file_from_milvus(file_path)`, to keep Milvus in sync.

The hooks now live in [`application/modules/milvus_hooks.py`](application/modules/milvus_hooks.py) so you can swap them out or
adjust their metadata filter (via `SOURCE_METADATA_KEY`) if your Milvus schema differs from the default defined in
`ingest/langchain_milvus_ingest.py`. They reuse the existing ingestion pipeline (`ingest_pages_to_milvus` and
`ingest_text_document`) and log which sources currently exist in the knowledge base after each update, making it easy to
verify what documents are indexed.

Run the watcher from the `application/` directory:

```bash
cd application
python -m modules.watcher
```

Stop it with `Ctrl+C`. The debounce worker ensures only stable files (unchanged for a couple of seconds) are ingested to avoid
partial uploads.

## 7. Troubleshooting

| Issue | Suggested action |
|-------|------------------|
| `LLM API key not configured` | Ensure `GEMINI_API_KEY` (or the configured variable name) is exported before starting the server. |
| `Vectorstore not wired in server` | Verify Milvus is running, accessible, and initialized via `vectorstore/milvus_vectorstore.py`. |
| `Invalid ingestion token` | Double-check the `X-Ingest-Token` header matches the server-side `INGEST_UPLOAD_TOKEN`. |
| Redis or Milvus connection errors | Update host/port settings in `application/config.py` or corresponding environment variables. |

---

With the above steps, you can run, stop, and interact with the Nastari RAG assistant shipped in the `application/` package.
