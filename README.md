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
   or for deployment:
   nohup /root/miniconda3/envs/rag-chatbot/bin/python /root/miniconda3/envs/rag-chatbot/bin/uvicorn app:app --reload --host 0.0.0.0 --port 5003 --access-log --log-level info > logs/app.log 2>&1 &
   ```

   The service exposes the following key endpoints:
   - `POST /session` – create a new Redis-backed chat session.
   - `POST /ask` – run the RAG pipeline for a question.
   - `POST /ingest/upload` – add a PDF to Milvus for retrieval.

3. Confirm the server is running by visiting the interactive docs: `http://localhost:8000/docs`.

### Profiles and config selection

- All chatbot endpoints accept an optional `profile` query/header parameter (for example, `POST /ask?profile=IRIS`). Leaving it empty keeps the legacy default behaviour and loads `application/config.py`.
- `POST /ask` and `POST /ingest/upload` accept an optional `business_unit` query parameter to scope reads/writes to the corresponding `"{collection_name}_{business_unit}"` Milvus collection. Spaces or punctuation in the business unit are normalized to underscores for Milvus-safe collection names.
- Valid values for `business_unit` are defined in `BUSINESS_UNITS` inside `application/config.py`.
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
   curl -X POST "http://localhost:8000/ingest/upload?business_unit=Plantations" \
     -H "X-Ingest-Token: $INGEST_UPLOAD_TOKEN" \
     -F "file=@/path/to/document.pdf"
   ```

   The service extracts tables and text using `ingest/pdf_with_tables.py`, generates Gemini embeddings, and stores chunks in Milvus. Duplicate chunks (same checksum) are skipped automatically.

   - If `pdfplumber` cannot read a page because the PDF is image-only/scanned, the ingest flow now falls back to AWS Textract. The PDF is uploaded to the S3 bucket configured by `S3_BUCKET_NAME`, Textract `LINE` blocks are stitched back into per-page narrative text, and downstream chunking continues as usual. Make sure `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and `S3_BUCKET_NAME` are available in your environment before ingesting scanned PDFs.

  - Any page that contains a table now triggers the Gemini image extractor so
    the entire page is parsed into atomic chunk strings **and** narrative
    chunks. Gemini is asked to output a simple JSON array of chunk strings so
    no manual chunking step is needed. Extracted pages are memoized so
    multiple tables on the same page do not trigger repeated model calls. You
    can still call the helper directly if you want to fine-tune the prompt:

    - Single-row heading banners that only contain a title are reclassified as
      narrative text so they remain searchable even when `pdfplumber` reports
      them as tiny tables.

    - Paragraph-heavy rows (e.g., long multi-line definition blocks) are not
      treated as stacked headers, so a short title row above a prose block no
      longer triggers the complex-table/Gemini path.

    - Two-column definition/procedure layouts with one skinny label column and
      long-form descriptions in the other still stay on the manual parser path
      when **no** table is detected, but once a table is present Gemini will
      process the full page to keep headers connected to notes.

    - Full-page framed definition/procedure boxes (1–2 columns, lots of text)
      are reclassified as narrative so their content is not lost to the table
      filter, but pages that include any true table will be routed through
      Gemini first. When ``use_gemini="use"``, this reclassification is skipped
      so framed paragraph blocks still reach Gemini before the manual fallback
      parser runs.

  - The Gemini extractor now guards against mixed text/JSON responses and
    normalizes string-only outputs so ingest can proceed even when the model
    omits table metadata. Temporary images created via `pdf2image` are cleaned
    up immediately after parsing. Structured logs now include the PDF name and
    page so you can trace which document a Gemini request belongs to while
    debugging ingest runs.

    - Blank table cells are coerced to empty strings instead of `null` so the
      Pydantic schema accepts every cell chunk without validation failures.

  - Control Gemini invocation with the `use_gemini` flag on
    [`extract_pdf_content`](application/ingest/pdf_with_tables.py). Accepted
    values are `"use"` (always call Gemini when a table is detected), `"skip"`
    (force the legacy parsers), or `"fallback"` (only call Gemini for pages that
    need it based on structure). The default is `"use"`, which keeps chunk
    formats consistent between Gemini-produced chunks and those built from the
    manual pipeline.

  - Gemini now returns ready-made chunk strings. The ingest layer accepts these
    directly when present so you can feed Gemini output into Milvus without
    re-chunking. Each chunk is wrapped with the document/page metadata during
    ingest to preserve provenance for downstream RAG usage.

  - Gemini-parsed pages still flow through the semantic/atomic page walker so
    headers or topic boundaries on Gemini-only pages update the carry-over
    context for subsequent pages.

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
          "You are an OCR + Vision-Language assistant used in a Retrieval-Augmented Generation (RAG) system.\n\n"
         ........
      ),
      api_key="${GOOGLE_API_KEY}",
  )
   # result["rows"] and result["chunks"] are validated by a Pydantic schema and
   # ready to ingest
   print(result["rows"])
   PY
   ```

3. **Ask a question**

   ```bash
   curl -X POST "http://localhost:8000/ask?business_unit=Plantations" \
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
- `modules/watcher.py` – folder watcher that mirrors files into Milvus; when files live in subfolders of `WATCH_DIR`, their vectors are written to `{collection_name}_{subfolder}` collections (subfolder names are normalized to Milvus-safe suffixes).

  ```bash
  # Inspect every available collection using the default config
  python application/milvus_inspector.py

  # Look at a single collection with a custom profile and fetch up to 5 rows
  python application/milvus_inspector.py --profile IRIS --collection company_policies --sample-limit 5
  ```

Refer to the module docstrings for advanced usage.

### Milvus collection schema

Ingested chunks are stored in Milvus collections created by
[`ingest/langchain_milvus_ingest.py`](application/ingest/langchain_milvus_ingest.py).
The schema is intentionally small to keep ingestion fast while retaining rich
metadata in a JSON field:

- `id` (`VARCHAR`, primary key, max length 128): chunk identifier (`doc_id` +
  positional suffix).
- `embedding` (`FLOAT_VECTOR`): Gemini embedding for the chunk text.
- `checksum` (`VARCHAR`, max length 128): deduplication key used to skip chunks
  already present in the collection.
- `text` (`VARCHAR`, max length 65535): raw chunk text used for hybrid search
  and debugging.
- `metadata` (`JSON`): structured attributes such as `source`, `page`,
  `element_type`, `table_id`/`row_id`/`list_id`, `breadcrumb`, and any Q&A
  fields. The ingestion step strips `chunk_text`/`text` from this blob to avoid
  duplication while preserving identifiers and breadcrumbs for result displays.

Collections inherit the description from `config.collection_description` and use
an HNSW/COSINE index by default. You can override the collection name, host, or
port in your config module before running ingestion.

## 6. Folder watcher for automatic Milvus sync

The watcher under [`application/modules/watcher.py`](application/modules/watcher.py) observes the `WATCH_DIR` configured in
`application/config.py` (default: `assets/`). Its responsibilities are deliberately narrow:

- detect file system changes (create/modify/delete/move) for supported extensions in `VALID_EXTS`;
- call two hook functions, `upsert_file_to_milvus(file_path)` and `delete_file_from_milvus(file_path)`, to keep Milvus in sync.
- update the status column in the table named by `KB_DB_TABLE` when a document begins processing (`processing`) and finishes (`done`) for both upserts and deletes, using the subfolder name as the `business_unit` lookup when available.

The hooks now live in [`application/modules/milvus_hooks.py`](application/modules/milvus_hooks.py) so you can swap them out or
adjust their metadata filter (via `SOURCE_METADATA_KEY`) if your Milvus schema differs from the default defined in
`ingest/langchain_milvus_ingest.py`. They reuse the existing ingestion pipeline (`ingest_pages_to_milvus` and
`ingest_text_document`) and log which sources currently exist in the knowledge base after each update, making it easy to
verify what documents are indexed. Deletes are defensive: they match both the filename and full file path across
`metadata.source`, `metadata.filename`, `metadata.path`, and `metadata.file_name` so older ingestion runs using different
keys or full-path values are cleaned up reliably.

To enable status updates, set `KB_DATABASE_URL` in your environment (for example in a `.env` file) alongside the configured
`KB_DB_TABLE` value in `application/config.py`. The status lookup uses `REGEXP_REPLACE`, so ensure your MySQL/MariaDB
version supports it (MySQL 8+), or adjust the normalization helpers in `application/modules/kb_status.py` for older
engines.

Run the watcher from the `application/` directory. You can pass `--profile` to
target a specific chatbot configuration (for example `config_IRIS.py`):

```bash
cd application
python -m modules.watcher --profile IRIS
```

To keep the watcher running in the background with log output, use `nohup` and
redirect stdout/stderr to a file:

```bash
cd application
nohup python -m modules.watcher --profile IRIS > logs/watcher.log 2>&1 &
```

Stop it with `Ctrl+C` (or terminate the background PID). The debounce worker
ensures only stable files (unchanged for a couple of seconds) are ingested to
avoid partial uploads.

## 7. Troubleshooting

| Issue | Suggested action |
|-------|------------------|
| `LLM API key not configured` | Ensure `GEMINI_API_KEY` (or the configured variable name) is exported before starting the server. |
| `Vectorstore not wired in server` | Verify Milvus is running, accessible, and initialized via `vectorstore/milvus_vectorstore.py`. |
| `Invalid ingestion token` | Double-check the `X-Ingest-Token` header matches the server-side `INGEST_UPLOAD_TOKEN`. |
| Redis or Milvus connection errors | Update host/port settings in `application/config.py` or corresponding environment variables. |

---

With the above steps, you can run, stop, and interact with the Nastari RAG assistant shipped in the `application/` package.
