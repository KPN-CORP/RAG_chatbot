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

  - Any page that contains a table now triggers the Gemini image extractor so
    the entire page is parsed into header-aware, cell-level chunks **and**
    narrative chunks. Gemini is asked to produce the atomic "cell as a chunk"
    output directly (with `table_index`, headers, row identifiers, and notes),
    so no manual chunking step is needed. Extracted pages are memoized so
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
      Gemini first.

  - The Gemini extractor now guards against mixed text/JSON responses and
    normalizes rows to the same list-of-dicts shape returned by the manual
    parser while also returning prebuilt cell/narrative chunks. Temporary
    images created via `pdf2image` are cleaned up immediately after parsing.
    Structured logs now include the PDF name and page so you can trace which
    document a Gemini request belongs to while debugging ingest runs.

    - Blank table cells are coerced to empty strings instead of `null` so the
      Pydantic schema accepts every cell chunk without validation failures.

  - Control Gemini invocation with the `use_gemini` flag on
    [`extract_pdf_content`](application/ingest/pdf_with_tables.py). Accepted
    values are `"use"` (always call Gemini when a table is detected), `"skip"`
    (force the legacy parsers), or `"fallback"` (only call Gemini for pages that
    need it based on structure). The default is `"use"`, which keeps chunk
    formats consistent between Gemini-produced chunks and those built from the
    manual pipeline.

  - Gemini now returns ready-made `chunks` alongside normalized table rows. The
    ingest layer accepts these directly when present so you can feed Gemini
    output into Milvus without re-chunking. Each chunk already contains enriched
    `chunk_text` with title chains and relevant notes for downstream RAG usage.

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
          "You receive exactly one document page at a time (image or rendered page). Your job is NOT to mechanically extract tables. "
          "Your job is to read, understand, and segment the page into the best possible set of useful information chunks.\n\n"
          "STRICT NON-HALLUCINATION RULES\n\n"
          "You MUST only output information that is explicitly visible on the page.\n\n"
          "- Never invent, infer, or correct policy content, numbers, dates, or ranges.\n"
          "- Never complete partially visible patterns based on prior or outside knowledge.\n"
          "- If some text or number is unreadable or too blurry, write \"[UNREADABLE]\" instead of guessing.\n"
          "- If you are not sure whether a value belongs to one column or another, mark the ambiguous part as \"[UNSURE]\" or skip that chunk entirely.\n"
          "- If a cell appears empty, do not fill it with \"-\", \"0\", or any assumed value unless that exact symbol is present.\n"
          "- Do not summarize, paraphrase, or expand the policy. Copy the wording as literally as possible.\n"
          "- If you are unsure whether a chunk’s content is truly present on the page, it is safer to SKIP that chunk than to guess.\n\n"
          "CHUNKING RULES\n\n"
          "Each chunk must:\n"
          "- Express one clear piece of information that can stand alone (semantically atomic).\n"
          "- Include the title(s) of the page or section and any relevant notes (so it is self-contained).\n"
          "- Be useful for retrieval and answering questions later.\n\n"
          "You must handle complex layouts, including:\n"
          "- Simple or messy tables, multi-level headers, merged cells, nested tables, matrices, or non-grid layouts.\n"
          "- Narrative text, bullets, numbered lists, and footnotes outside tables.\n\n"
          "1. Title chain:\n"
          "   - Detect the main title and any subtitles or section titles.\n"
          "   - For every chunk, include this title chain.\n\n"
          "2. Notes:\n"
          "   - Detect notes/footnotes/legends (for example *, 1), 2), “Catatan”, “Note”).\n"
          "   - If a note clearly applies to some information, include its text in the chunk.\n\n"
          "3. Focus on INFORMATION, not structure:\n"
          "   - Keep policy statements, conditions, ranges, durations, entitlements, monetary values, percentages, eligibility rules, procedures, etc.\n"
          "   - Ignore empty cells, pure layout labels, row/column indices, decorative elements, or “CONTROLLED COPY”-type stamps unless they change policy meaning.\n"
          "   - Ignore “No” columns or row numbers if they have no semantic meaning.\n\n"
          "4. Tables:\n"
          "   - First understand the table logically (headers and what they mean).\n"
          "   - Create chunks that represent meaningful rules, not raw grid cells.\n"
          "   - Example: the row “Lembur di Hari Kerja, ≥2–5 jam, Gol.2 = Sesuai UU, Gol.3 = 96.000, Gol.4 = 126.000” should be one chunk describing that rule, not four separate chunks.\n"
          "   - If different cells in a row clearly express separate rules, you may split them, but never in a way that loses meaning.\n\n"
          "5. Narrative text outside tables:\n"
          "   - Each coherent paragraph, bullet list, or numbered list becomes one chunk if it expresses a single concept or procedure.\n"
          "   - If a paragraph contains several distinct rules, you may split it into multiple chunks, but do not cut sentences in half.\n\n"
          "6. Language and fidelity:\n"
          "   - Preserve the original language, numbers, symbols, ranges, and units exactly (for example “≥ 2 s.d. ≤ 5 jam”, “Rp60.000,00”).\n"
          "   - Do not rewrite in your own words; keep the meaning faithful to the original.\n"
          "   - If OCR is uncertain, write your best guess, but never add information that is not there.\n\n"
          "CHUNK TEXT FORMAT\n\n"
          "For each chunk, output a single string that combines (if present):\n"
          "- The title chain.\n"
          "- A short semantic context (for example a section name, row label, or condition).\n"
          "- The actual rule or information.\n"
          "- Any relevant notes.\n\n"
          "Use a consistent pattern:\n\n"
          "\"<Title chain> | <Semantic context> | <Information> | Notes: <note text or '-'>\"\n\n"
          "Examples of semantic context:\n"
          "- \"Lembur di Hari Kerja, ≥2–5 jam, Gol.2–4\"\n"
          "- \"Pengajuan lembur\"\n"
          "- \"Durasi lembur\"\n"
          "- \"Perhitungan realisasi jam kerja lembur\"\n\n"
          "OUTPUT FORMAT\n\n"
          "Your final output must be ONLY a JSON array of strings.\n"
          "Each string is one chunk text as described above.\n"
          "Do not include any objects, keys, metadata, comments, or explanations.\n"
          "Do not add any text before or after the JSON.\n\n"
          "Example output shape (structure only):\n\n"
          "[\n"
          "  \"KOMPENSASI KELEBIHAN JAM KERJA DI HEAD OFFICE – LAMPIRAN | Lembur di Hari Kerja, ≥2–5 jam, Gol.2–4 | Gol.2: Sesuai UU; Gol.3: 96.000; Gol.4: 126.000 | Notes: -\",\n"
          "  \"KOMPENSASI KELEBIHAN JAM KERJA DI HEAD OFFICE – LAMPIRAN | Durasi lembur | Durasi < 30 menit tidak dihitung sebagai jam kerja lembur | Notes: -\"\n"
          "]\n\n"
          "BEFORE OUTPUT\n\n"
          "Before returning the JSON array, check each candidate chunk:\n"
          "- Does it contain real information (not just structure)?\n"
          "- Can it stand alone and still make sense?\n"
          "- Does it include the title chain and any relevant notes?\n"
          "- Is it faithful to the original page?\n\n"
          "If the answer is yes, include it. If not, adjust or discard the chunk.\n\n"
          "The JSON array you output must be freshly generated from the current page.\n"
          "Do not copy any example chunks from this prompt.\n"
          "Do not include any explanation, comments, code fences, or extra text before or after the JSON array.\n"
          "{EXAMPLE_OUTPUT_FORMAT}"
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

The hooks now live in [`application/modules/milvus_hooks.py`](application/modules/milvus_hooks.py) so you can swap them out or
adjust their metadata filter (via `SOURCE_METADATA_KEY`) if your Milvus schema differs from the default defined in
`ingest/langchain_milvus_ingest.py`. They reuse the existing ingestion pipeline (`ingest_pages_to_milvus` and
`ingest_text_document`) and log which sources currently exist in the knowledge base after each update, making it easy to
verify what documents are indexed.

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
