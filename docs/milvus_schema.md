# Milvus collection schema

This project writes all chunks to Milvus via `ingest/langchain_milvus_ingest.py`. The
schema is intentionally compact so inserts stay fast while metadata remains
accessible for result rendering and debugging.

## Fields

| Field | Type | Purpose |
| --- | --- | --- |
| `id` | `VARCHAR` (primary key, max length 128) | Stable chunk identifier built from `doc_id` plus a positional suffix. |
| `embedding` | `FLOAT_VECTOR` | Gemini embedding for the chunk text. Dimension matches the configured embedding model. |
| `checksum` | `VARCHAR` (max length 128) | Dedupe key; existing checksums are queried before inserting a batch. |
| `text` | `VARCHAR` (max length 65,535) | Raw chunk text stored alongside the vector for hybrid search and inspection. |
| `metadata` | `JSON` | Structured attributes such as `source`, `page`, `element_type`, table/list identifiers, breadcrumbs, and Q&A fields. |

Collections use an HNSW + COSINE index by default with parameters `M=16` and
`efConstruction=200`. The description is pulled from
`config.collection_description`, and the connection target (`host`, `port`, and
collection name) comes from the active config module.

## Metadata layout

The JSON metadata mirrors the dictionary produced in `chunks_to_records`. Key
fields include:

- `source`: original file name or URI of the ingested document.
- `page`: 1-based page number for PDF sources.
- `element_type`: high-level chunk type (e.g., `table_cell`, `paragraph`, `qa_pair`).
- `table_id`, `row_id`, `list_id`: positional markers to reconstruct layout.
- `breadcrumb`: hierarchical label path for nested bullets/headers.
- `col_name`, `value_num`: helpers for numeric columns.
- `qa_question`, `qa_answer`: populated for detected question/answer pairs.

Chunk text itself is not duplicated inside `metadata`; it is stored in the `text`
column and removed from the metadata blob just before insertion.

## Example document flow

1. Chunks are built (or accepted from Gemini) with `chunk_id`, `chunk_text`,
   layout markers, and source metadata.
2. Checksums are calculated per chunk. Existing checksums in the target
   collection are skipped to prevent duplicates.
3. Texts are embedded in batches, the collection is created if missing, and the
   vectors plus metadata are inserted.

Use `application/milvus_inspector.py` to preview inserted rows and confirm the
schema matches your Milvus deployment.
