"""Utility to inspect Milvus collections and preview ingested documents.

Run this module directly to list collections and sample the stored metadata.
It uses the existing config/profile loading helper so it stays aligned with
how the FastAPI service connects to Milvus.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config_loader import load_config
from logging_config import get_logger
from pymilvus import Collection, connections, utility

logger = get_logger(__name__)

# Default number of rows to fetch when showing document samples so the command
# finishes quickly even on large collections.
DEFAULT_SAMPLE_LIMIT = 10
# Default number of rows to scan when summarizing document sources.
DEFAULT_SOURCE_LIMIT = 200
# Default number of unique documents to list per collection.
DEFAULT_DOCUMENT_LIMIT = 20


def connect_to_milvus(host: str, port: str) -> None:
    """Connect to the Milvus instance using the provided host/port."""

    logger.info("Connecting to Milvus", extra={"host": host, "port": port})
    connections.connect(alias="default", host=host, port=port)


def list_collections() -> List[str]:
    """Return all available collection names from the connected Milvus."""

    collections = utility.list_collections()
    logger.info("Found Milvus collections", extra={"count": len(collections)})
    return collections


def unwrap_source(record: Dict) -> str:
    """Return the best available source label from a Milvus record."""

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        return "<unknown source>"

    raw_source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("file_name")
        or metadata.get("path")
    )
    if not raw_source:
        return "<unknown source>"

    return Path(str(raw_source)).name or str(raw_source)


def summarize_sources(records: Iterable[Dict], *, max_items: int) -> Tuple[Counter, List[Dict]]:
    """Summarize source metadata and return one sample record per source."""

    source_counter: Counter = Counter()
    samples: List[Dict] = []
    sampled_sources: set[str] = set()

    for record in records:
        source = unwrap_source(record)
        source_counter[source] += 1
        if len(samples) >= max_items or source in sampled_sources:
            continue
        samples.append(record)
        sampled_sources.add(source)

    return source_counter, samples


def fetch_collection_records(collection: Collection, limit: int) -> List[Dict]:
    """Fetch a small number of records from the given collection for inspection."""

    if limit <= 0:
        return []

    return collection.query(
        expr="id != ''",  # basic filter to return all records
        output_fields=["id", "text", "metadata"],
        limit=limit,
    )


def show_collection_overview(
    collection_name: str,
    sample_limit: int,
    source_limit: int,
    document_limit: int,
) -> None:
    """Log and print a brief overview of the given collection."""

    collection = Collection(collection_name)
    collection.load()

    logger.info(
        "Inspecting collection", extra={"collection": collection_name, "entities": collection.num_entities},
    )

    source_records = fetch_collection_records(collection, source_limit)
    sources, samples = summarize_sources(source_records, max_items=sample_limit)

    print(f"\nCollection: {collection_name}")
    print(f"Total entities: {collection.num_entities}")
    print(f"Unique document sources in scanned rows: {len(sources)}")
    print(f"Rows scanned for source summary: {len(source_records)}")
    if sources:
        shown_document_count = min(len(sources), max(document_limit, 0))
        print(f"Documents detected (showing up to {shown_document_count} of {len(sources)}):")
        for source, count in sources.most_common(max(document_limit, 0)):
            print(f"  - {source}: {count} chunk(s)")
    else:
        print("No source metadata found in the sample set.")

    if samples:
        print("\nSample documents:")
        for sample in samples:
            metadata = sample.get("metadata") or {}
            source = unwrap_source(sample)
            page = metadata.get("page")
            snippet = (sample.get("text") or "").replace("\n", " ")
            snippet = (snippet[:120] + "...") if len(snippet) > 120 else snippet
            page_info = f" (page {page})" if page is not None else ""
            print(f"- {sample.get('id')}: {source}{page_info}")
            if snippet:
                print(f"  text: {snippet}")
    else:
        print("\nNo sample records returned from the collection.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the inspection utility."""

    parser = argparse.ArgumentParser(
        description="List Milvus collections and preview ingested documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Config profile to load (maps to config_<PROFILE>.py).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Inspect only this collection instead of all available ones.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help="Number of records to fetch per collection for inspection.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=DEFAULT_SOURCE_LIMIT,
        help="Number of records to scan when summarizing document sources.",
    )
    parser.add_argument(
        "--document-limit",
        type=int,
        default=DEFAULT_DOCUMENT_LIMIT,
        help="Maximum number of unique documents to list per collection.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for the CLI inspection tool."""

    args = parse_args()
    config = load_config(args.profile)

    connect_to_milvus(config.milvus_host, str(config.milvus_port))

    target_collections = [args.collection] if args.collection else list_collections()
    if not target_collections:
        print("No collections found in Milvus.")
        return

    for collection_name in target_collections:
        if not utility.has_collection(collection_name):
            logger.warning("Collection missing in Milvus", extra={"collection": collection_name})
            continue
        show_collection_overview(collection_name, args.sample_limit, args.source_limit, args.document_limit)


if __name__ == "__main__":
    main()
