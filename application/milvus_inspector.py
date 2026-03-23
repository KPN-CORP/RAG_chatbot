"""Utility to inspect Milvus collections and preview ingested documents.

Run this module directly to list collections and sample the stored metadata.
It uses the existing config/profile loading helper so it stays aligned with
how the FastAPI service connects to Milvus.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


def connect_to_milvus(host: str, port: str) -> None:
    """Connect to the Milvus instance using the provided host/port."""

    logger.info("Connecting to Milvus", extra={"host": host, "port": port})
    connections.connect(alias="default", host=host, port=port)


def list_collections() -> List[str]:
    """Return all available collection names from the connected Milvus."""

    collections = utility.list_collections()
    logger.info("Found Milvus collections", extra={"count": len(collections)})
    return collections


def summarize_sources(records: Iterable[Dict], *, max_items: int) -> Tuple[Counter, List[Dict]]:
    """Summarize source metadata and return a limited set of sample records.

    The function counts how many times each source appears in the result set
    and also returns the original records (trimmed to `max_items`) so callers
    can show the associated text/metadata.
    """

    source_counter: Counter = Counter()
    samples: List[Dict] = []

    for record in records:
        metadata = record.get("metadata") or {}
        source = metadata.get("source") or "<unknown source>"
        source_counter[source] += 1
        if len(samples) < max_items:
            samples.append(record)

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
    if sources:
        print("Document sources (sample counts):")
        for source, count in sources.most_common():
            print(f"  - {source}: {count} sample chunk(s)")
    else:
        print("No source metadata found in the sample set.")

    if samples:
        print("\nSample documents:")
        for sample in samples:
            metadata = sample.get("metadata") or {}
            source = metadata.get("source") or "<unknown>"
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
        print("sample_limit", args.sample_limit, "source_limit", args.source_limit)
        show_collection_overview(collection_name, args.sample_limit, args.source_limit)


if __name__ == "__main__":
    main()
