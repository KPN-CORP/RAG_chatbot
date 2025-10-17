"""Utilities for listing and deleting Milvus records by their PDF source filename."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from config import collection_name, milvus_host, milvus_port
from logging_config import get_logger
from modules.utils import meta_filename
from pymilvus import Collection, connections, utility

logger = get_logger(__name__)


def _connect(host: str, port: str) -> None:
    """Ensure a Milvus connection exists for the default alias."""

    logger.debug("Connecting to Milvus", extra={"host": host, "port": port})
    connections.connect(alias="default", host=host, port=str(port))


def _load_collection(name: str, host: str, port: str) -> Collection:
    """Return the Milvus collection, raising if it does not exist."""

    _connect(host, port)
    if not utility.has_collection(name):
        logger.error("Milvus collection not found", extra={"collection": name})
        raise RuntimeError(f"Milvus collection '{name}' does not exist")
    collection = Collection(name)
    collection.load()
    return collection


def _normalise_filenames(filenames: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    for raw in filenames:
        if not raw:
            continue
        name = Path(raw).name.strip()
        if not name:
            continue
        seen.add(name)
    return sorted(seen)


def list_document_filenames(
    *,
    collection: str = collection_name,
    host: str = milvus_host,
    port: str = milvus_port,
    batch_size: int = 16_384,
) -> List[str]:
    """Return the unique PDF filenames stored in the Milvus collection."""

    coll = _load_collection(collection, host, port)
    total = coll.num_entities
    if total == 0:
        logger.info("Milvus collection is empty", extra={"collection": collection})
        return []

    filenames: Set[str] = set()
    offset = 0
    expr = 'id != ""'
    while offset < total:
        limit = min(batch_size, total - offset)
        logger.debug(
            "Querying Milvus batch", extra={"offset": offset, "limit": limit, "total": total}
        )
        try:
            rows = coll.query(
                expr=expr,
                output_fields=["metadata"],
                offset=offset,
                limit=limit,
                consistency_level="Strong",
            )
        except TypeError:
            if offset != 0:
                logger.warning(
                    "Milvus client does not support offset; returning partial results",
                    extra={"retrieved": len(filenames), "total": total},
                )
                break
            rows = coll.query(
                expr=expr,
                output_fields=["metadata"],
                limit=limit,
                consistency_level="Strong",
            )
        except Exception:
            logger.exception("Failed to query Milvus for filenames")
            break
        if not rows:
            break
        for row in rows:
            name = meta_filename(row)
            if name and name != "N/A":
                filenames.add(name)
        offset += limit
        if len(rows) < limit:
            break

    sorted_names = sorted(filenames)
    logger.info(
        "Collected filenames from Milvus", extra={"collection": collection, "count": len(sorted_names)}
    )
    return sorted_names


def delete_documents_by_filename(
    filenames: Sequence[str],
    *,
    collection: str = collection_name,
    host: str = milvus_host,
    port: str = milvus_port,
) -> int:
    """Delete all Milvus records whose metadata source/filename matches the provided list."""

    targets = _normalise_filenames(filenames)
    if not targets:
        logger.warning("No filenames provided for deletion")
        return 0

    coll = _load_collection(collection, host, port)
    quoted = ", ".join(json.dumps(name) for name in targets)
    clauses = [f'metadata["source"] in [{quoted}]', f'metadata["filename"] in [{quoted}]']
    expr = " or ".join(f"({clause})" for clause in clauses)

    logger.info(
        "Deleting Milvus records by filename",
        extra={"collection": collection, "filenames": targets, "expr": expr},
    )
    try:
        result = coll.delete(expr=expr)
    except Exception:
        logger.exception("Failed to delete records from Milvus", extra={"expr": expr})
        return 0

    deleted = getattr(result, "delete_count", 0)
    try:
        coll.flush()
    except Exception:
        logger.warning("Milvus flush after delete failed", exc_info=True)
    logger.info(
        "Deleted Milvus records",
        extra={"collection": collection, "filenames": targets, "deleted_count": deleted},
    )
    return int(deleted)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and delete Milvus records based on stored PDF filenames."
    )
    parser.add_argument(
        "--collection",
        default=collection_name,
        help="Milvus collection name (default: configured collection)",
    )
    parser.add_argument("--host", default=milvus_host, help="Milvus host (default: configured host)")
    parser.add_argument("--port", default=milvus_port, help="Milvus port (default: configured port)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List filenames currently stored in Milvus")
    list_parser.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        help="Number of rows to fetch per Milvus query batch",
    )

    delete_parser = subparsers.add_parser(
        "delete", help="Delete all Milvus records that match the provided filenames"
    )
    delete_parser.add_argument(
        "filenames",
        nargs="+",
        help="One or more PDF filenames to delete (only the basename is used)",
    )

    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    if args.command == "list":
        names = list_document_filenames(
            collection=args.collection, host=args.host, port=args.port, batch_size=args.batch_size
        )
        if not names:
            print("No documents found in Milvus collection.")
        else:
            print(f"Found {len(names)} filename(s) in Milvus collection '{args.collection}':")
            for name in names:
                print(f" - {name}")
    elif args.command == "delete":
        deleted = delete_documents_by_filename(
            args.filenames, collection=args.collection, host=args.host, port=args.port
        )
        print(f"Deleted {deleted} record(s) from Milvus collection '{args.collection}'.")


if __name__ == "__main__":
    main()
