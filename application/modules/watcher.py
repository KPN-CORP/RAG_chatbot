"""Cron-style folder watcher that mirrors documents into Milvus.

This module no longer relies on ``watchdog`` file-system events. Instead, it
runs a periodic scan (cron-job style) of ``WATCH_DIR`` and compares snapshots to
detect created, modified, and deleted files. Detected changes are delegated to
hooks in ``modules/milvus_hooks.py``.

The CLI keeps ``--profile`` support so the watcher can load
``config_{PROFILE}.py`` through ``config_loader.load_config``.
"""

from __future__ import annotations

import argparse
import os
import time
from types import ModuleType
from typing import Dict, Tuple

from config_loader import load_config
from logging_config import get_logger
from modules.milvus_hooks import delete_file_from_milvus, upsert_file_to_milvus
from modules.utils import sanitize_collection_component

logger = get_logger(__name__)

DEFAULT_SCAN_INTERVAL_SECONDS = 10.0

# Loaded at runtime so ``--profile`` can map to config_{PROFILE}.py.
CURRENT_CONFIG: ModuleType | None = None
WATCH_DIR: str = ""
VALID_EXTS: set[str] = set()

# file_path -> last modified timestamp
Snapshot = Dict[str, float]


def configure_watcher(profile: str = "") -> ModuleType:
    """Load config for ``profile`` and populate module-level settings."""

    global CURRENT_CONFIG, WATCH_DIR, VALID_EXTS

    config = load_config(profile)
    CURRENT_CONFIG = config
    WATCH_DIR = getattr(config, "WATCH_DIR", "../assets/")
    VALID_EXTS = set(getattr(config, "VALID_EXTS", {".pdf", ".docx", ".txt"}))

    logger.info(
        "Watcher configuration loaded",
        extra={
            "profile": profile or "default",
            "watch_dir": WATCH_DIR,
            "watch_dir_abs": os.path.abspath(WATCH_DIR),
            "valid_exts": sorted(VALID_EXTS),
            "config_module": getattr(config, "__name__", "unknown"),
            "config_file": getattr(config, "__file__", "unknown"),
        },
    )
    return config


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for the watcher."""

    parser = argparse.ArgumentParser(description="Scan a folder and sync Milvus")
    parser.add_argument(
        "--profile",
        default="",
        help=(
            "Chatbot profile name; maps to config_{PROFILE.upper()}.py ("
            "default config.py when omitted)."
        ),
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_SCAN_INTERVAL_SECONDS,
        help="How often to scan WATCH_DIR for changes.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Scan once, process differences, then exit (useful for cron).",
    )
    return parser.parse_args()


def should_index(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTS


def _resolve_collection_for_path(path: str) -> str:
    if not CURRENT_CONFIG:
        raise RuntimeError("Watcher config not loaded before resolving collection name.")
    base_collection = CURRENT_CONFIG.collection_name
    relative_path = os.path.relpath(path, WATCH_DIR)
    if relative_path.startswith(".."):
        return base_collection
    parts = relative_path.split(os.sep)
    subfolder = parts[0] if len(parts) > 1 else ""
    if not subfolder:
        return base_collection
    suffix = sanitize_collection_component(subfolder)
    return f"{base_collection}_{suffix}" if suffix else base_collection


def _scan_snapshot() -> Snapshot:
    """Return current file snapshot for indexed files in WATCH_DIR."""

    snapshot: Snapshot = {}
    for root, _dirs, files in os.walk(WATCH_DIR):
        for name in files:
            full_path = os.path.join(root, name)
            if not should_index(full_path):
                continue
            try:
                snapshot[full_path] = os.path.getmtime(full_path)
            except FileNotFoundError:
                # File may disappear during scan; it will be picked up next cycle.
                continue
    return snapshot


def _diff_snapshots(previous: Snapshot, current: Snapshot) -> Tuple[list[str], list[str], list[str]]:
    """Return (created, modified, deleted) paths between two snapshots."""

    previous_paths = set(previous)
    current_paths = set(current)

    created = sorted(current_paths - previous_paths)
    deleted = sorted(previous_paths - current_paths)
    modified = sorted(
        path
        for path in previous_paths & current_paths
        if current[path] > previous[path]
    )
    return created, modified, deleted


def _upsert(path: str) -> None:
    collection_name = _resolve_collection_for_path(path)
    logger.info("Upserting changed file", extra={"path": path, "collection": collection_name})
    upsert_file_to_milvus(path, config=CURRENT_CONFIG, collection_name=collection_name)


def _delete(path: str) -> None:
    collection_name = _resolve_collection_for_path(path)
    logger.info("Deleting removed file", extra={"path": path, "collection": collection_name})
    delete_file_from_milvus(path, config=CURRENT_CONFIG, collection_name=collection_name)


def run_watcher(
    profile: str = "",
    interval_seconds: float = DEFAULT_SCAN_INTERVAL_SECONDS,
    run_once: bool = False,
):
    """Start a cron-style watcher loop for the given profile."""

    config = configure_watcher(profile)
    os.makedirs(WATCH_DIR, exist_ok=True)

    previous_snapshot = _scan_snapshot()
    logger.info(
        "Cron watcher started",
        extra={
            "path": WATCH_DIR,
            "profile": profile or "default",
            "collection": config.collection_name,
            "interval_seconds": interval_seconds,
            "tracked_files": len(previous_snapshot),
        },
    )

    try:
        while True:
            time.sleep(interval_seconds)
            current_snapshot = _scan_snapshot()
            created, modified, deleted = _diff_snapshots(previous_snapshot, current_snapshot)

            for path in created:
                _upsert(path)
            for path in modified:
                _upsert(path)
            for path in deleted:
                _delete(path)

            previous_snapshot = current_snapshot
            if run_once:
                logger.info("Cron watcher run_once completed.")
                break
    except KeyboardInterrupt:
        logger.info("Cron watcher stopped by user.")


if __name__ == "__main__":
    args = parse_args()
    run_watcher(
        profile=args.profile,
        interval_seconds=args.interval_seconds,
        run_once=args.run_once,
    )
