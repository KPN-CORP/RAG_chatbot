"""File system watcher that mirrors documents into Milvus.

The watcher has a single responsibility: detect file changes under the
configured ``WATCH_DIR`` and call hook functions that manage the Milvus
knowledge base. The hook implementations live in ``modules/milvus_hooks.py`` so
they can be swapped out or extended without touching the watcher logic.

The CLI now supports ``--profile`` to load a chatbot-specific config module
(e.g. ``config_IRIS.py``) using ``config_loader.load_config``. This keeps the
watcher in sync with the same collection and paths the chatbot uses.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from types import ModuleType
from typing import Dict, Iterable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config_loader import load_config
from logging_config import get_logger
from modules.milvus_hooks import delete_file_from_milvus, upsert_file_to_milvus
from modules.utils import sanitize_collection_component

logger = get_logger(__name__)

DEBOUNCE_SECONDS = 2.0

pending_updates: Dict[str, float] = {}  # file_path -> last_event_time
pending_lock = threading.Lock()
stop_event = threading.Event()

# Loaded at runtime so ``--profile`` can map to config_{PROFILE}.py.
CURRENT_CONFIG: ModuleType | None = None
WATCH_DIR: str = ""
VALID_EXTS: set[str] = set()


def configure_watcher(profile: str = "") -> ModuleType:
    """Load config for ``profile`` and populate module-level settings.

    If ``profile`` is empty, the default ``config.py`` is used. Otherwise the
    watcher attempts to import ``config_{profile.upper()}.py`` via
    ``config_loader.load_config``. The function also primes ``WATCH_DIR`` and
    ``VALID_EXTS`` so the event handler and debounce worker see consistent
    values.
    """

    global CURRENT_CONFIG, WATCH_DIR, VALID_EXTS

    config = load_config(profile)
    # logger.info("Config loaded for watcher", extra={"profile": profile or "default"})
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

    parser = argparse.ArgumentParser(description="Watch a folder and sync Milvus")
    parser.add_argument(
        "--profile",
        default="",
        help=(
            "Chatbot profile name; maps to config_{PROFILE.upper()}.py ("
            "default config.py when omitted)."
        ),
    )
    return parser.parse_args()


def should_index(path: str) -> bool:
    """Return ``True`` if the file extension is part of the supported list."""

    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTS


def _resolve_collection_for_path(path: str) -> str:
    """Return the Milvus collection name for the file, based on subfolders."""

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


def mark_dirty(path: str) -> None:
    """Remember a changed file so the debounce worker can process it later."""

    if not should_index(path):
        logger.info("Ignoring file with unsupported extension", extra={"path": path})
        return
    now = time.time()
    with pending_lock:
        last = pending_updates.get(path)
        # Only update and log if it's new or "old enough"
        if last is None or now - last > 0.5:
            pending_updates[path] = now
            logger.debug("Queued file for ingestion", extra={"path": path, "timestamp": now})


class DocsEventHandler(FileSystemEventHandler):
    """Handle file system events by delegating to the ingestion hooks."""

    def on_created(self, event):
        if event.is_directory:
            return
        logger.info("File created", extra={"path": event.src_path})
        mark_dirty(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        logger.info("File modified", extra={"path": event.src_path})
        mark_dirty(event.src_path)

    def on_deleted(self, event):
        if event.is_directory or not should_index(event.src_path):
            return
        collection_name = _resolve_collection_for_path(event.src_path)
        logger.info("File deleted", extra={"path": event.src_path})
        delete_file_from_milvus(
            event.src_path,
            config=CURRENT_CONFIG,
            collection_name=collection_name,
        )

    def on_moved(self, event):
        if event.is_directory:
            return
        if should_index(event.src_path):
            collection_name = _resolve_collection_for_path(event.src_path)
            logger.info("File moved (old)", extra={"path": event.src_path})
            delete_file_from_milvus(
                event.src_path,
                config=CURRENT_CONFIG,
                collection_name=collection_name,
            )
        if should_index(event.dest_path):
            logger.info("File moved (new)", extra={"path": event.dest_path})
            mark_dirty(event.dest_path)


def _process_pending_updates(to_process: Iterable[str]) -> None:
    """Ingest or upsert a batch of dirty files using the active config."""

    for path in to_process:
        if os.path.exists(path):
            collection_name = _resolve_collection_for_path(path)
            logger.info("Processing file after debounce", extra={"path": path})
            upsert_file_to_milvus(
                path,
                config=CURRENT_CONFIG,
                collection_name=collection_name,
            )
        else:
            # File vanished between event and worker
            logger.info("Skipping missing file", extra={"path": path})


def debounce_worker():
    """Background thread that ingests files once they've been stable for N seconds."""

    logger.info("Debounce worker started", extra={"debounce_seconds": DEBOUNCE_SECONDS})
    while not stop_event.is_set():
        now = time.time()
        to_process = []

        # Pick files whose last event is older than DEBOUNCE_SECONDS
        with pending_lock:
            for path, last_event_time in list(pending_updates.items()):
                if now - last_event_time >= DEBOUNCE_SECONDS:
                    to_process.append(path)
                    del pending_updates[path]

        _process_pending_updates(to_process)
        time.sleep(1.0)
    logger.info("Debounce worker stopping")


def run_watcher(profile: str = ""):
    """Start the folder watcher and debounce worker for the given profile."""

    config = configure_watcher(profile)
    os.makedirs(WATCH_DIR, exist_ok=True)

    event_handler = DocsEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=True)
    observer.start()

    worker_thread = threading.Thread(target=debounce_worker, daemon=True)
    worker_thread.start()

    logger.info(
        "Watching for changes",
        extra={
            "path": WATCH_DIR,
            "profile": profile or "default",
            "collection": config.collection_name,
            "recursive": True,
        },
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watcher...")
        stop_event.set()
        observer.stop()

    observer.join()
    worker_thread.join()
    logger.info("Watcher stopped.")


if __name__ == "__main__":
    args = parse_args()
    run_watcher(profile=args.profile)
