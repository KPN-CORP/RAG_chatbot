"""File system watcher that mirrors documents into Milvus.

The watcher has a single responsibility: detect file changes under the
configured ``WATCH_DIR`` and call hook functions that manage the Milvus
knowledge base. The hook implementations live in ``modules/milvus_hooks.py`` so
they can be swapped out or extended without touching the watcher logic.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, Iterable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import VALID_EXTS, WATCH_DIR
from logging_config import get_logger
from modules.milvus_hooks import delete_file_from_milvus, upsert_file_to_milvus

logger = get_logger(__name__)

DEBOUNCE_SECONDS = 2.0

pending_updates: Dict[str, float] = {}  # file_path -> last_event_time
pending_lock = threading.Lock()
stop_event = threading.Event()


def should_index(path: str) -> bool:
    """Return ``True`` if the file extension is part of the supported list."""

    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTS


def mark_dirty(path: str) -> None:
    """Remember a changed file so the debounce worker can process it later."""

    if not should_index(path):
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
        logger.info("File deleted", extra={"path": event.src_path})
        delete_file_from_milvus(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if should_index(event.src_path):
            logger.info("File moved (old)", extra={"path": event.src_path})
            delete_file_from_milvus(event.src_path)
        if should_index(event.dest_path):
            logger.info("File moved (new)", extra={"path": event.dest_path})
            mark_dirty(event.dest_path)


def _process_pending_updates(to_process: Iterable[str]) -> None:
    """Ingest or upsert a batch of dirty files."""

    for path in to_process:
        if os.path.exists(path):
            logger.info("Processing file after debounce", extra={"path": path})
            upsert_file_to_milvus(path)
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


def run_watcher():
    """Start the folder watcher and debounce worker."""

    os.makedirs(WATCH_DIR, exist_ok=True)

    event_handler = DocsEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=True)
    observer.start()

    worker_thread = threading.Thread(target=debounce_worker, daemon=True)
    worker_thread.start()

    logger.info("Watching for changes", extra={"path": WATCH_DIR})
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
    run_watcher()
