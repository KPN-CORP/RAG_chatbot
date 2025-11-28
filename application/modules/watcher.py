import threading
import os
import time
import threading
from typing import Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import WATCH_DIR, VALID_EXTS

DEBOUNCE_SECONDS = 2.0

pending_updates: Dict[str, float] = {}  # file_path -> last_event_time
pending_lock = threading.Lock()
stop_event = threading.Event()

def should_index(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTS

def mark_dirty(path: str) -> None:
    if not should_index(path):
        return
    now = time.time()
    with pending_lock:
        last = pending_updates.get(path)
        # Only update and log if it's new or "old enough"
        if last is None or now - last > 0.5:
            pending_updates[path] = now
            print(f"[DIRTY] {path} at {now}")

class DocsEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        print(f"[CREATED] {event.src_path}")
        mark_dirty(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"[MODIFIED] {event.src_path}")
        mark_dirty(event.src_path)

    def on_deleted(self, event):
        if event.is_directory or not should_index(event.src_path):
            return
        print(f"[DELETED] {event.src_path}")
#         delete_file_from_milvus(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if should_index(event.src_path):
            print(f"[MOVED-OLD] {event.src_path}")
#             delete_file_from_milvus(event.src_path)
        if should_index(event.dest_path):
            print(f"[MOVED-NEW] {event.dest_path}")
            mark_dirty(event.dest_path)

def debounce_worker():
    """Background thread that ingests files once they've been stable for N seconds."""
    print("[DEBOUNCE] worker started")
    i=0
    while not stop_event.is_set():
        now = time.time()
        to_process = []

        # Pick files whose last event is older than DEBOUNCE_SECONDS
        with pending_lock:
            for path, last_event_time in list(pending_updates.items()):
                if now - last_event_time >= DEBOUNCE_SECONDS:
                    to_process.append(path)
                    del pending_updates[path]

        # Process outside the lock
        for path in to_process:
            if os.path.exists(path):
                print(f"[INGEST] {path}")
#                 upsert_file_to_milvus(path)
            else:
                # File vanished between event and worker
                print(f"[SKIP] {path} no longer exists")
        time.sleep(1.0)
    print("[DEBOUNCE] worker stopping")


def run_watcher():
    os.makedirs(WATCH_DIR, exist_ok=True)

    event_handler = DocsEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=True)
    observer.start()

    worker_thread = threading.Thread(target=debounce_worker, daemon=True)
    worker_thread.start()

    print(f"Watching {WATCH_DIR} for changes...")
    # ===========================================
    # i = 0
    # ===========================================
    try:
        while True:
            time.sleep(1)
            # ===========================================
            # i+=1
            # if i>60:
            #     print("Already one minute...")
            #     stop_event.set()
            #     observer.stop()
            #     break
            # ===========================================
    except KeyboardInterrupt:
        print("Stopping watcher...")
        stop_event.set()
        observer.stop()

    observer.join()
    worker_thread.join()
    print("Watcher stopped.")


if __name__ == "__main__":
    run_watcher()