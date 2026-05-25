import sys
from pathlib import Path
from types import SimpleNamespace

APPLICATION_DIR = Path(__file__).resolve().parents[1]
if str(APPLICATION_DIR) not in sys.path:
    sys.path.append(str(APPLICATION_DIR))

from modules import watcher
from modules.watcher import _diff_snapshots


def test_diff_snapshots_detects_create_modify_delete():
    previous = {
        "/tmp/a.pdf": 10.0,
        "/tmp/b.pdf": 10.0,
    }
    current = {
        "/tmp/b.pdf": 12.0,
        "/tmp/c.pdf": 1.0,
    }

    created, modified, deleted = _diff_snapshots(previous, current)

    assert created == ["/tmp/c.pdf"]
    assert modified == ["/tmp/b.pdf"]
    assert deleted == ["/tmp/a.pdf"]


def test_run_watcher_run_once_processes_single_cycle(monkeypatch):
    monkeypatch.setattr(
        watcher,
        "configure_watcher",
        lambda profile: SimpleNamespace(collection_name="demo"),
    )
    monkeypatch.setattr(watcher.os, "makedirs", lambda *args, **kwargs: None)

    snapshots = iter([
        {"/tmp/a.pdf": 1.0},
        {"/tmp/a.pdf": 2.0, "/tmp/b.pdf": 1.0},
    ])
    monkeypatch.setattr(watcher, "_scan_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(watcher.time, "sleep", lambda _seconds: None)

    upserts = []
    deletes = []
    monkeypatch.setattr(watcher, "_upsert", lambda path: upserts.append(path))
    monkeypatch.setattr(watcher, "_delete", lambda path: deletes.append(path))

    watcher.run_watcher(profile="IRIS", interval_seconds=0, run_once=True)

    assert upserts == ["/tmp/b.pdf", "/tmp/a.pdf"]
    assert deletes == []
