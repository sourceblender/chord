"""Stored responses expire even when nobody reads the expired id."""
import time

from chord.responses_store import RETENTION_S, ResponseStore


def test_a_later_write_drops_expired_rows_that_were_never_read(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.put({"id": "resp_old", "output": ["image-bytes"]}, [], [])
    store._db.execute(
        "UPDATE responses SET created_at = ? WHERE id = ?",
        (time.time() - RETENTION_S - 10, "resp_old"),
    )
    store.put({"id": "resp_new"}, [], [])
    assert store.get("resp_old") is None
    assert store.get("resp_new")["id"] == "resp_new"


def test_a_live_volume_gains_the_retention_indexes(tmp_path):
    """The retention sweeps run inside EVERY read and write, and MAX(seq) inside
    every insert, so a volume created before the indexes existed must gain them
    on open (review 2026-09-22: each was a full scan on the event loop)."""
    import sqlite3

    path = tmp_path / "responses.sqlite3"
    old = sqlite3.connect(path)
    old.execute("""CREATE TABLE responses (id TEXT PRIMARY KEY, created_at INTEGER NOT NULL,
                   response TEXT NOT NULL, input_items TEXT NOT NULL, conversation TEXT NOT NULL)""")
    old.execute("""CREATE TABLE chat_completions (id TEXT PRIMARY KEY, created_at INTEGER NOT NULL,
                   seq INTEGER NOT NULL, model TEXT NOT NULL, metadata TEXT NOT NULL,
                   completion TEXT NOT NULL, messages TEXT NOT NULL)""")
    old.commit()
    old.close()

    ResponseStore(path)

    db = sqlite3.connect(path)
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"responses_created_at", "chat_completions_created_at", "chat_completions_seq", "files_seq"} <= names


def test_get_chat_reads_one_row_and_both_readers_expire(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.put_chat({"id": "chatcmpl-a", "model": "m"}, {"team": "kitchen"},
                   [{"role": "user", "content": "x" * 10}])
    store.put_chat({"id": "chatcmpl-b", "model": "m"}, {}, [])

    listed, _ = store.list_chat()
    assert [c["id"] for c, _ in listed] == ["chatcmpl-a", "chatcmpl-b"]      # seq order
    assert listed[0][1] == {"team": "kitchen"}

    completion, metadata, messages = store.get_chat("chatcmpl-a")
    assert completion["id"] == "chatcmpl-a"
    assert metadata == {"team": "kitchen"}
    assert messages == [{"role": "user", "content": "x" * 10}]
    assert store.get_chat("chatcmpl-missing") is None

    store._db.execute("UPDATE chat_completions SET created_at = ? WHERE id = ?",
                      (time.time() - RETENTION_S - 10, "chatcmpl-a"))
    assert store.get_chat("chatcmpl-a") is None
    assert [c["id"] for c, _ in store.list_chat()[0]] == ["chatcmpl-b"]


def test_a_restart_fails_every_orphan_not_only_background_ones(tmp_path):
    """After a restart no worker exists for ANY non-terminal row. The
    background-only sweep left a plain streamed response orphaned by a
    disconnect in_progress forever -- a poller polled forever, and the
    chaining refusal made the orphan permanently unchainable (review,
    batch 2 required fix)."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.put({"id": "resp_stream_orphan", "status": "in_progress", "output": []}, [], [])
    store.put({"id": "resp_bg_orphan", "status": "queued", "background": True, "output": []}, [], [])
    store.put({"id": "resp_done", "status": "completed", "output": []}, [], [])

    store.abandon_interrupted()

    assert store.get("resp_stream_orphan")["status"] == "failed"
    assert store.get("resp_bg_orphan")["status"] == "failed"
    assert store.get("resp_done")["status"] == "completed"
