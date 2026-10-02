"""The release's declared SQLite compatibility must match startup behavior."""

import sqlite3

import pytest

from chord.responses_store import ResponseStore
from chord.store_schema import STORES
from chord.videos import VideoStore


def version(path):
    with sqlite3.connect(path) as db:
        return db.execute("PRAGMA user_version").fetchone()[0]


def test_unversioned_responses_are_migrated_and_remain_readable(tmp_path):
    path = tmp_path / STORES["responses"].path
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE responses (id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, "
            "response TEXT NOT NULL, input_items TEXT NOT NULL, conversation TEXT NOT NULL)"
        )
        db.execute("INSERT INTO responses VALUES ('r1', 1, '{}', '[]', '[]')")
    assert version(path) == 0

    store = ResponseStore(path)
    assert store._db.execute("SELECT id FROM responses").fetchone()[0] == "r1"
    assert "events" in {r[1] for r in store._db.execute("PRAGMA table_info(responses)")}
    assert version(path) == STORES["responses"].write_version == 1
    store._db.close()


def test_unversioned_videos_are_migrated_and_remain_readable(tmp_path):
    path = tmp_path / STORES["videos"].path
    path.parent.mkdir()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE videos (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        db.execute(
            "INSERT INTO videos VALUES ('video_01hzz', "
            '\'{"id":"video_01hzz","status":"completed","created_at":1}\')'
        )
    assert version(path) == 0

    store = VideoStore(path.parent)
    assert store.get("video_01hzz")["status"] == "completed"
    assert version(path) == STORES["videos"].write_version == 1
    store.close()


@pytest.mark.parametrize(
    ("name", "open_store"),
    [
        ("responses", lambda path: ResponseStore(path)),
        ("videos", lambda path: VideoStore(path.parent)),
    ],
)
def test_future_schema_refuses_before_creating_tables(tmp_path, name, open_store):
    path = tmp_path / STORES[name].path
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=2")
    with pytest.raises(RuntimeError, match="schema version 2 is not readable"):
        open_store(path)
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            == []
        )
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
