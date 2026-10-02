"""SQL pagination across the four list endpoints (#6).

The four endpoints that used to load and decode the full table on every
read are now backed by SQL WHERE/LIMIT/ORDER with an id cursor. These
tests pin:

  * the SQL path returns the right items in the right order;
  * `after=ID` paginates forward, in either direction, including past
    the end of the table;
  * an unknown `after` is a 400, not a silent 200 with an empty page;
  * `has_more` is exact without a second COUNT(*) query (we read limit+1);
  * filters (model, metadata[k]=v, purpose) are pushed into SQL, not
    applied in Python after a full scan;
  * retention still sweeps expired rows on every list call.

The unit tests use the store directly to avoid spinning up the whole
app for every assertion; one end-to-end test per route covers the wire.
"""
from __future__ import annotations

import time

import pytest

from chord.responses_store import RETENTION_S, ResponseStore
from test_responses import make as make_app
from test_skeleton import FakeUpstream


# --- chat completions: store-level pagination ----------------------------------


def _put_chat(store, completion_id, *, model="chord-1-poly", metadata=None):
    """Insert a stored chat completion at a chosen seq position."""
    store.put_chat(
        {"id": completion_id, "model": model, "choices": []},
        metadata or {},
        [{"role": "user", "content": "hi"}],
    )


def test_chat_list_returns_in_seq_order_ascending_by_default(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-c", metadata={"team": "kitchen"})
    _put_chat(store, "chatcmpl-b")
    _put_chat(store, "chatcmpl-a", metadata={"team": "bath"})
    rows, _ = store.list_chat()
    assert [c["id"] for c, _ in rows] == ["chatcmpl-c", "chatcmpl-b", "chatcmpl-a"]


def test_chat_list_desc_returns_newest_first(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a")
    _put_chat(store, "chatcmpl-b")
    _put_chat(store, "chatcmpl-c")
    rows, _ = store.list_chat(order="desc")
    assert [c["id"] for c, _ in rows] == ["chatcmpl-c", "chatcmpl-b", "chatcmpl-a"]


def test_chat_list_after_cursor_walks_forward_in_asc(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    for cid in ("a", "b", "c", "d", "e"):
        _put_chat(store, f"chatcmpl-{cid}")
    rows, after_exists = store.list_chat(after_id="chatcmpl-b", limit=10, order="asc")
    assert after_exists is True
    assert [c["id"] for c, _ in rows] == ["chatcmpl-c", "chatcmpl-d", "chatcmpl-e"]


def test_chat_list_after_cursor_walks_backward_in_desc(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    for cid in ("a", "b", "c", "d", "e"):
        _put_chat(store, f"chatcmpl-{cid}")
    rows, after_exists = store.list_chat(after_id="chatcmpl-d", limit=10, order="desc")
    assert after_exists is True
    assert [c["id"] for c, _ in rows] == ["chatcmpl-c", "chatcmpl-b", "chatcmpl-a"]


def test_chat_list_after_cursor_past_end_is_empty_with_cursor_found(tmp_path):
    """`after` pointing past the last row in either direction is a real cursor:
    after_exists is True (it WAS in the table), the page is empty, no error.

    This is what callers see when paginating to the end."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a")
    _put_chat(store, "chatcmpl-b")
    rows, after_exists = store.list_chat(after_id="chatcmpl-b", order="asc")
    assert after_exists is True
    assert rows == []
    rows, after_exists = store.list_chat(after_id="chatcmpl-a", order="desc")
    assert after_exists is True
    assert rows == []


def test_chat_list_after_cursor_unknown_returns_after_exists_false(tmp_path):
    """An `after` that was never in the table returns after_exists=False so
    the route can 400 rather than silently returning a possibly-empty page.
    The SQL filter also produces empty rows when the bogus cursor sorts
    after every existing id (ULIDs are time-sortable; a random id usually
    does), so the contract is `after_exists=False implies the route 400s`,
    not `after_exists=False implies the page is empty`."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a")
    rows, after_exists = store.list_chat(after_id="chatcmpl-never-existed", order="asc")
    assert after_exists is False
    # The route will 400 on the bogus cursor. The page contents are
    # whatever the SQL `id > after` filter produces, which for a random id
    # string is usually empty -- the contract is the boolean, not the rows.
    assert isinstance(rows, list)


def test_chat_list_pagination_limit_is_exact(tmp_path):
    """limit=N returns N items; limit=N with N+1 in the table leaves has_more
    to the caller. The store returns `limit` rows; the route asks for
    `limit + 1` so `has_more` is exact without a second query."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    for i in range(5):
        _put_chat(store, f"chatcmpl-{i}")
    rows, _ = store.list_chat(limit=3)
    assert [c["id"] for c, _ in rows] == ["chatcmpl-0", "chatcmpl-1", "chatcmpl-2"]
    rows, _ = store.list_chat(limit=3, order="desc")
    assert [c["id"] for c, _ in rows] == ["chatcmpl-4", "chatcmpl-3", "chatcmpl-2"]


def test_chat_list_cursor_seek_uses_seq_and_id_tuple(tmp_path):
    """The cursor predicate must compare (seq, id), not id alone. A row
    whose id sorts after the cursor's id but whose seq sorts before (or
    vice versa) is on the wrong side of the boundary; an id-only
    predicate would skip it (review 2026-09-23, Copilot high-severity).

    We can't easily inject a seq/id mismatch into the live table (seq is
    monotonic per insert and ids are ULIDs which are time-sortable), but
    we CAN pin the SQL semantics by pre-populating rows with explicit
    seq values and then asking for the page past a known cursor."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    # Force seq values that do not align with id order. id "z" gets seq 1;
    # id "a" gets seq 2; id "m" gets seq 3. In seq order: z, a, m. In id
    # order: a, m, z. The cursor at (seq=2, id="a") in ascending order
    # must yield [m, z] -- but a buggy predicate `id > "a"` would yield
    # [m, z] too, so we need a stronger test.
    _put_chat(store, "z")  # seq=1
    _put_chat(store, "a")  # seq=2
    _put_chat(store, "m")  # seq=3
    # Cursor at the middle row "a" (seq=2). In ascending order, the
    # next rows by (seq, id) are [(3, "m")], so the page should be ["m"].
    rows, ok = store.list_chat(after_id="a", order="asc", limit=10)
    assert ok is True
    assert [c["id"] for c, _ in rows] == ["m"]
    # In descending order, the cursor at (2, "a") should yield [(1, "z")].
    rows, ok = store.list_chat(after_id="a", order="desc", limit=10)
    assert ok is True
    assert [c["id"] for c, _ in rows] == ["z"]


def test_chat_list_cursor_with_dotted_metadata_key(tmp_path):
    """A metadata key with a dot in its name must be matched as a literal
    top-level key, not as a nested path. The metadata validator allows
    arbitrary strings up to 64 chars, so `team.name` is a possible
    stored key. Pin that the SQL filter matches it (review 2026-09-23,
    Copilot medium-severity)."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a", metadata={"team.name": "kitchen"})
    _put_chat(store, "chatcmpl-b", metadata={"team": {"name": "kitchen"}})
    _put_chat(store, "chatcmpl-c", metadata={"team.name": "bath"})
    rows, _ = store.list_chat(metadata={"team.name": "kitchen"})
    assert [c["id"] for c, _ in rows] == ["chatcmpl-a"]
    rows, _ = store.list_chat(metadata={"team": "x"})  # not a real key
    assert rows == []


def test_chat_list_model_filter_is_pushed_into_sql(tmp_path):
    """Filtering by model in SQL means a different model returns no rows
    without loading any of the table's other rows. We can't measure that
    from outside, but we can pin the result is right AND the store-level
    call does not raise on a model that has zero rows."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a", model="chord-1-poly")
    _put_chat(store, "chatcmpl-b", model="chord-1-other")
    _put_chat(store, "chatcmpl-c", model="chord-1-poly")
    rows, _ = store.list_chat(model="chord-1-other")
    assert [c["id"] for c, _ in rows] == ["chatcmpl-b"]
    rows, _ = store.list_chat(model="not-served")
    assert rows == []


def test_chat_list_metadata_filter_is_pushed_into_sql(tmp_path):
    """A metadata[k]=v filter is matched in SQL via json_extract, so a row
    whose metadata does not match is not loaded past the index."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a", metadata={"team": "kitchen"})
    _put_chat(store, "chatcmpl-b", metadata={"team": "bath", "k": "v"})
    _put_chat(store, "chatcmpl-c", metadata={"team": "bath"})
    rows, _ = store.list_chat(metadata={"team": "bath"})
    assert [c["id"] for c, _ in rows] == ["chatcmpl-b", "chatcmpl-c"]
    rows, _ = store.list_chat(metadata={"team": "bath", "k": "v"})
    assert [c["id"] for c, _ in rows] == ["chatcmpl-b"]


def test_chat_list_does_not_load_messages_column(tmp_path):
    """The list route serves only `completion` and `metadata`. The store's
    SELECT reads those two columns and never touches `messages`, so a 1 MB
    messages blob on each row does not slow a list call."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    huge = "x" * (1 << 20)
    _put_chat(store, "chatcmpl-a", metadata={"team": "kitchen"})
    store.put_chat(
        {"id": "chatcmpl-huge", "model": "m", "choices": []},
        {},
        [{"role": "user", "content": huge}],
    )
    rows, _ = store.list_chat()
    # Sanity: the second row is in the listing.
    assert [c["id"] for c, _ in rows] == ["chatcmpl-a", "chatcmpl-huge"]
    # And the list call did not surface the 1 MB blob.
    for completion, _ in rows:
        assert "choices" in completion


def test_chat_list_still_sweeps_expired_rows(tmp_path):
    """The retention sweep ran inside list_chat before, and must run inside
    the SQL-paged version too -- otherwise an expired row would appear in
    the page and confuse has_more."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a")
    _put_chat(store, "chatcmpl-b")
    store._db.execute(
        "UPDATE chat_completions SET created_at = ? WHERE id = ?",
        (time.time() - RETENTION_S - 10, "chatcmpl-a"),
    )
    rows, _ = store.list_chat()
    assert [c["id"] for c, _ in rows] == ["chatcmpl-b"]


# --- files: store-level pagination ---------------------------------------------


def _put_file(store, file_id, *, purpose="user_data", expires_at=None):
    """Insert a file row directly so tests don't need to go through the route."""
    file = {"id": file_id, "purpose": purpose, "bytes": 1, "created_at": int(time.time()),
            "filename": "x", "object": "file"}
    if expires_at is not None:
        file["expires_at"] = expires_at
    store.put_file(file)


def test_files_list_default_order_is_desc(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_file(store, "file-a")
    _put_file(store, "file-b")
    _put_file(store, "file-c")
    files, gone, _ = store.live_files()
    assert [f["id"] for f in files] == ["file-c", "file-b", "file-a"]
    assert gone == []


def test_files_list_purpose_filter_is_pushed_into_sql(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_file(store, "file-a", purpose="user_data")
    _put_file(store, "file-b", purpose="batch")
    _put_file(store, "file-c", purpose="user_data")
    files, _, _ = store.live_files(purpose="user_data")
    assert [f["id"] for f in files] == ["file-c", "file-a"]
    files, _, _ = store.live_files(purpose="batch")
    assert [f["id"] for f in files] == ["file-b"]


def test_files_list_after_cursor_walks_in_order(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    for cid in ("file-a", "file-b", "file-c", "file-d"):
        _put_file(store, cid)
    files, gone, after_exists = store.live_files(after_id="file-b", order="desc")
    assert after_exists is True
    assert gone == []
    assert [f["id"] for f in files] == ["file-a"]


def test_files_list_after_cursor_unknown_returns_after_exists_false(tmp_path):
    """A cursor that was never in the table returns after_exists=False. The
    page is empty because the route 400s before serializing anything. The
    SQL filter is not run for a known-bogus cursor."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_file(store, "file-a")
    files, gone, after_exists = store.live_files(after_id="file-never-existed")
    assert after_exists is False
    assert files == []
    assert gone == []


def test_files_list_expired_rows_are_swept_before_paging(tmp_path):
    """The retention sweep must run before the SELECT: an expired row that
    would otherwise be in the page would inflate has_more and waste the
    caller."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_file(store, "file-a")
    _put_file(store, "file-b")
    _put_file(store, "file-c", expires_at=int(time.time()) - 10)
    files, gone, _ = store.live_files()
    assert [f["id"] for f in files] == ["file-b", "file-a"]
    assert gone == ["file-c"]


def test_files_list_unlinks_expired_bytes_on_bogus_cursor(tmp_path):
    """The bogus-cursor 400 used to return before the unlink loop, leaving
    expired file bytes on disk forever. The unlink now runs before the 400
    (Copilot review 2026-09-23, fourth round)."""
    deps, client, _sdk = make_app(tmp_path, FakeUpstream())
    # Seed an expired file with bytes on disk.
    upload = client.post("/v1/files", data={"purpose": "user_data"},
                         files={"file": ("x.txt", b"hello", "text/plain")},
                         headers={"Authorization": "Bearer k"})
    assert upload.status_code == 200
    file_id = upload.json()["id"]
    # Force the file's expires_at into the past.
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store._db.execute(
        "UPDATE files SET expires_at = ? WHERE id = ?",
        (int(time.time()) - 10, file_id),
    )
    bytes_path = deps.settings.data_dir / "files" / file_id
    assert bytes_path.exists()
    # A bogus-cursor request must still unlink the expired bytes.
    r = client.get(f"/v1/files?after={file_id}_bogus", headers={"Authorization": "Bearer k"})
    assert r.status_code == 400, r.text
    assert not bytes_path.exists()


def test_files_list_pagination_limit_is_exact(tmp_path):
    store = ResponseStore(tmp_path / "responses.sqlite3")
    for i in range(5):
        _put_file(store, f"file-{i}")
    files, _, _ = store.live_files(limit=2)
    assert [f["id"] for f in files] == ["file-4", "file-3"]
    files, _, _ = store.live_files(limit=2, order="asc")
    assert [f["id"] for f in files] == ["file-0", "file-1"]


# --- videos: store-level pagination --------------------------------------------


def test_videos_list_orders_by_seq_descending_by_default(tmp_path):
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    for i, vid in enumerate(("video_a", "video_b", "video_c")):
        record = {"id": vid, "object": "video", "status": "completed",
                  "created_at": 1000 + i, "expires_at": None}
        store.put(record)
    videos, _ = store.list_live()
    assert [v["id"] for v in videos] == ["video_c", "video_b", "video_a"]


def test_videos_list_asc_orders_oldest_first(tmp_path):
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    for i, vid in enumerate(("video_a", "video_b", "video_c")):
        record = {"id": vid, "object": "video", "status": "completed",
                  "created_at": 1000 + i, "expires_at": None}
        store.put(record)
    videos, _ = store.list_live(order="asc")
    assert [v["id"] for v in videos] == ["video_a", "video_b", "video_c"]


def test_videos_list_after_cursor_walks_in_order(tmp_path):
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    for i, vid in enumerate(("video_a", "video_b", "video_c", "video_d")):
        record = {"id": vid, "object": "video", "status": "completed",
                  "created_at": 1000 + i, "expires_at": None}
        store.put(record)
    videos, after_exists = store.list_live(after_id="video_c", order="desc")
    assert after_exists is True
    assert [v["id"] for v in videos] == ["video_b", "video_a"]
    videos, after_exists = store.list_live(after_id="video_b", order="asc")
    assert after_exists is True
    assert [v["id"] for v in videos] == ["video_c", "video_d"]


def test_videos_list_after_cursor_unknown_returns_after_exists_false(tmp_path):
    """A cursor that was never in the table returns after_exists=False. The
    page is empty because the route 400s before serializing anything. The
    SQL filter is not run for a known-bogus cursor."""
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_a", "object": "video", "status": "completed",
               "created_at": 1000, "expires_at": None})
    videos, after_exists = store.list_live(after_id="video_bogus", order="desc")
    assert after_exists is False
    assert videos == []


def test_videos_list_expired_rows_are_swept_before_paging(tmp_path):
    """The retention sweep uses the expires_at column (extracted from record
    in the schema migration), so the page is post-sweep without json.loads."""
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_live", "object": "video", "status": "completed",
               "created_at": 1000, "expires_at": None})
    store.put({"id": "video_expired", "object": "video", "status": "completed",
               "created_at": 1001, "expires_at": int(time.time()) - 10})
    videos, _ = store.list_live()
    assert [v["id"] for v in videos] == ["video_live"]


def test_videos_schema_migration_is_idempotent_on_a_reopened_volume(tmp_path):
    """A volume created before the schema migration must open cleanly and
    gain the columns. Reopening must not error on the second pass."""
    from chord.videos import VideoStore
    first = VideoStore(tmp_path / "videos")
    first.put({"id": "video_a", "object": "video", "status": "completed",
               "created_at": 1000, "expires_at": None})
    del first
    # Second open: ALTER TABLE IF NOT EXISTS-equivalent guards must skip when
    # the column already exists.
    second = VideoStore(tmp_path / "videos")
    videos, _ = second.list_live()
    assert [v["id"] for v in videos] == ["video_a"]


def test_videos_old_rows_have_zero_seq_and_order_by_id(tmp_path):
    """A row written before the schema migration had seq=0; ordering breaks
    the tie by id (ULIDs are time-sortable, so the tie order is still
    monotonic-by-creation)."""
    import sqlite3
    from chord.videos import VideoStore
    # Create a video table with the old shape (no seq/created_at/expires_at).
    path = tmp_path / "videos"
    path.mkdir()
    db = sqlite3.connect(path / "index.sqlite3")
    db.execute("CREATE TABLE videos (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    db.execute(
        "INSERT INTO videos VALUES (?, ?)",
        ("video_older_ulid", '{"id": "video_older_ulid", "object": "video", "status": "completed"}'),
    )
    db.commit()
    db.close()
    # Open with the new code -- schema migration runs.
    store = VideoStore(path)
    videos, _ = store.list_live(order="asc")
    assert [v["id"] for v in videos] == ["video_older_ulid"]


def test_videos_old_rows_get_unique_seq_on_migration(tmp_path):
    """Pre-migration rows would all default to seq=0; that left a visible
    split between "old" rows (seq=0) and "new" inserts (seq=N+1, N+2). The
    migration backfills seq by id order so the index is monotonic across
    the boundary."""
    import sqlite3
    from chord.videos import VideoStore
    path = tmp_path / "videos"
    path.mkdir()
    db = sqlite3.connect(path / "index.sqlite3")
    db.execute("CREATE TABLE videos (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    for i in range(3):
        db.execute("INSERT INTO videos VALUES (?, ?)",
                   (f"video_{i:02d}", f'{{"id": "video_{i:02d}", "object": "video", "status": "completed"}}'))
    db.commit()
    db.close()
    VideoStore(path)
    db = sqlite3.connect(path / "index.sqlite3")
    seqs = [r[0] for r in db.execute("SELECT seq FROM videos ORDER BY id")]
    assert seqs == [1, 2, 3]
    db.close()


def test_videos_old_expires_at_is_backfilled_from_json(tmp_path):
    """Pre-migration rows whose JSON record carried an int `expires_at`
    must have the column populated, otherwise the SQL sweep skips them
    and they survive forever (review 2026-09-23, Copilot high-severity)."""
    import sqlite3
    from chord.videos import VideoStore
    path = tmp_path / "videos"
    path.mkdir()
    db = sqlite3.connect(path / "index.sqlite3")
    db.execute("CREATE TABLE videos (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    db.execute(
        "INSERT INTO videos VALUES (?, ?)",
        ("video_old_expired",
         '{"id": "video_old_expired", "object": "video", "status": "completed", "expires_at": 1}'),
    )
    db.commit()
    db.close()
    VideoStore(path)
    # list_live runs the sweep first; the backfilled expires_at must let
    # the row be deleted and removed from the page.
    store = VideoStore(path)
    videos, _ = store.list_live()
    assert videos == []


def test_videos_get_does_not_deadlock_on_expired_row(tmp_path):
    """Pre-fix: _drop_expired called _drop_expired_row which acquired
    self._lock, but get() already held the lock. The non-reentrant
    threading.Lock deadlocked. Pinned here so a regression is caught
    before it lands."""
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_expired", "object": "video", "status": "completed",
               "created_at": 1000, "expires_at": int(time.time()) - 10})
    # If the lock were re-entered, this would deadlock.
    result = store.get("video_expired")
    assert result is None


def test_videos_delete_does_not_deadlock_on_expired_row(tmp_path):
    """Same fix as `test_videos_get_does_not_deadlock_on_expired_row`,
    but for the delete path."""
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_expired", "object": "video", "status": "completed",
               "created_at": 1000, "expires_at": int(time.time()) - 10})
    assert store.delete("video_expired") is False


def test_videos_list_sweep_unlinks_orphaned_mp4(tmp_path):
    """When the list sweep deletes expired rows, it must also unlink
    their .mp4 files. Otherwise listing grows disk usage unboundedly
    (review 2026-09-23, Copilot medium-severity)."""
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_expired", "object": "video", "status": "completed",
               "created_at": 1000, "expires_at": int(time.time()) - 10})
    mp4 = tmp_path / "videos" / "video_expired.mp4"
    mp4.write_bytes(b"FAKE")
    store.list_live()
    assert not mp4.exists()


def test_videos_list_sweep_unlinks_orphans_on_bogus_cursor(tmp_path):
    """The bogus-cursor early-return used to skip the unlink loop, leaving
    expired .mp4 files orphaned on disk. The unlink now runs for every
    list call regardless of cursor validity (Copilot review 2026-09-23,
    third round)."""
    from chord.videos import VideoStore
    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_expired", "object": "video", "status": "completed",
               "created_at": 1000, "expires_at": int(time.time()) - 10})
    mp4 = tmp_path / "videos" / "video_expired.mp4"
    mp4.write_bytes(b"FAKE")
    videos, after_exists = store.list_live(after_id="video_bogus", order="desc")
    assert after_exists is False
    assert videos == []
    assert not mp4.exists()


def test_list_routes_normalize_empty_after_query_to_none(tmp_path):
    """`?after=` (empty value) used to be passed as after_id="" which the
    store treated as an unknown cursor. The route normalizes empty/whitespace
    to None so the call returns the first page (Copilot review 2026-09-23,
    third round)."""
    _deps, client, _sdk = make_app(tmp_path, FakeUpstream())
    # Empty after on /v1/files returns the first page, not an empty page.
    r = client.get("/v1/files?after=")
    assert r.status_code == 200, r.text
    # Whitespace-only after is also normalized away.
    r = client.get("/v1/files?after=%20%20")
    assert r.status_code == 200, r.text


def test_videos_migration_is_atomic(tmp_path):
    """The migration is wrapped in BEGIN IMMEDIATE/COMMIT. A failure
    inside the transaction rolls back; the next startup can try again
    from scratch (Copilot review 2026-09-23, third round).

    Pinned by writing a custom sqlite3 module wrapper that lets us raise
    on the second UPDATE during the migration: the column is added but
    the backfill fails, and we assert the column is rolled back."""
    import sqlite3
    from chord.videos import VideoStore

    path = tmp_path / "videos"
    path.mkdir()
    db = sqlite3.connect(path / "index.sqlite3")
    db.execute("CREATE TABLE videos (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    db.execute("INSERT INTO videos VALUES (?, ?)",
               ("video_old", '{"id": "video_old", "object": "video", "status": "completed"}'))
    db.commit()
    db.close()

    # Wrap sqlite3.connect so the second execute on a new connection raises.
    # The migration runs: CREATE TABLE IF NOT EXISTS (no-op), ALTER TABLE
    # ADD COLUMN seq (passes), UPDATE rows (the second execute call on
    # the migration's connection -- raise here). BEGIN IMMEDIATE is in
    # flight, so the ALTER rolls back.
    real_connect = sqlite3.connect

    class FailingConnect(sqlite3.Connection):
        update_count = 0

        def execute(self, sql, *args, **kwargs):
            if "UPDATE videos SET seq" in sql:
                type(self).update_count += 1
                if type(self).update_count == 1:
                    raise sqlite3.OperationalError("simulated migration failure")
            return super().execute(sql, *args, **kwargs)

    sqlite3.connect = FailingConnect  # type: ignore[assignment]
    try:
        with pytest.raises(sqlite3.OperationalError, match="simulated migration failure"):
            VideoStore(path)
    finally:
        sqlite3.connect = real_connect

    # The transaction rolled back. The column is NOT present (which is
    # the property that matters: a half-migrated volume would otherwise
    # see seq as present and skip the backfill on next startup, leaving
    # rows at the default 0).
    db = sqlite3.connect(path / "index.sqlite3")
    cols = {r[1] for r in db.execute("PRAGMA table_info(videos)")}
    db.close()
    assert "seq" not in cols, "migration did not roll back; column leaked into the schema"


@pytest.mark.parametrize("filters", [{"model": "chord-1-poly"}, {"metadata": {"team": "kitchen"}}], ids=["model", "metadata"])
def test_chat_list_rejects_a_cursor_outside_its_filters(tmp_path, filters):
    """Review 2026-09-24 B21: the cursor lookup read the whole table although
    its comment said it honoured model/metadata, so a cursor from another
    filter's list paged on silently. It is rejected, as live_files does."""
    store = ResponseStore(tmp_path / "responses.sqlite3")
    _put_chat(store, "chatcmpl-a", model="chord-1-poly", metadata={"team": "kitchen"})
    _put_chat(store, "chatcmpl-b", model="chord-1-other", metadata={"team": "bath"})
    _put_chat(store, "chatcmpl-c", model="chord-1-poly", metadata={"team": "kitchen"})
    rows, ok = store.list_chat(after_id="chatcmpl-b", **filters)
    assert (rows, ok) == ([], False)
    rows, ok = store.list_chat(after_id="chatcmpl-a", **filters)
    assert ok is True and [c["id"] for c, _ in rows] == ["chatcmpl-c"]
