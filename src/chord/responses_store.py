"""Stored responses (Responses API, #146): the durable half of `store`.

One SQLite file on the service's data volume. A stored row keeps three things:
the public `Response` object exactly as returned, the input items with the ids
we assigned (for GET /responses/{id}/input_items), and the whole conversation
as chat messages up to and including this response's own output (so a later
`previous_response_id` continues it without the caller resending anything).

Retention: kept at least RETENTION_S (30 days, the floor OpenAI documents);
an expired row is gone for every reader, deleted on read. Delete is a real
delete of the row, never a flag.

Conversations (Phase 4) live in the same file: a conversation row (id,
created_at, metadata) and its ordered items, kept until deleted (#142).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .store_schema import STORES

RETENTION_S = 30 * 24 * 3600


class DuplicateItemId(Exception):
    """append_items refused a batch: `item_id` is already in the conversation,
    or twice in the batch. Nothing of the batch was written."""

    def __init__(self, item_id: str) -> None:
        super().__init__(f"an item with id '{item_id}' is already in this conversation")
        self.item_id = item_id


class ResponseStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        schema = STORES["responses"]
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if not schema.readable_min <= version <= schema.readable_max:
            self._db.close()
            raise RuntimeError(f"responses store schema version {version} is not readable")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                   id TEXT PRIMARY KEY,
                   created_at INTEGER NOT NULL,
                   response TEXT NOT NULL,
                   input_items TEXT NOT NULL,
                   conversation TEXT NOT NULL)"""
        )

        # The events a streamed response actually sent, for GET ?stream=true resumes.
        if "events" not in {r[1] for r in self._db.execute("PRAGMA table_info(responses)")}:
            self._db.execute("ALTER TABLE responses ADD COLUMN events TEXT")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                   id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, metadata TEXT NOT NULL)"""
        )
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS conversation_items (
                   conversation_id TEXT NOT NULL, seq INTEGER NOT NULL, item_id TEXT NOT NULL, item TEXT NOT NULL,
                   PRIMARY KEY (conversation_id, seq))"""
        )

        self._db.execute(
            """CREATE TABLE IF NOT EXISTS chat_completions (
                   id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, seq INTEGER NOT NULL, model TEXT NOT NULL,
                   metadata TEXT NOT NULL, completion TEXT NOT NULL, messages TEXT NOT NULL)"""
        )

        self._db.execute(
            """CREATE TABLE IF NOT EXISTS files (
                   id TEXT PRIMARY KEY, seq INTEGER NOT NULL, created_at INTEGER NOT NULL,
                   expires_at INTEGER, file TEXT NOT NULL)"""
        )

        # The retention sweeps run inside EVERY read and write of these tables,
        # and MAX(seq) runs inside every insert. Without indexes each is a full
        # table scan on the event loop, growing forever with 30 days of stored
        # traffic (review 2026-09-22). IF NOT EXISTS migrates a live volume.
        self._db.execute("CREATE INDEX IF NOT EXISTS responses_created_at ON responses (created_at)")
        self._db.execute("CREATE INDEX IF NOT EXISTS chat_completions_created_at ON chat_completions (created_at)")
        self._db.execute("CREATE INDEX IF NOT EXISTS chat_completions_seq ON chat_completions (seq)")
        self._db.execute("CREATE INDEX IF NOT EXISTS files_seq ON files (seq)")
        # All migrations above are idempotent. If startup stops before this
        # marker, the next opener safely retries them from version 0.
        self._db.execute(f"PRAGMA user_version={schema.write_version}")

    # --- files (files_api.py) -----------------------------------------------------------
    # The row holds the public File object; the bytes live at files_dir/<id>. An expired
    # file is gone for every reader: its row and its bytes are deleted on the next read.

    def put_file(self, file: dict) -> None:
        with self._lock:
            seq = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM files").fetchone()[0]
            self._db.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?)",
                             (file["id"], seq, file["created_at"], file.get("expires_at"), json.dumps(file)))

    def live_files(self, *, after_id: str | None = None, limit: int | None = None,
                   order: str = "desc", purpose: str | None = None
                   ) -> tuple[list[dict], list[str], bool]:
        """(files in requested order, expired-and-removed ids, after_exists).

        `after_exists` mirrors the `list_chat` contract: True when `after_id`
        was in the table at the start of the call, False when it never was
        (the caller 400s on a bogus cursor instead of silently returning empty).

        Expired rows are deleted at the start of every call (same retention
        contract as before), then the rest are paged in SQL by `seq` with an id
        cursor for `after`. `purpose` is matched against the JSON column with
        `json_extract` (review 2026-09-23)."""
        if order not in ("asc", "desc"):
            raise ValueError(f"order must be 'asc' or 'desc', got {order!r}")
        with self._lock:
            now = int(time.time())
            gone = [r[0] for r in self._db.execute(
                "SELECT id FROM files WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            )]
            self._db.execute("DELETE FROM files WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
            # Look up the cursor's seq so the WHERE clause seeks past
            # (seq, id), not just id. Comparing ids alone skips rows whose
            # seq comes before but whose id sorts after, on the pages
            # where seq and id are not aligned (review 2026-09-23, Copilot
            # high-severity). The cursor check uses the purpose filter so
            # a cursor from a different purpose is rejected (Copilot
            # medium-severity).
            cursor_seq: int | None = None
            if after_id is not None:
                where_for_cursor = ["id = ?"]
                cursor_args: list = [after_id]
                if purpose is not None:
                    where_for_cursor.append("json_extract(file, '$.purpose') = ?")
                    cursor_args.append(purpose)
                row = self._db.execute(
                    f"SELECT seq FROM files WHERE {' AND '.join(where_for_cursor)}",
                    cursor_args,
                ).fetchone()
                if row is None:
                    return [], gone, False
                cursor_seq = row[0]
            clauses = ["1=1"]
            params: list = []
            if purpose is not None:
                clauses.append("json_extract(file, '$.purpose') = ?")
                params.append(purpose)
            if after_id is not None:
                assert cursor_seq is not None
                if order == "asc":
                    clauses.append("(seq > ? OR (seq = ? AND id > ?))")
                    params.extend([cursor_seq, cursor_seq, after_id])
                else:
                    clauses.append("(seq < ? OR (seq = ? AND id < ?))")
                    params.extend([cursor_seq, cursor_seq, after_id])
            sql = (
                "SELECT id, file FROM files "
                f"WHERE {' AND '.join(clauses)} "
                f"ORDER BY seq {order.upper()}, id {order.upper()} "
            )
            args = list(params)
            if limit is not None:
                sql += "LIMIT ?"
                args.append(limit)
            rows = self._db.execute(sql, args).fetchall()
        return [json.loads(r[1]) for r in rows], gone, True

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM files WHERE id = ?", (file_id,)).rowcount == 1

    # --- stored chat completions (S13e) ------------------------------------------------

    def put_chat(self, completion: dict, metadata: dict, messages: list[dict]) -> None:
        with self._lock:
            seq = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM chat_completions").fetchone()[0]
            self._db.execute("INSERT OR REPLACE INTO chat_completions VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (completion["id"], int(time.time()), seq, completion["model"], json.dumps(metadata),
                              json.dumps(completion), json.dumps(messages)))

    def _expire_chats(self) -> None:
        self._db.execute("DELETE FROM chat_completions WHERE created_at < ?", (time.time() - RETENTION_S,))

    def list_chat(self, *, after_id: str | None = None, limit: int | None = None,
                  order: str = "asc", model: str | None = None,
                  metadata: dict[str, str] | None = None
                  ) -> tuple[list[tuple[dict, dict]], bool]:
        """(rows, after_exists). `after_exists` is True when `after_id` was
        found in the list the model/metadata filters select; False when it is
        outside them, never existed, or was already swept by retention. The caller uses it to raise a clean 400 rather
        than silently returning an empty page for a bogus cursor (review
        2026-09-23).

        The list route serves only `completion` and `metadata`, not `messages`,
        so the SQL SELECT reads only those two columns -- the `messages` blob
        never leaves the row. WHERE/LIMIT/ORDER are pushed into SQLite: a 30-day
        table returns `limit` rows, not `count(*)`, on the event loop.

        `after_id` is an opaque id cursor. `metadata` is a flat equality filter
        matched against the JSON column with `json_extract`. `model` filters on
        the indexed string column. The result excludes expired rows; the
        retention sweep still runs on every call to keep `list` honest about
        what is on disk."""
        if order not in ("asc", "desc"):
            raise ValueError(f"order must be 'asc' or 'desc', got {order!r}")
        with self._lock:
            self._expire_chats()
            clauses = ["1=1"]
            params: list = []
            if model is not None:
                clauses.append("model = ?")
                params.append(model)
            if metadata:
                for k, v in metadata.items():
                    # Quote the key segment so dots and other JSON-path syntax
                    # do not change the lookup semantics. The metadata
                    # validator allows arbitrary keys up to 64 chars (any
                    # string), so without quoting `team.name` would be parsed
                    # as a nested path, returning NULL for the flat key the
                    # caller actually stored (review 2026-09-23, Copilot
                    # medium-severity).
                    clauses.append("json_extract(metadata, ?) = ?")
                    params.append(f'$."{k}"')
                    params.append(v)
            # When the caller supplied a cursor, look up its (seq, id) so the
            # WHERE clause can seek past the cursor's full position, not just
            # the id. Comparing ids alone would skip rows whose seq comes
            # before but whose id sorts after, on the pages where seq and id
            # are not aligned (review 2026-09-23, Copilot high-severity).
            cursor_seq: int | None = None
            if after_id is not None:
                # Under the SAME model/metadata filters as the page: a cursor
                # from another filter's list is not in this one. The lookup
                # used to read the whole table while this comment said it was
                # filtered (review 2026-09-24 B21), unlike live_files.
                row = self._db.execute(
                    f"SELECT seq FROM chat_completions WHERE {' AND '.join(clauses)} AND id = ?",
                    [*params, after_id],
                ).fetchone()
                if row is None:
                    # The cursor is not in this filtered list (another
                    # model/metadata filter's row, or simply unknown). The
                    # route 400s on this before serializing a page.
                    return [], False
                cursor_seq = row[0]
            if after_id is not None:
                assert cursor_seq is not None
                # Seek past (cursor_seq, cursor_id) in the chosen order.
                # (seq, id) is a stable total order over the table.
                if order == "asc":
                    clauses.append("(seq > ? OR (seq = ? AND id > ?))")
                    params.extend([cursor_seq, cursor_seq, after_id])
                else:
                    clauses.append("(seq < ? OR (seq = ? AND id < ?))")
                    params.extend([cursor_seq, cursor_seq, after_id])
            sql = (
                "SELECT id, completion, metadata FROM chat_completions "
                f"WHERE {' AND '.join(clauses)} "
                f"ORDER BY seq {order.upper()}, id {order.upper()} "
            )
            args = list(params)
            if limit is not None:
                sql += "LIMIT ?"
                args.append(limit)
            rows = self._db.execute(sql, args).fetchall()
        return [(json.loads(r[1]), json.loads(r[2])) for r in rows], True

    def get_chat(self, completion_id: str) -> tuple[dict, dict, list[dict]] | None:
        with self._lock:
            self._expire_chats()
            row = self._db.execute(
                "SELECT completion, metadata, messages FROM chat_completions WHERE id = ?",
                (completion_id,)).fetchone()
        return (json.loads(row[0]), json.loads(row[1]), json.loads(row[2])) if row else None

    def update_chat_metadata(self, completion_id: str, metadata: dict) -> bool:
        with self._lock:
            return self._db.execute("UPDATE chat_completions SET metadata = ? WHERE id = ?",
                                    (json.dumps(metadata), completion_id)).rowcount == 1

    def delete_chat(self, completion_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM chat_completions WHERE id = ?", (completion_id,)).rowcount == 1

    # --- conversations -------------------------------------------------------------

    def create_conversation(self, conversation_id: str, metadata: dict, items: list[dict]) -> dict:
        created = int(time.time())
        with self._lock:
            self._db.execute("INSERT INTO conversations VALUES (?, ?, ?)", (conversation_id, created, json.dumps(metadata)))
        self.append_items(conversation_id, items)
        resource = self.conversation_resource(conversation_id)
        assert resource is not None   # the row was inserted two statements above
        return resource

    def conversation_resource(self, conversation_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT created_at, metadata FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not row:
            return None
        return {"id": conversation_id, "object": "conversation", "created_at": row[0], "metadata": json.loads(row[1])}

    def update_conversation(self, conversation_id: str, metadata: dict) -> dict | None:
        with self._lock:
            n = self._db.execute("UPDATE conversations SET metadata = ? WHERE id = ?",
                                 (json.dumps(metadata), conversation_id)).rowcount
        return self.conversation_resource(conversation_id) if n else None

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            n = self._db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount
            self._db.execute("DELETE FROM conversation_items WHERE conversation_id = ?", (conversation_id,))
        return n == 1

    def append_items(self, conversation_id: str, items: list[dict]) -> bool:
        """Append in order; False, writing nothing, when the conversation is
        gone. A turn that finishes after its conversation was DELETEd used to
        leave rows no route can reach (review 2026-09-24 B21). The check and
        the insert share the lock, so a delete cannot land between them.

        Raises DuplicateItemId, writing nothing, when an item id is already in
        the conversation or repeated in `items`. This check is the guarantee;
        the routes' preflight only spares the model call for an obvious
        duplicate. A preflight alone let two requests carrying the same id
        both pass while each awaited its model, then both insert (Copilot on
        #332)."""
        with self._lock:
            if self._db.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is None:
                return False
            seen = {r[0] for r in self._db.execute("SELECT item_id FROM conversation_items WHERE conversation_id = ?",
                                                   (conversation_id,))}
            for item in items:
                if item["id"] in seen:
                    raise DuplicateItemId(item["id"])
                seen.add(item["id"])
            start = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM conversation_items WHERE conversation_id = ?",
                                     (conversation_id,)).fetchone()[0]
            self._db.executemany("INSERT INTO conversation_items VALUES (?, ?, ?, ?)",
                                 [(conversation_id, start + i, item["id"], json.dumps(item)) for i, item in enumerate(items)])
        return True

    def conversation_items(self, conversation_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT item FROM conversation_items WHERE conversation_id = ? ORDER BY seq",
                                    (conversation_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def delete_item(self, conversation_id: str, item_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM conversation_items WHERE conversation_id = ? AND item_id = ?",
                                    (conversation_id, item_id)).rowcount >= 1

    # --- responses -----------------------------------------------------------------

    def replace_response(self, response: dict) -> None:
        """Write a new public object for an existing id. Does not touch retention,
        input items, or the conversation: those belong to the original turn."""
        with self._lock:
            self._db.execute("UPDATE responses SET response = ? WHERE id = ?",
                             (json.dumps(response), response["id"]))

    def abandon_interrupted(self) -> None:
        """A queued or in-progress row belongs to a process that is gone.

        The spec's status has to become terminal. `failed` rather than `cancelled`:
        the caller did not cancel, the task is simply no longer running. EVERY
        non-terminal row, not only background ones: after a restart no worker
        exists for a plain streamed response orphaned by a disconnect either,
        and the background-only sweep left those in_progress forever."""
        with self._lock:
            rows = self._db.execute("SELECT id, response FROM responses").fetchall()
            for response_id, raw in rows:
                response = json.loads(raw)
                if response.get("status") in ("completed", "incomplete", "failed", "cancelled"):
                    continue
                response["status"] = "failed"
                response["error"] = {"code": "server_error", "message": "The response was interrupted before it finished."}
                self._db.execute("UPDATE responses SET response = ? WHERE id = ?",
                                 (json.dumps(response), response_id))

    def put(self, response: dict, input_items: list[dict], conversation: list[dict], events: list[dict] | None = None) -> None:
        with self._lock:
            self._expire_responses()
            self._db.execute(
                "INSERT OR REPLACE INTO responses (id, created_at, response, input_items, conversation, events) VALUES (?, ?, ?, ?, ?, ?)",
                (response["id"], int(time.time()), json.dumps(response), json.dumps(input_items), json.dumps(conversation),
                 json.dumps(events) if events is not None else None),
            )

    def put_if_present(self, response: dict, input_items: list[dict], conversation: list[dict],
                       events: list[dict] | None = None) -> bool:
        """Update an EXISTING row; False when the id is gone.

        A DELETEd response is a tombstone: a background turn still running
        must not resurrect it (2026-09-22, #1). INSERT OR REPLACE
        cannot tell 'never stored' from 'deleted on purpose'; the WHERE can."""
        with self._lock:
            self._expire_responses()
            return self._db.execute(
                "UPDATE responses SET created_at = ?, response = ?, input_items = ?, conversation = ?, events = ?"
                " WHERE id = ?",
                (int(time.time()), json.dumps(response), json.dumps(input_items), json.dumps(conversation),
                 json.dumps(events) if events is not None else None, response["id"])).rowcount == 1

    def _expire_responses(self) -> None:
        """Drop every row past retention, not only the id being read.

        Store defaults to true and a row can carry a base64 image. Expiring
        only inside `_row` left a create-and-never-retrieve response on the
        volume until something asked for that exact id."""
        self._db.execute("DELETE FROM responses WHERE created_at < ?", (time.time() - RETENTION_S,))

    def _row(self, response_id: str):
        with self._lock:
            self._expire_responses()
            return self._db.execute(
                "SELECT created_at, response, input_items, conversation, events FROM responses WHERE id = ?", (response_id,)
            ).fetchone()

    def get(self, response_id: str) -> dict | None:
        row = self._row(response_id)
        return json.loads(row[1]) if row else None

    def events(self, response_id: str) -> list[dict] | None:
        row = self._row(response_id)
        return json.loads(row[4]) if row and row[4] else None

    def input_items(self, response_id: str) -> list[dict] | None:
        row = self._row(response_id)
        return json.loads(row[2]) if row else None

    def conversation(self, response_id: str) -> list[dict] | None:
        row = self._row(response_id)
        return json.loads(row[3]) if row else None

    def delete(self, response_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM responses WHERE id = ?", (response_id,)).rowcount == 1
