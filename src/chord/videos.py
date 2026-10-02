"""First-draft OpenAI-compatible video creation backed by ComfyUI.

The public request and response speak the OpenAI Videos contract.  Create
validates, stores a `queued` row and returns it; a background task moves the row
to `in_progress` and then to `completed` or `failed`.  Renders are serialised by
one lock inside the backend, so a queued job waits its turn rather than
competing for the GPU.  Every terminal row carries an `expires_at`.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from ulid import ULID

from .http_transport import BODY_SLACK, read_body_within_cap
from .store_schema import STORES


# A reference frame, not a file upload. Same ceiling as a transcription: large
# enough for the sizes this route accepts, small enough to reject before the
# whole body is resident.
MAX_REFERENCE_BYTES = 25 * 1024 * 1024
# How long a terminal video row (completed OR failed) stays retrievable, matching
# the spec's 24h rather than a shorter local guess: a caller that retrieves the
# object and then fetches content and a thumbnail must not race the expiry.
# Failed rows get the same window; the spec keeps a failed job retrievable too,
# and its error is the whole reason the caller polls.
# Every terminal row gets one: `_drop_expired` only acts on an int, so a row left
# with `expires_at: None` is retained forever and keeps appearing in GET /v1/videos.
RETENTION_S = 86_400
# Admission bound for the render queue. Renders serialise on one GPU, every
# queued job pins its reference (up to MAX_REFERENCE_BYTES) in its task
# closure, and image edits wait on the same lock INLINE in their HTTP request:
# an unbounded queue let any authenticated caller hold the GPU for days and
# stack gigabytes of closures behind it (review 2026-09-22). A job past the
# bound gets a 429 error envelope; the spec knows no queue-full status.
MAX_OUTSTANDING = 4
# A startup cancel of an orphaned ComfyUI prompt that did not settle (ComfyUI
# unreachable, a stop refused) is retried after these waits, in seconds; a
# marker still standing after the last one is retried at the next start
# (Copilot on #336).
ORPHAN_RETRY_DELAYS: tuple[float, ...] = (10.0, 60.0, 300.0)
# What the admission count filters on, spelled once so the index below and the
# query are the same expression and SQLite can use one for the other.
_STATUS = "json_extract(record, '$.status')"
class _RowDeleted(RuntimeError):
    """The row was deleted while its render waited for the GPU: abort without
    a trace -- the deleter owns the record now."""


MODELS = {"sora-2", "sora-2-pro"}
SECONDS = {4, 8, 12}
SIZES = {
    "720x1280": (720, 1280),
    "1280x720": (1280, 720),
    "1024x1792": (1024, 1792),
    "1792x1024": (1792, 1024),
}
FIELDS = {"prompt", "input_reference", "model", "seconds", "size"}
# The pinned spec's maxLength on ImageRefParam.image_url (20 MiB of URL text).
# Checked before decoding so the base64 expansion is bounded by it too.
IMAGE_URL_MAX_CHARS = 20971520


def video_tools_available() -> bool:
    """Video output requires operator-provided ffmpeg and ffprobe on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _scrub_mp4_metadata(data: bytes) -> bytes:
    """Remux only the media streams; no provider prompt or workflow metadata leaves Chord."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("video metadata scrubber is unavailable")
    with tempfile.TemporaryDirectory(prefix="chord-video-") as directory:
        source = Path(directory) / "source.mp4"
        clean = Path(directory) / "clean.mp4"
        source.write_bytes(data)
        try:
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
                 "-map", "0:v", "-map", "0:a?", "-map_metadata", "-1", "-map_metadata:s", "-1",
                 "-map_chapters", "-1", "-c", "copy", "-movflags", "+faststart", str(clean)],
                capture_output=True, timeout=120, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("video metadata scrub failed") from exc
        if result.returncode != 0 or not clean.is_file() or clean.stat().st_size == 0:
            raise ValueError("video metadata scrub failed")
        return clean.read_bytes()
# A data URI's image subtype becomes part of a filename, so it must be a MIME
# token (RFC 6838): alphanumerics plus !#$&^_.+- and nothing else. Applied
# after lowercasing. Deliberately excludes / \ " ' whitespace and every control
# character, which is what makes the derived filename safe to hand to an upload.
_MIME_SUBTYPE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*")
logger = logging.getLogger(__name__)
_VIDEO_ID = re.compile(r"video_[0-9a-z]+")


class VideoStore:
    """The video object the spec says is retrievable until `expires_at`.

    The index is on the data volume, next to the MP4. An `in_progress` row at
    startup is a render this process did not finish, so it is `failed` rather
    than a status that never ends. A row past `expires_at` is gone for every
    reader, file included.
    """

    def __init__(self, root: Path, *, recover_interrupted: bool = True) -> None:
        """`recover_interrupted=False` is for a second opener in the SAME
        process (the daily retention sweep): there an `in_progress` row is a
        live render, not an interrupted one, and failing it would lie."""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(root / "index.sqlite3", check_same_thread=False, isolation_level=None)
        # WAL, as the responses store already uses: the retention sweep is a second
        # connection in the same process, and in rollback-journal mode its write
        # locks out the live route's reads for up to the 5s busy timeout, on the
        # event loop.
        self._db.execute("PRAGMA journal_mode=WAL")
        schema = STORES["videos"]
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if not schema.readable_min <= version <= schema.readable_max:
            self._db.close()
            raise RuntimeError(f"videos store schema version {version} is not readable")
        # Fresh volumes get the columns at table-creation time; old volumes
        # get them via ALTER TABLE below. Either path ends with the same
        # index, so a process crash between ALTER and CREATE INDEX cannot
        # leave the volume permanently without it (the index is idempotent
        # and runs after every column path -- review 2026-09-23).
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS videos (
                   id TEXT PRIMARY KEY,
                   record TEXT NOT NULL,
                   seq INTEGER NOT NULL DEFAULT 0,
                   created_at INTEGER NOT NULL DEFAULT 0,
                   expires_at INTEGER)"""
        )
        # ComfyUI prompts submitted and not yet known settled (review 2026-09-24
        # B20). Apart from the rows on purpose: DELETE and expiry remove rows,
        # and a prompt still rendering must outlive both until a stop settles
        # or a read finds it gone (Copilot on #336).
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS comfy_prompts (
                   prompt_id TEXT PRIMARY KEY,
                   video_id TEXT NOT NULL,
                   submitted_at INTEGER NOT NULL)"""
        )
        # Schema migration for volumes created before the columns existed.
        # Wrapped in a single transaction so a process crash between any
        # pair of statements rolls back the whole batch -- without this,
        # a crash after ALTER seq but before the seq backfill would leave
        # every old row at the default seq=0 and a subsequent startup
        # would see the column already present and skip the backfill
        # (Copilot review 2026-09-23, third round).
        self._db.execute("BEGIN IMMEDIATE")
        try:
            existing = {r[1] for r in self._db.execute("PRAGMA table_info(videos)")}
            if "seq" not in existing:
                self._db.execute("ALTER TABLE videos ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")
                # Backfill seq so old rows have a stable, monotonic-by-id order.
                # Without this, every old row gets seq=0 and the next insert
                # (seq=N+1) splits the table into "old" and "new" groups.
                existing_rows = self._db.execute("SELECT id FROM videos ORDER BY id").fetchall()
                for i, (vid,) in enumerate(existing_rows):
                    self._db.execute("UPDATE videos SET seq = ? WHERE id = ?", (i + 1, vid))
            if "created_at" not in existing:
                self._db.execute("ALTER TABLE videos ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0")
                self._db.execute(
                    "UPDATE videos SET created_at = COALESCE("
                    " CAST(json_extract(record, '$.created_at') AS INTEGER), created_at)"
                )
            if "expires_at" not in existing:
                self._db.execute("ALTER TABLE videos ADD COLUMN expires_at INTEGER")
                # Backfill from the JSON record. The SQL sweep below only looks
                # at the column; without this backfill an expired terminal row
                # in an old volume survives listing/retrieval forever.
                self._db.execute(
                    "UPDATE videos SET expires_at = CAST(json_extract(record, '$.expires_at') AS INTEGER) "
                    "WHERE json_extract(record, '$.expires_at') IS NOT NULL"
                )
            # Idempotent index creation. Runs unconditionally so a crash
            # between ALTER and CREATE INDEX cannot leave a volume without it
            # (Copilot review 2026-09-23). Now safe because the columns are
            # always present by this point.
            self._db.execute("CREATE INDEX IF NOT EXISTS videos_seq ON videos (seq, id)")
            # Every image edit and variation asks how deep the render queue is,
            # and a scan that JSON-parsed every retained row in Python made that
            # O(the day's videos) per request (Copilot on #336).
            self._db.execute(f"CREATE INDEX IF NOT EXISTS videos_status ON videos ({_STATUS})")
            self._db.execute(f"PRAGMA user_version={schema.write_version}")
            self._db.execute("COMMIT")
        except Exception:  # noqa: BLE001 — a migration failure must not leave a half-migrated volume; we ROLLBACK and re-raise so the next startup retries from scratch
            # A migration failure must not leave a half-migrated volume;
            # ROLLBACK restores the pre-migration state so the next startup
            # can try again from scratch.
            self._db.execute("ROLLBACK")
            raise
        # ComfyUI prompts a previous process submitted and never saw settle;
        # `register` cancels them once the app is serving (review 2026-09-24
        # B20). A second opener in the same process (the retention sweep)
        # recovers nothing: its in_progress rows are live renders (B22).
        self.orphaned_prompts: list[str] = self._fail_interrupted() if recover_interrupted else []

    def _fail_interrupted(self) -> list[str]:
        """Fail the rows the previous process left outstanding, and return
        every prompt not yet known settled. A prompt leaves the list only when
        its render finished here or a stop settled, so one whose cancel never
        landed is offered again, start after start (Copilot on #336)."""
        with self._lock:
            orphans = [pid for (pid,) in self._db.execute(
                "SELECT prompt_id FROM comfy_prompts ORDER BY submitted_at, prompt_id")]
            rows = self._db.execute("SELECT id, record FROM videos").fetchall()
            for video_id, raw in rows:
                record = json.loads(raw)
                if record.get("status") not in ("queued", "in_progress"):
                    continue
                record["status"] = "failed"
                record["expires_at"] = int(time.time()) + RETENTION_S
                record["error"] = {
                    "code": "generation_failed",
                    "message": "video generation was interrupted",
                    "misalignment": None,
                }
                self._db.execute(
                    "UPDATE videos SET record = ?, expires_at = ? WHERE id = ?",
                    (json.dumps(record), record["expires_at"], video_id),
                )
        return orphans

    def set_prompt_id(self, video_id: str, prompt_id: str) -> None:
        """Record a submitted ComfyUI prompt, outside the public record, so a
        restart can stop it (review 2026-09-24 B20). Recorded even when the row
        is already gone: the prompt still runs."""
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO comfy_prompts (prompt_id, video_id, submitted_at) "
                             "VALUES (?, ?, ?)", (prompt_id, video_id, int(time.time())))

    def clear_prompt_id(self, prompt_id: str) -> None:
        """The prompt is settled (finished here, or a stop landed): no start
        needs to stop it."""
        with self._lock:
            self._db.execute("DELETE FROM comfy_prompts WHERE prompt_id = ?", (prompt_id,))

    def put(self, record: dict) -> None:
        with self._lock:
            # Allocate the next seq atomically inside the same lock that holds
            # the INSERT, so concurrent writers cannot share a seq (review
            # 2026-09-23). MAX(seq)+1 on a single row is the simplest correct
            # form: SQLite serialises statements within a connection.
            seq = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM videos").fetchone()[0]
            self._db.execute(
                "INSERT OR REPLACE INTO videos (id, record, seq, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (record["id"], json.dumps(record), seq,
                 int(record.get("created_at") or time.time()),
                 record.get("expires_at")),
            )

    def put_if_present(self, record: dict) -> bool:
        """Update an EXISTING row; False when the id is gone.

        A DELETEd video is a tombstone: the render task's terminal write must
        not resurrect the row as `completed` after the caller deleted it (#1).
        INSERT OR REPLACE cannot tell 'never stored'
        from 'deleted on purpose'; the WHERE can.

        Keeps the original `seq` and `created_at` from the existing row so
        the index order does not shift on terminal writes."""
        with self._lock:
            return self._db.execute(
                "UPDATE videos SET record = ?, expires_at = ? WHERE id = ?",
                (json.dumps(record), record.get("expires_at"), record["id"])
            ).rowcount == 1

    def count_outstanding(self) -> int:
        """Queued or rendering rows: the admission queue's depth. Both statuses
        carry `expires_at: None` (only terminal rows expire), so the count needs
        no sweep. The whole table is 24h of index rows, and the record column
        is the only place status lives."""
        with self._lock:
            (count,) = self._db.execute(
                f"SELECT COUNT(*) FROM videos WHERE {_STATUS} IN ('queued', 'in_progress')").fetchone()
        return int(count)

    def get(self, video_id: str) -> dict | None:
        if _VIDEO_ID.fullmatch(video_id) is None:
            return None
        with self._lock:
            row = self._db.execute("SELECT record FROM videos WHERE id = ?", (video_id,)).fetchone()
            if row is None:
                return None
            record = json.loads(row[0])
            if self._drop_expired(video_id, record):
                return None
            return record

    def file(self, video_id: str) -> Path | None:
        record = self.get(video_id)
        if record is None or record.get("status") != "completed":
            return None
        path = self.root / f"{video_id}.mp4"
        return path if path.is_file() else None

    def _delete_expired_locked(self, video_id: str) -> bool:
        """Drop a row whose expires_at has passed. Caller already holds self._lock.

        Reads only the column index, not the JSON record, so the per-row cost
        is a primary-key probe. Returns True when the row was deleted.

        The mp4 is removed outside the lock: file deletion can take meaningful
        time and we do not want to hold the lock during it."""
        row = self._db.execute("SELECT expires_at FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row is None or row[0] is None or row[0] > int(time.time()):
            return False
        self._db.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        return True

    def _drop_expired(self, video_id: str, record: dict) -> bool:
        """Drop a row whose expires_at has passed. Caller already holds self._lock.

        `record` is read from the JSON column above; this helper trusts the
        caller to be inside the lock so we do not deadlock the request."""
        expires = record.get("expires_at")
        if isinstance(expires, int) and expires <= int(time.time()):
            if self._delete_expired_locked(video_id):
                (self.root / f"{video_id}.mp4").unlink(missing_ok=True)
                return True
        return False

    def list_live(self, *, after_id: str | None = None, limit: int | None = None,
                  order: str = "desc") -> tuple[list[dict], bool]:
        """(videos in order, after_exists).

        Default desc matches the route's default and the test that asserts
        newest-first. ORDER BY seq, id is index-backed; the per-row json.loads
        is what the route returns, so it stays (review 2026-09-23).

        The retention sweep runs FIRST so the page we return is post-sweep;
        we do not filter by expires_at in SQL because only terminal rows have
        one set, and the count of in-flight rows is unchanged by their own
        expiry.
        """
        if order not in ("asc", "desc"):
            raise ValueError(f"order must be 'asc' or 'desc', got {order!r}")
        with self._lock:
            expired = self._take_expired_locked()
            # Look up the cursor's seq so the WHERE clause seeks past
            # (seq, id), not just id. Comparing ids alone skips rows whose
            # seq comes before but whose id sorts after, on the pages
            # where seq and id are not aligned (review 2026-09-23, Copilot
            # high-severity).
            cursor_seq: int | None = None
            cursor_found = True
            if after_id is not None:
                row = self._db.execute(
                    "SELECT seq FROM videos WHERE id = ?", (after_id,),
                ).fetchone()
                if row is None:
                    # Cursor is not in the table at all. The route 400s on
                    # this; the page contents are whatever the SQL filter
                    # would produce, which is empty.
                    cursor_found = False
                else:
                    cursor_seq = row[0]
            rows: list = []
            if cursor_found:
                clauses = ["1=1"]
                params: list = []
                if after_id is not None:
                    assert cursor_seq is not None
                    if order == "asc":
                        clauses.append("(seq > ? OR (seq = ? AND id > ?))")
                        params.extend([cursor_seq, cursor_seq, after_id])
                    else:
                        clauses.append("(seq < ? OR (seq = ? AND id < ?))")
                        params.extend([cursor_seq, cursor_seq, after_id])
                sql = (
                    "SELECT id, record FROM videos "
                    f"WHERE {' AND '.join(clauses)} "
                    f"ORDER BY seq {order.upper()}, id {order.upper()} "
                )
                args = list(params)
                if limit is not None:
                    sql += "LIMIT ?"
                    args.append(limit)
                rows = self._db.execute(sql, args).fetchall()
        # Unlink the mp4 files outside the lock so a slow filesystem does not
        # stall concurrent reads, AND so the cleanup runs even on the bogus-
        # cursor path (Copilot review 2026-09-23, third round).
        self._unlink_expired(expired)
        return [json.loads(r[1]) for r in rows], cursor_found

    def _take_expired_locked(self) -> list[str]:
        """Delete every row past expires_at and return their ids. Caller holds
        self._lock; the ids are captured first because the DELETE returns no
        rows and the caller must unlink their MP4s outside the lock."""
        now = int(time.time())
        expired = [r[0] for r in self._db.execute(
            "SELECT id FROM videos WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )]
        self._db.execute(
            "DELETE FROM videos WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        return expired

    def _unlink_expired(self, expired: list[str]) -> None:
        # `missing_ok` because a retried upload can leave the row gone but the
        # bytes already removed. An unlink that fails is logged, not raised:
        # the row is already gone, so raising would only 500 the list that
        # triggered it, and sweep_expired reaps the file on its next pass.
        for video_id in expired:
            try:
                (self.root / f"{video_id}.mp4").unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("could not remove expired %s.mp4 (%s); the next sweep retries",
                               video_id, type(exc).__name__)

    def sweep_expired(self) -> int:
        """Drop every expired row and its MP4; the count of rows removed.

        The list route's own sweep, callable without a reader: expired MP4s
        used to leave the volume only when someone listed or fetched them, and
        the daily volume sweep never looked here (review 2026-09-24 B22).

        Then it reaps every MP4 with no row. The row is deleted before its
        file, so an unlink that failed left a file nothing named any more
        (Copilot on #334). An MP4 without a row is never live: create stores
        the row before the render task exists, and a render whose row was
        deleted removes its own file."""
        with self._lock:
            expired = self._take_expired_locked()
        self._unlink_expired(expired)
        on_disk = [p.stem for p in self.root.glob("video_*.mp4")]
        if on_disk:
            with self._lock:
                known = {r[0] for r in self._db.execute(
                    f"SELECT id FROM videos WHERE id IN ({','.join('?' * len(on_disk))})", on_disk)}
            self._unlink_expired([vid for vid in on_disk if vid not in known])
        return len(expired)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def delete(self, video_id: str) -> bool:
        if _VIDEO_ID.fullmatch(video_id) is None:
            return False
        with self._lock:
            row = self._db.execute("SELECT record FROM videos WHERE id = ?", (video_id,)).fetchone()
            if row is None:
                return False
            record = json.loads(row[0])
            if self._drop_expired(video_id, record):
                return False
            self._db.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            (self.root / f"{video_id}.mp4").unlink(missing_ok=True)
            return True


def _error(status: int, message: str, code: str, param: str | None = None,
           kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind, "param": param, "code": code}},
        status_code=status,
    )


def _decode_data_image(url: str) -> tuple[bytes, str] | str:
    """(bytes, filename) from a `data:image/...;base64,` URI, or an error string.

    The caller has already run the SSRF policy, which is what guarantees the
    scheme is `data:image/`; this only has to parse the media type and decode.
    Base64 is required: a percent-encoded data URI is text, not an image, and
    guessing which the caller meant is how a malformed reference becomes a
    confusing failure downstream. The decoded size is bounded twice -- by the
    URL length cap at the door and again here -- so a hostile payload cannot
    allocate past the reference ceiling."""
    header, sep, payload = url.partition(",")
    if not sep:
        return "image_url data URI is malformed: expected data:image/<subtype>;base64,<payload>"
    # Split the header into the media type and its parameters, then require
    # `base64` to BE a parameter -- not merely to occur inside one. A substring
    # check accepted `;base64evil` and would have decoded a payload the caller
    # never declared as base64 (Copilot review, #329). Comparison is per-segment
    # and case-insensitive, which is what RFC 2397 means by the marker.
    segments = header.split(";")
    mediatype = segments[0]
    if not any(param.lower() == "base64" for param in segments[1:]):
        return "image_url data URI must be base64-encoded"
    if not mediatype.lower().startswith("data:image/"):
        return "image_url data URI is malformed: expected data:image/<subtype>;base64,<payload>"
    # Deliberately NOT stripped: `.strip()` laundered `data:image/ png ;base64,`
    # into a valid token, so a subtype carrying whitespace -- which the MIME
    # grammar does not allow -- reached the renderer. Validating the raw text
    # makes whitespace fail on its own merits (Copilot review, #329).
    subtype = mediatype[len("data:image/"):].lower()
    if not subtype:
        return "image_url data URI names no image subtype"
    # The subtype becomes part of a FILENAME handed to ComfyUI's upload, so it
    # is validated as a MIME token rather than merely checked for non-emptiness.
    # `data:image/../../evil;base64,...` would otherwise put a path-shaped
    # string into the filename -- the multipart branch sanitizes with
    # os.path.basename for exactly this reason, and a data URI has no basename
    # to take. RFC 6838 token characters only: no slash, no backslash, no
    # quotes, no whitespace, no control characters (Copilot review, #329).
    if _MIME_SUBTYPE.fullmatch(subtype) is None:
        return "image_url data URI subtype is not a valid MIME token"
    try:
        data = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError):
        return "image_url data URI is not valid base64"
    if not data:
        return "image_url data URI is empty"
    if len(data) > MAX_REFERENCE_BYTES:
        return f"image_url must decode to at most {MAX_REFERENCE_BYTES} bytes"
    return data, f"reference.{subtype.split('+', 1)[0]}"


async def _fetch_reference(url: str) -> tuple[bytes, str] | JSONResponse:
    """Fetch an allowlisted http(s) reference image, bounded, redirects OFF.

    `follow_redirects=False` is the load-bearing part: the SSRF policy
    allowlists an exact host, and a 3xx from that host to an internal address
    is precisely the bypass the allowlist exists to prevent. A redirect is
    therefore a refusal, not a hop. The read is streamed and counted so a host
    that advertises a small body and sends a large one is cut off at the
    ceiling rather than after it is resident."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, connect=10.0),
        ) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    return _error(400, f"input_reference image_url answered {resp.status_code}",
                                  "invalid_value", "image_url")
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes(64 * 1024):
                    total += len(chunk)
                    if total > MAX_REFERENCE_BYTES:
                        return _error(413, f"input_reference must be at most {MAX_REFERENCE_BYTES} bytes",
                                      "file_too_large", "image_url")
                    chunks.append(chunk)
    except httpx.HTTPError:
        # Unreachable host, TLS failure, timeout: the caller named a URL we
        # could not read. Their input, their 400 -- never a 500, and never the
        # transport's own detail on the wire.
        return _error(400, "input_reference image_url could not be fetched",
                      "invalid_value", "image_url")
    data = b"".join(chunks)
    if not data:
        return _error(400, "input_reference image_url returned no bytes",
                      "invalid_value", "image_url")
    # A basename off a caller-controlled URL, so it is sanitized exactly the
    # way the multipart branch sanitizes an uploaded filename.
    name = os.path.basename(urlparse(url).path) or "reference.png"
    return data, name


class _RemoteReference(str):
    """An allowlisted http(s) reference URL whose fetch is DEFERRED.

    The fetch used to run while parsing the body, before the prompt check and
    the 429 admission check: a request that was going to be refused still made
    an outbound fetch of up to 25 MB (review 2026-09-24 B22). Parsing now
    validates the URL against the policy and hands this back; the create route
    fetches it only once every cheap check has passed."""


async def _image_url_reference(url: object, body: dict,
                               settings) -> tuple[dict, bytes | _RemoteReference, str | None] | JSONResponse:
    """Resolve `input_reference: {image_url}` to bytes, under chat's SSRF policy.

    The policy is imported from chat_api rather than restated: two doors that
    each hand-roll an allowlist check drift, and the drift is invisible until
    one of them is the bypass. Same rules, same refusal text, same refusal to
    enumerate the allowlist."""
    if not isinstance(url, str) or not url:
        return _error(400, "input_reference.image_url must be a non-empty string",
                      "invalid_value", "image_url")
    # The pinned spec caps the URL string at 20 MiB; checking the length before
    # decoding bounds the base64 expansion too.
    if len(url) > IMAGE_URL_MAX_CHARS:
        return _error(413, f"input_reference image_url must be at most {IMAGE_URL_MAX_CHARS} characters",
                      "file_too_large", "image_url")
    from .chat_api import _image_url_policy_error
    allowed = settings.image_url_allowed_hosts if settings is not None else frozenset()
    # The shared policy calls urlparse, which RAISES ValueError on a malformed
    # authority (`http://[::1` -- an unterminated IPv6 literal) rather than
    # returning a refusal. Unguarded, that is a 500 for input that is simply a
    # bad URL; malformed input must be a named 400 and never
    # a 500. Same envelope as every other refusal here, and the message is the
    # policy's own so the allowlist is still never enumerated (Copilot, #329).
    try:
        why = _image_url_policy_error({"image_url": {"url": url}}, allowed)
    except ValueError:
        return _error(400, "input_reference image_url is not a parseable URL",
                      "invalid_value", "image_url")
    if why is not None:
        return _error(400, why, "invalid_value", "image_url")
    if url.lower().startswith("data:"):
        decoded = _decode_data_image(url)
        if isinstance(decoded, str):
            return _error(400, decoded, "invalid_value", "image_url")
        data, name = decoded
        return body, data, name
    return body, _RemoteReference(url), None


async def _body(request: Request, store=None, files_dir: Path | None = None,
                settings=None) -> tuple[dict, bytes | _RemoteReference | None, str | None] | JSONResponse:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        # Bound ingress BEFORE the form parse spools an unbounded body to disk;
        # the per-part 25 MB check downstream can only refuse what already
        # landed (2026-09-22, #3). Same envelope as that check.
        raw = await read_body_within_cap(request, MAX_REFERENCE_BYTES + BODY_SLACK)
        if raw is None:
            return _error(413, f"input_reference must be at most {MAX_REFERENCE_BYTES} bytes",
                          "file_too_large", "input_reference")
        request._body = raw
        try:
            form = await request.form()
        except Exception:  # malformed multipart is the caller's input
            return _error(400, "the multipart body could not be parsed", "invalid_request")
        unknown = sorted(set(form.keys()) - FIELDS)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}",
                          "unknown_parameter", unknown[0])
        body = {key: form.get(key) for key in FIELDS if key != "input_reference" and form.get(key) is not None}
        upload = form.get("input_reference")
        if upload is None:
            return body, None, None
        if isinstance(upload, str) or len(form.getlist("input_reference")) != 1:
            return _error(400, "input_reference must be one uploaded image", "invalid_value", "input_reference")
        if upload.size is not None and upload.size > MAX_REFERENCE_BYTES:
            return _error(413, f"input_reference must be at most {MAX_REFERENCE_BYTES} bytes",
                          "file_too_large", "input_reference")
        data = await upload.read(MAX_REFERENCE_BYTES + 1)
        if len(data) > MAX_REFERENCE_BYTES:
            return _error(413, f"input_reference must be at most {MAX_REFERENCE_BYTES} bytes",
                          "file_too_large", "input_reference")
        if not data:
            return _error(400, "input_reference is empty", "invalid_value", "input_reference")
        return body, data, os.path.basename(upload.filename or "reference.png")

    try:
        body = await request.json()
    except (ValueError, RecursionError):
        return _error(400, "body is not JSON", "invalid_json")
    if not isinstance(body, dict):
        return _error(400, "body must be a JSON object", "invalid_json")
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        return _error(400, f"Unrecognized request argument supplied: {unknown[0]}",
                      "unknown_parameter", unknown[0])
    # The spec's ImageRefParam is exactly one of `image_url` or `file_id`. Both
    # resolve to reference bytes here, so the create route validates dimensions
    # identically whichever the caller chose (#321).
    ref = body.get("input_reference")
    if ref is None:
        return body, None, None
    if not isinstance(ref, dict):
        return _error(400, "input_reference must be an uploaded image or one file_id",
                      "unsupported_value", "input_reference")
    keys = set(ref)
    if keys == {"image_url"}:
        return await _image_url_reference(ref["image_url"], body, settings)
    file_id = ref.get("file_id")
    if keys != {"file_id"} or not isinstance(file_id, str):
        return _error(400, "input_reference must be an uploaded image or one file_id",
                      "unsupported_value", "input_reference")
    if store is None or files_dir is None:
        return _error(400, "input_reference must be uploaded as multipart", "unsupported_value", "input_reference")
    from .files_api import StoredFileTooLarge, load_stored_file
    try:
        # The multipart branch bounds its read; the stored-file branch must too,
        # or a 512 MB File object arrives whole in memory before anything can
        # measure it (review 2026-09-22). Same cap, same refusal as multipart.
        loaded = load_stored_file(store, files_dir, file_id, max_bytes=MAX_REFERENCE_BYTES)
    except StoredFileTooLarge:
        return _error(413, f"input_reference must be at most {MAX_REFERENCE_BYTES} bytes",
                      "file_too_large", "input_reference")
    if loaded is None:
        return _error(404, f"No such File object: {file_id}", "not_found", "file_id")
    file, data = loaded
    return body, data, file.get("filename") or "reference.png"


def _first_frame_png(path: Path) -> bytes | None:
    """One PNG frame from a stored MP4. ffmpeg is the decoder; nothing else here reads video."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    try:
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            capture_output=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.exception("thumbnail extraction failed for %s", path)
        return None
    if proc.returncode != 0 or not proc.stdout.startswith(b"\x89PNG"):
        logger.warning("thumbnail extraction failed for %s: %s", path, proc.stderr[:300])
        return None
    return proc.stdout


# The spritesheet grid: up to 16 evenly-spaced frames tiled 4x4, each scaled to
# SPRITE_W wide so the sheet stays a bounded size whatever the source
# resolution. Sampling by a step derived from the probed frame count is what
# makes the sheet describe the WHOLE clip rather than its first second.
SPRITESHEET_GRID = 4
SPRITESHEET_FRAMES = SPRITESHEET_GRID * SPRITESHEET_GRID
SPRITE_W = 160
# Extraction can shell out to ffmpeg twice (probe + encode); the thumbnail's
# 20s ceiling was for one pass. Same order of magnitude, doubled.
SPRITESHEET_TIMEOUT_S = 40
# Sanity ceiling on a probed frame count. This backend renders at most 12
# seconds, so even an hour at 120 fps would be far under this; a count above it
# is corrupt metadata rather than a video, and honouring it would make the
# sampling step astronomical and select frame 0 alone.
MAX_PLAUSIBLE_FRAMES = 10_000_000


def sprite_step(frames: int) -> int:
    """The sampling step that selects AT MOST `SPRITESHEET_FRAMES` frames.

    Ceiling, not floor. Floor division over-selects whenever the count is not
    a multiple of the grid: 100 frames at step 6 selects 17 and `tile=4x4`
    keeps the first 16, so the sheet stops at frame 90; 20 frames at step 1
    selects all 20 and the sheet shows only frames 0-15, never reaching the
    end of the clip at all. Ceiling guarantees at most 16 selected frames and
    therefore that the last one is near the end (Copilot review, #329)."""
    return max(1, -(-frames // SPRITESHEET_FRAMES))


def _frames_from_probe(stream: dict) -> int | None:
    """Frame count from one ffprobe stream entry. Pure -- no I/O -- so the
    `nb_frames`-absent fallback is directly testable instead of only reachable
    through a container that happens to omit the field (Copilot review, #329).

    `nb_frames` is the cheap answer and is present on the MP4s this backend
    writes. Where a container omits it, or reports it as `N/A`, `duration *
    avg_frame_rate` is the honest fallback. None when neither can be read."""
    raw = stream.get("nb_frames")
    if raw not in (None, "", "N/A"):
        count = int(raw)
        # An absurd count is unreadable metadata, not a video: a 400-digit
        # nb_frames would make the sampling step astronomical and select frame
        # 0 alone. Refuse it here rather than build a nonsense sheet.
        return count if 0 < count <= MAX_PLAUSIBLE_FRAMES else None
    # Fallback: duration * average frame rate. avg_frame_rate is a "num/den"
    # fraction; a zero denominator means "unknown", not zero. `duration` can be
    # the string "inf" on a corrupt container, and float("inf") survives every
    # comparison below before OverflowError-ing in int() -- so finiteness is
    # checked explicitly rather than left to the caller's handler.
    duration = float(stream.get("duration") or 0)
    if not math.isfinite(duration):
        return None
    rate = str(stream.get("avg_frame_rate") or "0/1")
    num, _, den = rate.partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    if not math.isfinite(fps) or duration <= 0 or fps <= 0:
        return None
    count = int(round(duration * fps))
    return count if 0 < count <= MAX_PLAUSIBLE_FRAMES else None


def _probe_frame_count(path: Path) -> int | None:
    """The video's frame count, from container metadata -- no decode.

    None when ffprobe is missing, the probe fails, or the shape is unreadable,
    which the caller treats as "cannot make a sheet" (a 502, never a 500)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.exception("frame probe failed for %s", path)
        return None
    if proc.returncode != 0:
        logger.warning("frame probe failed for %s: %s", path, proc.stderr[:300])
        return None
    # ffprobe's output is an external tool's answer about a caller-supplied
    # container, so every shape assumption is checked and the handler is wide.
    # Narrow typing here was the bug: a top-level JSON list made `.get` raise
    # AttributeError and `duration: "inf"` made int() raise OverflowError, and
    # neither was in the caught set -- both escaped to the route as a 500,
    # contradicting this function's own contract (Copilot review, #329).
    try:
        parsed = json.loads(proc.stdout)
        if not isinstance(parsed, dict):
            logger.warning("frame probe returned a non-object for %s", path)
            return None
        streams = parsed.get("streams")
        if not isinstance(streams, list) or not streams:
            return None
        stream = streams[0]
        if not isinstance(stream, dict):
            logger.warning("frame probe stream entry is not an object for %s", path)
            return None
        return _frames_from_probe(stream)
    except Exception:
        # Deliberately broad: this is the boundary that turns "we could not read
        # this container" into a 502. Any exception at all -- including one from
        # a metadata shape nobody predicted -- must map to None, because the
        # alternative is a 500 on a request that was perfectly well formed.
        logger.warning("frame probe returned an unreadable shape for %s", path, exc_info=True)
        return None


def _spritesheet_png(path: Path) -> bytes | None:
    """A 4x4 grid of frames sampled evenly across a stored MP4, as one PNG.

    ffmpeg does both halves: `select` keeps every `step`-th frame and `tile`
    lays them out. `step` comes from the probed frame count so the grid
    spans the whole clip at any duration -- a fixed step would show only the
    opening seconds of a long render and a fixed fps would overflow the grid.
    None when ffmpeg or ffprobe is missing, the probe is unreadable, or the
    encode fails; the route turns that into a 502, never a 500."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    frames = _probe_frame_count(path)
    if frames is None:
        return None
    step = sprite_step(frames)
    # A clip shorter than the grid samples every frame it has (step 1) and the
    # tile is partly filled -- still an honest sheet, and `-frames:v 1` takes
    # the single tiled output ffmpeg emits.
    # `\,` escapes the comma inside the select expression for the shell-free
    # argv form ffmpeg expects.
    vf = (f"select='not(mod(n\\,{step}))',"
          f"scale={SPRITE_W}:-1,tile={SPRITESHEET_GRID}x{SPRITESHEET_GRID}")
    try:
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-vf", vf,
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            capture_output=True, timeout=SPRITESHEET_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.exception("spritesheet extraction failed for %s", path)
        return None
    if proc.returncode != 0 or not proc.stdout.startswith(b"\x89PNG"):
        logger.warning("spritesheet extraction failed for %s: %s", path, proc.stderr[:300])
        return None
    return proc.stdout


def _reference_dimensions(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            return image.size
    # DecompressionBombError is Pillow refusing a tiny file whose IHDR claims
    # absurd dimensions (a 66-byte PNG can declare 196M pixels). It inherits
    # from Exception alone -- neither name below catches it -- so an uncaught
    # one left the caller with a 500 for input that is simply not a readable
    # image (review 2026-09-22). Malformed input is a 400, never a 500.
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return None


def register(app: FastAPI, deps, store=None) -> None:
    videos = VideoStore(deps.settings.data_dir / "videos")
    # Live render tasks by video id, so DELETE cancels the work instead of
    # only removing the row (review 2026-09-22, #1: the row came back, and
    # create-4/delete-4 loops stacked orphaned GPU renders past MAX_OUTSTANDING
    # because admission counted rows while the lock queued tasks).
    renders: dict[str, asyncio.Task] = {}
    from .render_admission import admission_for
    admission = admission_for(app)
    admission.video_rows = videos.count_outstanding

    # Review 2026-09-24 B20: the rows failed at startup may have left prompts
    # running on ComfyUI (which other producers may also use). Cancel each the
    # safe way, once the app serves: a background task, best effort, logged,
    # never holding startup.
    orphans = list(videos.orphaned_prompts)
    recovery: list[asyncio.Task] = []

    async def _cancel_orphans() -> None:
        backend = getattr(deps, "video_backend", None)
        cancel = getattr(backend, "cancel_orphan", None)
        if cancel is None:
            if orphans:
                logger.warning("no video backend to cancel %d orphaned ComfyUI prompt(s)", len(orphans))
            return
        waiting = list(orphans)
        for delay in (0.0, *ORPHAN_RETRY_DELAYS):
            if not waiting:
                return
            await asyncio.sleep(delay)
            unsettled = []
            for prompt_id in waiting:
                try:
                    settled = await cancel(prompt_id) is True
                except Exception:
                    logger.exception("cancelling orphaned ComfyUI prompt %s failed", prompt_id)
                    settled = False
                if settled:
                    videos.clear_prompt_id(prompt_id)
                    logger.info("cancelled orphaned ComfyUI prompt %s", prompt_id)
                else:
                    unsettled.append(prompt_id)
            waiting = unsettled
        if waiting:
            logger.warning("%d orphaned ComfyUI prompt(s) not settled; retried at the next start",
                           len(waiting))

    async def _start_recovery() -> None:
        if orphans:
            recovery.append(asyncio.get_running_loop().create_task(_cancel_orphans()))

    async def _stop_recovery() -> None:
        for task in recovery:
            task.cancel()

    app.router.on_startup.append(_start_recovery)
    app.router.on_shutdown.append(_stop_recovery)

    @app.post("/v1/videos")
    async def create_video(request: Request):
        parsed = await _body(request, store, deps.settings.data_dir / "files", deps.settings)
        if isinstance(parsed, JSONResponse):
            return parsed
        body, reference, reference_filename = parsed

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error(400, "prompt must be a non-empty string", "invalid_value", "prompt")
        if len(prompt) > 32_000:
            return _error(400, "prompt must contain at most 32000 characters", "invalid_value", "prompt")

        model = body.get("model") or "sora-2"
        # isinstance first: `in` on a set hashes, and a list where the spec
        # says a string is an unhashable-type TypeError -- a 500 for the
        # caller's malformed body (2026-09-22, #5).
        if not isinstance(model, str) or model not in MODELS:
            return _error(400, f"model must be one of {sorted(MODELS)}", "invalid_value", "model")
        try:
            seconds = int(body.get("seconds") or 4)
        except (TypeError, ValueError):
            seconds = 0
        if seconds not in SECONDS:
            return _error(400, "seconds must be one of 4, 8, or 12", "invalid_value", "seconds")
        size = body.get("size") or "720x1280"
        if not isinstance(size, str) or size not in SIZES:      # same hash guard as model

            return _error(400, f"size must be one of {sorted(SIZES)}", "invalid_value", "size")
        width, height = SIZES[size]

        def _dimension_error(data: bytes) -> JSONResponse | None:
            dimensions = _reference_dimensions(data)
            if dimensions is None:
                return _error(400, "input_reference must be a readable image", "invalid_value", "input_reference")
            if dimensions != (width, height):
                return _error(400, f"input_reference must be {size}; received {dimensions[0]}x{dimensions[1]}",
                              "invalid_value", "input_reference")
            return None

        if reference is not None and not isinstance(reference, _RemoteReference):
            refused = _dimension_error(reference)
            if refused is not None:
                return refused

        def _queue_full() -> JSONResponse | None:
            # One budget with image edits and variations (review 2026-09-24 B10).
            if admission.full():
                return _error(429, f"too many videos are already queued; retry once fewer than {MAX_OUTSTANDING} are outstanding",
                              "rate_limit_exceeded", kind="rate_limit_error")
            return None

        backend = getattr(deps, "video_backend", None)
        if backend is None:
            return _error(503, "video generation is not configured", "backend_unavailable")
        kind = "r2v" if reference is not None else "t2v"
        if hasattr(backend, "supports") and not backend.supports(kind):
            return _error(503, f"{kind} video workflow is not configured", "video_workflow_unavailable")
        if not video_tools_available():
            return _error(503, "video requires ffmpeg and ffprobe on PATH", "video_tools_unavailable")
        if (full := _queue_full()) is not None:
            return full

        if isinstance(reference, _RemoteReference):
            # The one outbound fetch, only now that nothing cheap refuses the
            # request (review 2026-09-24 B22). Admission is checked again after
            # it: the fetch awaits, and the check-then-put below was atomic
            # only because nothing awaited between them.
            fetched = await _fetch_reference(str(reference))
            if isinstance(fetched, JSONResponse):
                return fetched
            reference, reference_filename = fetched
            refused = _dimension_error(reference)
            if refused is not None:
                return refused
            if (full := _queue_full()) is not None:
                return full

        video_id = f"video_{str(ULID()).lower()}"
        created = int(time.time())
        queued = {
            "id": video_id,
            "object": "video",
            "model": model,
            "status": "queued",
            "progress": 0,
            "created_at": created,
            "completed_at": None,
            "expires_at": None,
            "error": None,
            "prompt": prompt,
            "remixed_from_video_id": None,
            "seconds": str(seconds),
            "size": size,
        }
        videos.put(queued)

        def _began() -> None:
            # Fired by the backend once the one-render lock is held: THIS
            # is when the row may say in_progress. Stamping it at task start
            # made `queued` a status no caller could ever observe -- a job
            # waiting behind a 30-minute render claimed a GPU it did not have
            # (batch 4). And the tombstone reaches into the lock queue: a row
            # deleted while queued aborts here, before any submit, so no
            # render burns GPU time for nobody.
            if not videos.put_if_present({**queued, "status": "in_progress"}):
                raise _RowDeleted(video_id)

        async def _finish() -> None:
            # Every write from here is put_if_present: the row this task was
            # created for can be DELETEd mid-render, and the unconditional
            # INSERT OR REPLACE used here brought it back as `completed` --
            # the deleted thing returned (2026-09-22, #1).
            try:
                data, media_type = await backend.render(
                    "r2v" if reference is not None else "t2v",
                    {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "seconds": seconds,
                        "seed": int.from_bytes(os.urandom(8), "big") & ((1 << 63) - 1),
                        "reference": reference,
                        "reference_filename": reference_filename,
                    },
                    on_start=_began,
                    # Recorded so a restart can stop it (B20); forgotten once
                    # nothing of it can still run (Copilot on #336).
                    on_submit=lambda prompt_id: videos.set_prompt_id(video_id, prompt_id),
                    on_settled=videos.clear_prompt_id,
                )
                if not data or media_type != "video/mp4":
                    raise ValueError("video backend did not return an MP4")
                # DELETE awaits this render task. Drain a scrub already running
                # in a worker thread before DELETE reports completion, so its
                # ffmpeg process and temporary files cannot outlive the delete.
                scrub = asyncio.ensure_future(asyncio.to_thread(_scrub_mp4_metadata, data))
                try:
                    data = await asyncio.shield(scrub)
                except asyncio.CancelledError:
                    await asyncio.wait({scrub})
                    raise
                # Tens of megabytes; an inline write stalled every concurrent
                # stream on this loop for its whole duration (review 2026-09-22).
                # Bytes land before the status flips, as before.
                path = videos.root / f"{video_id}.mp4"
                write = asyncio.ensure_future(asyncio.to_thread(path.write_bytes, data))
                try:
                    await asyncio.shield(write)
                except asyncio.CancelledError:
                    # Cancelling this task does not stop the worker thread: a
                    # DELETE landing mid-write removed the row, unlinked a file
                    # that was not there yet, and then the thread wrote an
                    # orphan MP4 beside no row (review 2026-09-24 B22). Let the
                    # bytes land first -- DELETE awaits this task, so its unlink
                    # comes after them -- and clean up ourselves if the row is
                    # already gone by then.
                    await asyncio.wait({write})
                    if videos.get(video_id) is None:
                        path.unlink(missing_ok=True)
                    raise
            except _RowDeleted:
                return                       # deleted while queued for the GPU: no work, no record
            except asyncio.CancelledError:
                raise                        # deleted mid-render: row and file belong to the deleter
            except Exception as exc:  # backend internals never cross the public boundary
                logger.exception("video generation %s failed (%s)", video_id, type(exc).__name__)
                videos.put_if_present({**queued, "status": "failed",
                                       "expires_at": int(time.time()) + RETENTION_S,
                                       "error": {
                                           "code": "generation_failed",
                                           "message": "video generation failed",
                                           "misalignment": None,
                                       }})
                return
            if not videos.put_if_present({**queued, "status": "completed", "progress": 100,
                                          "completed_at": int(time.time()),
                                          "expires_at": int(time.time()) + RETENTION_S}):
                # Deleted between the render and the flip: keep the volume as
                # consistent as the tombstone -- no orphan MP4 beside no row.
                (videos.root / f"{video_id}.mp4").unlink(missing_ok=True)

        task = asyncio.get_running_loop().create_task(_finish())
        renders[video_id] = task

        def _discard(done: asyncio.Task, vid: str = video_id) -> None:
            renders.pop(vid, None)
            if not done.cancelled() and done.exception() is not None:
                logger.error("video task %s died: %r", vid, done.exception())

        task.add_done_callback(_discard)
        return JSONResponse(queued)

    @app.get("/v1/videos")
    async def list_videos(request: Request):
        q = request.query_params
        try:
            limit = int(q.get("limit") or 20)
        except ValueError:
            limit = 0
        if not 1 <= limit <= 100:
            return _error(400, "limit must be between 1 and 100", "invalid_value", "limit")
        order = q.get("order") or "desc"
        if order not in ("asc", "desc"):
            return _error(400, "order must be asc or desc", "invalid_value", "order")
        after = q.get("after")
        # An empty `after=` query string is `""` here, not None. Normalize so
        # the store sees no cursor and `after=` matches omitting the parameter
        # (Copilot review 2026-09-23, third round).
        if after is not None:
            after = after.strip() or None
        # SQL does the order, the cursor, and the slice (limit + 1 so
        # `has_more` is exact without a second COUNT query). The video
        # store keeps expiry on the row, so the per-list retention sweep
        # is one DELETE before the SELECT (review 2026-09-23).
        items, after_exists = videos.list_live(after_id=after, limit=limit + 1, order=order)
        if after and not after_exists:
            return _error(400, f"no video '{after}' in this list", "invalid_value", "after")
        page = items[:limit]
        return JSONResponse({
            "object": "list",
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": len(items) > limit,
        })

    @app.delete("/v1/videos/{video_id}")
    async def delete_video(video_id: str):
        # Cancel the render BEFORE removing the row: the tombstone writes stop
        # the resurrection, and the cancel stops the GPU work and frees the
        # admission slot (2026-09-22, #1). The cancel reaches into
        # the render loop: its CancelledError path interrupts the ComfyUI job
        # before releasing the one-render lock, so the global interrupt can
        # only hit this prompt -- a deleted job does not keep burning the GPU
        # with the next one queued behind the orphan (the required fix). The
        # shield pattern is the cancel route's: the task's own cancellation is
        # expected; ours (a client leaving mid-delete) must propagate.
        task = renders.get(video_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    # We were cancelled while the render drained. The render is
                    # already cancelled, so nothing else will ever remove this
                    # row: it would sit in_progress forever, pinning an
                    # admission slot. Finish the delete when the drain ends
                    # (Copilot on #334, review 2026-09-24 B22).
                    task.add_done_callback(lambda _done, vid=video_id: videos.delete(vid))
                    raise
            except Exception:  # noqa: BLE001 - _finish already recorded the failure
                pass
        if not videos.delete(video_id):
            return _error(404, f"No such video: {video_id}", "not_found", "video_id")
        return JSONResponse({"id": video_id, "object": "video.deleted", "deleted": True})

    @app.get("/v1/videos/{video_id}")
    async def retrieve_video(video_id: str):
        record = videos.get(video_id)
        if record is None:
            return _error(404, f"No such video: {video_id}", "not_found", "video_id")
        return JSONResponse(record)

    # Deliberately sync, unlike every other route here: both image variants shell
    # out to ffmpeg (the sheet also to ffprobe) under a bounded ceiling, and
    # `subprocess.run` inside a coroutine stalls the whole event loop for that
    # long. FastAPI runs a sync path operation in a worker thread, so one awkward
    # MP4 costs one thread instead of the process. Do not make this `async def`
    # without moving ffmpeg off the loop first.
    @app.get("/v1/videos/{video_id}/content")
    def video_content(video_id: str, variant: str = "video"):
        path = videos.file(video_id)
        if path is None:
            return _error(404, f"No video content for: {video_id}", "not_found", "video_id")
        if variant in ("thumbnail", "spritesheet"):
            # One envelope for both derived images: the thing asked for could
            # not be made from this video. Named by variant so a caller can
            # tell which derivation failed, and a 502 rather than a 500
            # because the request was well-formed and the source was the
            # problem (missing ffmpeg, unreadable container, failed encode).
            png = _first_frame_png(path) if variant == "thumbnail" else _spritesheet_png(path)
            if png is None:
                return JSONResponse(
                    {"error": {
                        "message": f"a {variant} could not be made from this video",
                        "type": "server_error",
                        "param": "variant",
                        "code": f"{variant}_unavailable",
                    }},
                    status_code=502,
                )
            return Response(png, media_type="image/png", headers={"content-disposition": f'inline; filename="{video_id}.png"'})
        if variant != "video":
            return _error(400, "variant must be video, thumbnail, or spritesheet", "unsupported_value", "variant")
        return FileResponse(path, media_type="video/mp4", filename=f"{video_id}.mp4")
