"""The Files API (#142: "files, uploads, batches, vector stores"): upload, list,
retrieve, delete and download, in the pinned spec's shapes.

Files are storage only; nothing reads them yet. They are the ground Batches and
Responses `input_file` stand on. The bytes live on the service's data volume
(`data_dir/files/<id>`), the File object in the same SQLite file as stored
responses.

Retention follows the spec text: `purpose=batch` expires after 30 days, every
other purpose is kept until deleted, and `expires_after` sets 1 hour to 30 days
from `created_at`. `evals` is accepted by CreateFileRequest but cannot be
represented in the File object's purpose enum, and evals are out of scope
(#142), so it is refused by name.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from ulid import ULID

from .responses_store import ResponseStore

MAX_BYTES = 512 * 1024 * 1024          # OpenAI's per-file limit
BODY_SLACK = 1024 * 1024               # multipart boundaries and the other fields around the file
PURPOSES = ("assistants", "batch", "fine-tune", "vision", "user_data")
BATCH_EXPIRY_S = 30 * 24 * 3600
FIELDS = {"file", "purpose", "expires_after[anchor]", "expires_after[seconds]"}


class StoredFileTooLarge(ValueError):
    """The stored file exists but is bigger than this consumer may load."""

    def __init__(self, size: int, max_bytes: int) -> None:
        super().__init__(f"stored file is {size} bytes, over the {max_bytes}-byte cap")
        self.size, self.max_bytes = size, max_bytes


# `_body_within_cap` holds the whole upload (up to MAX_BYTES + slack) resident
# while the form parses, and nothing else bounds how many uploads do that at
# once: N concurrent callers meant N x 513 MB in one process (review
# 2026-09-22). Two slots keep a legitimate second upload waiting-free in
# practice while capping aggregate memory at roughly 1 GB; a busy service
# answers 429 instead of trading latency for an OOM kill that takes every
# in-flight stream with it.
_MAX_CONCURRENT_UPLOADS = 2
_upload_slots = asyncio.Semaphore(_MAX_CONCURRENT_UPLOADS)


def load_stored_file(store, files_dir: Path, file_id: str, max_bytes: int | None = None) -> tuple[dict, bytes] | None:
    """The File object and its bytes, or None when the id is unknown or expired.

    Expired rows are deleted here, the same way a retrieve does, and their bytes
    are removed so a later reader cannot find an orphan file.

    `max_bytes` is the CONSUMER's cap, not the store's: a stored file may be up
    to MAX_BYTES (512 MB) while an image-edit source may not exceed 25 MB.
    Over the cap raises StoredFileTooLarge -- checked against the file's size
    BEFORE reading, with a bounded read as the backstop, because reading first
    and measuring after only enforces the cap once the bytes are already
    resident (review 2026-09-22)."""
    files, gone, _ = store.live_files()
    for gone_id in gone:
        (files_dir / gone_id).unlink(missing_ok=True)
    file = next((item for item in files if item["id"] == file_id), None)
    path = files_dir / file_id
    if file is None or not path.is_file():
        return None
    try:
        if max_bytes is not None:
            size = path.stat().st_size
            if size > max_bytes:
                raise StoredFileTooLarge(size, max_bytes)
            with path.open("rb") as fh:
                data = fh.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise StoredFileTooLarge(len(data), max_bytes)
            return file, data
        return file, path.read_bytes()
    except FileNotFoundError:
        # Deleted between the is_file() check and the read (a concurrent
        # DELETE or expiry sweep). Unknown now, same as a moment ago: 404,
        # never an unhandled 500.
        return None


async def _body_within_cap(request: Request) -> bytes | JSONResponse:
    """Read the body only up to the file cap, then let form() parse that copy.

    `request.form()` otherwise reads the whole upload into a spool before the
    per-file check can refuse it."""
    limit = MAX_BYTES + BODY_SLACK
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_big = int(declared) > limit
        except ValueError:
            return _error(400, "Content-Length is not an integer", "invalid_value", "file")
        if too_big:
            return _error(400, f"file is larger than {MAX_BYTES} bytes", "file_too_large", "file")
    buf = bytearray()
    async for chunk in request.stream():
        if len(buf) + len(chunk) > limit:
            return _error(400, f"file is larger than {MAX_BYTES} bytes", "file_too_large", "file")
        buf.extend(chunk)
    request._body = bytes(buf)
    return request._body


def _error(status: int, message: str, code: str | None, param: str | None = None,
           kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "param": param, "code": code}}, status_code=status)


def _not_found(file_id: str) -> JSONResponse:
    return _error(404, f"No such File object: {file_id}", None, "file_id")


def register(app: FastAPI, deps, store: ResponseStore) -> None:
    files_dir: Path = deps.settings.data_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    def live() -> list[dict]:
        """All files (post-retention sweep) for callers that scan the table:
        `find()` and any consumer that needs every row, not a page."""
        files, gone, _ = store.live_files()
        for file_id in gone:
            (files_dir / file_id).unlink(missing_ok=True)
        return files

    def find(file_id: str) -> dict | None:
        return next((f for f in live() if f["id"] == file_id), None)

    @app.post("/v1/files")
    async def create_file(request: Request):
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            return _error(400, "the request must be multipart/form-data", "invalid_request")
        # The capped body stays resident in process memory (request._body) for
        # the handler's whole life, so the slots bound AGGREGATE memory, not
        # one request's. The local _error helper emits the API's 429 vocabulary (review
        # 2026-09-22: N concurrent uploads were N x 513 MB with no bound).
        if _upload_slots.locked():
            return _error(429, f"more than {_MAX_CONCURRENT_UPLOADS} file uploads are already in flight; retry shortly",
                          "rate_limit_exceeded", None, kind="rate_limit_error")
        async with _upload_slots:
            return await _store_upload(request)

    async def _store_upload(request: Request):
        limited = await _body_within_cap(request)
        if isinstance(limited, JSONResponse):
            return limited
        try:
            form = await request.form()
        except Exception:  # noqa: BLE001 - a malformed body is the caller's
            return _error(400, "the multipart body could not be parsed", "invalid_request")
        unknown = sorted(set(form.keys()) - FIELDS)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        purpose = form.get("purpose")
        if purpose == "evals":
            return _error(400, "purpose evals is not supported", "unsupported_value", "purpose")
        if purpose not in PURPOSES:
            return _error(400, f"purpose must be one of {list(PURPOSES)}", "invalid_value", "purpose")
        anchor, seconds = form.get("expires_after[anchor]"), form.get("expires_after[seconds]")
        if (anchor is None) != (seconds is None):
            missing = "expires_after[seconds]" if seconds is None else "expires_after[anchor]"
            return _error(400, f"{missing} is required with expires_after", "invalid_value", missing)
        # The validated window, or None when no expiry was asked for. One
        # variable, set once: the raw form value keeps its UploadFile|str|None
        # flow type past this block, and a second `anchor is not None` check
        # downstream cannot re-narrow it (pyright batch C).
        expires_in: int | None = None
        if anchor is not None:
            if anchor != "created_at":
                return _error(400, "expires_after[anchor] must be created_at", "invalid_value", "expires_after[anchor]")
            # A misfiled file part lands where the old int() raised TypeError:
            # it becomes 0 and the range check answers, exactly as before.
            try:
                expires_in = int(seconds) if isinstance(seconds, str) else 0
            except ValueError:
                expires_in = 0
            if not 3600 <= expires_in <= 2592000:
                return _error(400, "expires_after[seconds] must be between 3600 and 2592000", "invalid_value",
                              "expires_after[seconds]")
        upload = form.get("file")
        if upload is None or isinstance(upload, str) or len(form.getlist("file")) != 1:
            return _error(400, "file is required, once, as a file part", "invalid_value", "file")

        file_id = f"file-{str(ULID()).lower()}"
        fd, tmp = tempfile.mkstemp(dir=files_dir, prefix=".upload-")
        size = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        return _error(400, f"file is larger than {MAX_BYTES} bytes", "file_too_large", "file")
                    out.write(chunk)
            if size == 0:
                return _error(400, "file is empty", "invalid_value", "file")
            os.replace(tmp, files_dir / file_id)
        except OSError:
            # A full or failing disk (#209: an ENOSPC mid-write escaped as a raw 500 and
            # left the temp file behind). The caller gets the envelope; nothing is kept.
            (files_dir / file_id).unlink(missing_ok=True)
            return _error(500, "the file could not be stored", "storage_failed", None, kind="server_error")
        finally:
            # Whatever happened, the temp file never outlives the request. After a successful
            # os.replace it no longer exists under this name, so this is a no-op.
            Path(tmp).unlink(missing_ok=True)

        created = int(time.time())
        file = {"id": file_id, "object": "file", "bytes": size, "created_at": created,
                "filename": os.path.basename(upload.filename or "") or "file", "purpose": purpose, "status": "processed"}
        if expires_in is not None:
            file["expires_at"] = created + expires_in
        elif purpose == "batch":
            file["expires_at"] = created + BATCH_EXPIRY_S
        store.put_file(file)
        return JSONResponse(file)

    @app.get("/v1/files")
    async def list_files(request: Request):
        q = request.query_params
        try:
            limit = int(q.get("limit") or 10000)
        except ValueError:
            limit = 0
        if not 1 <= limit <= 10000:
            return _error(400, "limit must be between 1 and 10000", "invalid_value", "limit")
        order = q.get("order") or "desc"
        if order not in ("asc", "desc"):
            return _error(400, "order must be asc or desc", "invalid_value", "order")
        purpose = q.get("purpose")
        after = q.get("after")
        # An empty `after=` query string is `""` here, not None. Normalize so
        # the store sees no cursor and `after=` matches omitting the parameter
        # (Copilot review 2026-09-23, third round).
        if after is not None:
            after = after.strip() or None
        # SQL does the order, the cursor, the purpose filter, and the slice
        # (limit + 1 so `has_more` is exact without a second COUNT query).
        # The store 400s on a cursor that is not in the filtered set, so a
        # cursor from a different purpose is rejected (review 2026-09-23,
        # Copilot medium-severity fix).
        items, gone, ok = store.live_files(after_id=after, limit=limit + 1, order=order, purpose=purpose)
        # Unlink the bytes the sweep removed BEFORE the bogus-cursor 400,
        # so an invalid-cursor request does not leave orphan files on disk
        # that no later list can discover to clean up (Copilot review
        # 2026-09-23, fourth round). The order matches videos.list_live,
        # which does the same.
        for file_id in gone:
            (files_dir / file_id).unlink(missing_ok=True)
        if after and not ok:
            return _error(400, f"no file '{after}' in this list", "invalid_value", "after")
        page = items[:limit]
        return JSONResponse({"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                             "last_id": page[-1]["id"] if page else None, "has_more": len(items) > limit})

    @app.get("/v1/files/{file_id}")
    async def retrieve_file(file_id: str):
        file = find(file_id)
        return JSONResponse(file) if file else _not_found(file_id)

    @app.delete("/v1/files/{file_id}")
    async def delete_file(file_id: str):
        if find(file_id) is None or not store.delete_file(file_id):
            return _not_found(file_id)
        (files_dir / file_id).unlink(missing_ok=True)
        return JSONResponse({"id": file_id, "object": "file", "deleted": True})

    @app.get("/v1/files/{file_id}/content")
    async def file_content(file_id: str):
        file = find(file_id)
        if file is None or not (files_dir / file_id).is_file():
            return _not_found(file_id)
        return FileResponse(files_dir / file_id, media_type="application/octet-stream", filename=file["filename"])
