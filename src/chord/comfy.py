"""A ComfyUI client that knows nothing about what it is rendering.

ComfyUI is a graph executor with four verbs: put an image where the graph can
see it, submit a graph, ask whether that graph finished, and collect what it
produced.  Nothing in this module names video, audio, a model or a prompt —
those belong to whoever builds the graph.  The next graph family we add (music
is already on the list) reuses this file untouched and writes only its own
filler.

The one judgement call here is `poll`.  ComfyUI reports fine-grained progress
over a websocket and nothing over HTTP, so a polled caller can honestly know
three things: the prompt is still queued, it is executing, or it is done.  This
returns exactly that.  It does not interpolate a percentage between them,
because a number that moves while nothing is known is worse than a coarse one
that is true.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx

# What ComfyUI calls an artifact depends on the node that saved it — SaveImage
# writes `images`, SaveVideo writes `videos`, and a custom node may invent its
# own key.  Rather than enumerate them, we look for the SHAPE they all share: a
# list of records carrying a filename.  A new saver node works on arrival.
_FILENAME = "filename"
logger = logging.getLogger(__name__)
_POLL_TIMEOUT = 30.0

CONTENT_TYPES = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


class ComfyError(RuntimeError):
    """A ComfyUI call did not do what was asked.

    Carries the status when the failure came back over HTTP so a caller can
    tell "the box refused this graph" from "the box never answered".
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ComfyClient:
    """One ComfyUI instance.

    `client_id` identifies this caller to ComfyUI so submissions are
    attributable in its queue; it is stable for the life of the client rather
    than per-request, which is what the server expects.
    """

    def __init__(self, base_url: str, timeout: float = 900.0,
                 client: httpx.AsyncClient | None = None) -> None:
        if not base_url.strip():
            raise ValueError("comfy base_url is required")
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def upload_image(self, data: bytes, filename: str) -> str:
        """Put `data` where a LoadImage node can reach it; return the name to use.

        The returned name is ComfyUI's, not the one we sent: it renames on
        collision, and a graph that referenced our name would then load someone
        else's picture.  Always inject what comes back.
        """
        if not data:
            raise ValueError("upload_image requires bytes")
        files = {"image": (filename, data)}
        # overwrite=false so two concurrent renders cannot clobber each other's
        # reference; ComfyUI answers with the deduplicated name it actually used.
        response = await self._client.post("/upload/image", files=files,
                                           data={"overwrite": "false"})
        self._raise_for_status(response, "upload_image")
        body = self._json(response, "upload_image")
        name = body.get("name")
        if not name:
            raise ComfyError(f"upload_image: no name in response {body!r}")
        subfolder = body.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else str(name)

    async def submit(self, graph: dict[str, Any]) -> str:
        """Queue `graph` for execution; return the prompt id that tracks it."""
        if not graph:
            raise ValueError("submit requires a graph")
        response = await self._client.post(
            "/prompt", json={"prompt": graph, "client_id": self.client_id})
        self._raise_for_status(response, "submit")
        body = self._json(response, "submit")
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            # ComfyUI reports graph validation failures in the 200 body as often
            # as by status, so an absent prompt_id is a rejection, not a shrug.
            raise ComfyError(f"submit: rejected, no prompt_id in {json.dumps(body)[:400]}")
        return str(prompt_id)

    async def poll(self, prompt_id: str) -> tuple[str, float]:
        """(status, progress) for `prompt_id`.

        Progress is coarse on purpose — see the module docstring.  A prompt that
        is not yet in history is either waiting or running, and the queue tells
        us which.
        """
        history = await self._history(prompt_id, timeout=_POLL_TIMEOUT)
        if history is not None:
            status = history.get("status") or {}
            if status.get("completed") is True and status.get("status_str") == "success":
                return COMPLETED, 1.0
            if status.get("completed") is True or status.get("status_str") == "error":
                return FAILED, 1.0
            return RUNNING, 0.5
        return (RUNNING, 0.5) if await self._is_running(prompt_id) else (QUEUED, 0.0)

    async def fetch(self, prompt_id: str, output_node_id: str | None = None) -> tuple[bytes, str]:
        """The artifact `prompt_id` produced, with its content type.

        Raises if the prompt failed, is unfinished, or saved nothing — each of
        those is a different bug and an empty body would hide all three.
        """
        history = await self._history(prompt_id)
        if history is None:
            raise ComfyError(f"fetch: {prompt_id} has no history; it is unfinished or unknown")
        status = history.get("status") or {}
        # Poll treats any completed-but-not-success history as FAILED and then
        # asks fetch to raise. Raising only on status_str == "error" let an
        # interrupted render (completed, some other status) return a partial file.
        succeeded = status.get("completed") is True and status.get("status_str") == "success"
        if not succeeded:
            if status.get("completed") is True or status.get("status_str") == "error":
                raise ComfyError(f"fetch: {prompt_id} failed: {self._failure(status)}")
            raise ComfyError(f"fetch: {prompt_id} is not complete")

        outputs = history.get("outputs") or {}
        if output_node_id is not None:
            if not isinstance(outputs, dict) or output_node_id not in outputs:
                raise ComfyError(f"fetch: output node {output_node_id!r} saved nothing")
            outputs = {output_node_id: outputs[output_node_id]}
        artifact = self._artifact(outputs)
        if artifact is None:
            raise ComfyError(f"fetch: {prompt_id} completed but saved no artifact")

        params = {
            "filename": artifact[_FILENAME],
            "subfolder": artifact.get("subfolder", ""),
            "type": artifact.get("type", "output"),
        }
        response = await self._client.get("/view", params=params)
        self._raise_for_status(response, "fetch")
        return response.content, self._content_type(response, str(artifact[_FILENAME]))

    async def fetch_pngs(self, prompt_id: str, output_node_id: str | None = None) -> list[bytes]:
        """Saved PNGs from a finished prompt, optionally from one bound node."""
        history = await self._history(prompt_id)
        if history is None:
            raise ComfyError(f"fetch: {prompt_id} has no history; it is unfinished or unknown")
        status = history.get("status") or {}
        succeeded = status.get("completed") is True and status.get("status_str") == "success"
        if not succeeded:
            if status.get("completed") is True or status.get("status_str") == "error":
                raise ComfyError(f"fetch: {prompt_id} failed: {self._failure(status)}")
            raise ComfyError(f"fetch: {prompt_id} is not complete")
        outputs = history.get("outputs") or {}
        if output_node_id is not None:
            if not isinstance(outputs, dict) or output_node_id not in outputs:
                raise ComfyError(f"fetch: output node {output_node_id!r} saved nothing")
            outputs = {output_node_id: outputs[output_node_id]}
        records = [
            record for record in self._records(outputs)
            if str(record.get(_FILENAME, "")).lower().endswith(".png")
        ]
        if not records:
            raise ComfyError("fetch: finished prompt saved no PNG")
        records.sort(key=lambda record: str(record[_FILENAME]))
        pngs: list[bytes] = []
        for record in records:
            params = {
                "filename": record[_FILENAME],
                "subfolder": record.get("subfolder", ""),
                "type": record.get("type", "output"),
            }
            response = await self._client.get("/view", params=params)
            self._raise_for_status(response, "fetch")
            pngs.append(response.content)
        return pngs

    async def interrupt(self, prompt_id: str) -> bool:
        """Stop `prompt_id`, and nothing that is not `prompt_id`.

        ComfyUI's POST /interrupt stops whatever is executing, and the ComfyUI
        Chord renders on may be shared with other producers, so a blind
        interrupt could kill someone else's job (review 2026-09-24 B20). So the
        queue is read first: our prompt still pending is removed by id with
        POST /queue {"delete": [...]}; our prompt executing is interrupted, and
        the interrupt names it, so ComfyUI re-checks the running prompt before
        stopping. That NARROWS the race, it does not close it: ComfyUI's
        handler (0.33.1 and 0.36.0, read at the tags) checks the id and then
        sets the global interrupt flag, not atomically, so a job starting in
        that gap is stopped. Queued deletion by id is exact; running-job
        cancellation stays uncertified. A queue that
        cannot be read means "unknown", and unknown is never ours: only the
        by-id delete, which cannot touch another prompt, is sent. A failure
        here does not hide the timeout, cancel or restart that asked for it.

        Returns whether the stop is SETTLED: the running prompt's interrupt was
        accepted, or a read shows the prompt gone. A startup
        cancel keeps its retry marker until this is true (Copilot on #336).
        """
        # A delete of a pending prompt is re-checked: ComfyUI can promote it to
        # running between the read and the delete, which makes the delete a
        # no-op, so only a later read can say it is gone (Copilot on #336).
        for _ in range(3):
            try:
                running, pending = await self._queue_state(prompt_id)
            except Exception:
                logger.exception("comfy queue read before stopping %s failed; not interrupting", prompt_id)
                await self._stop(prompt_id, "queue delete")      # by id: cannot touch another prompt
                return False
            if not running and not pending:
                return True
            if running:
                return await self._stop(prompt_id, "interrupt")
            if not await self._stop(prompt_id, "queue delete"):
                return False
        return False

    async def _stop(self, prompt_id: str, call: str) -> bool:
        try:
            if call == "interrupt":
                response = await self._client.post("/interrupt", json={"prompt_id": prompt_id}, timeout=10)
            else:
                response = await self._client.post("/queue", json={"delete": [prompt_id]}, timeout=10)
            self._raise_for_status(response, call)
            return True
        except Exception:
            logger.exception("comfy %s for %s failed", call, prompt_id)
            return False

    async def _queue_state(self, prompt_id: str) -> tuple[bool, bool]:
        """(running, pending) for `prompt_id`, from one GET /queue."""
        response = await self._client.get("/queue", timeout=_POLL_TIMEOUT)
        self._raise_for_status(response, "queue")
        body = self._json(response, "queue")
        # Absence is proof the prompt is gone only from a queue that was really
        # listed: a 200 whose fields are not lists used to read as "not
        # there", and a stop reported settled for a prompt still rendering
        # (Copilot on #336). A missing queue_pending reads as empty (older
        # answers omit it); queue_running must be present.
        running, pending = body.get("queue_running"), body.get("queue_pending", [])
        if not isinstance(running, list) or not isinstance(pending, list):
            raise ComfyError("comfy queue answer is not a listing")
        return self._queue_has(running, prompt_id), self._queue_has(pending, prompt_id)

    @staticmethod
    def _queue_has(entries: list, prompt_id: str) -> bool:
        # Queue entries are positional arrays; the prompt id is one of the
        # leading scalars and its index has moved between ComfyUI versions,
        # so match on the value rather than on a position.
        return any(isinstance(item, list) and prompt_id in [f for f in item if isinstance(f, str)]
                   for item in entries)

    async def _history(self, prompt_id: str, timeout: float = _POLL_TIMEOUT) -> dict[str, Any] | None:
        # httpx reads an explicit timeout=None as NO TIMEOUT AT ALL, not "the
        # client default". A stalled history call -- made while the caller
        # holds the one-render lock -- therefore wedged every later edit,
        # variation and video behind a lock that never released, until process
        # restart (2026-09-22, #4). The poll loop already passed
        # _POLL_TIMEOUT explicitly; fetch/fetch_pngs did not, and the default
        # None disabled the 900 s the client was constructed with.
        response = await self._client.get(f"/history/{prompt_id}", timeout=timeout)
        self._raise_for_status(response, "history")
        body = self._json(response, "history")
        entry = body.get(prompt_id)
        return entry if isinstance(entry, dict) else None

    async def _is_running(self, prompt_id: str) -> bool:
        running, _pending = await self._queue_state(prompt_id)
        return running

    @staticmethod
    def _records(outputs: dict[str, Any]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for node in outputs.values():
            if not isinstance(node, dict):
                continue
            for records in node.values():
                if not isinstance(records, list):
                    continue
                for record in records:
                    if isinstance(record, dict) and record.get(_FILENAME):
                        found.append(record)
        return found

    @staticmethod
    def _artifact(outputs: dict[str, Any]) -> dict[str, Any] | None:
        found = ComfyClient._records(outputs)
        for record in found:
            name = str(record[_FILENAME])
            if name.lower().endswith(".mp4"):
                return record
        return found[0] if found else None

    @staticmethod
    def _content_type(response: httpx.Response, filename: str) -> str:
        header = response.headers.get("content-type", "")
        # ComfyUI serves /view as application/octet-stream for video, so the
        # extension is the better witness; the header wins only when it is
        # specific enough to be worth anything.
        if header and not header.startswith("application/octet-stream"):
            return header.split(";")[0].strip()
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return CONTENT_TYPES.get(suffix, "application/octet-stream")

    @staticmethod
    def _failure(status: dict[str, Any]) -> str:
        for name, payload in status.get("messages") or []:
            if name == "execution_error" and isinstance(payload, dict):
                node = payload.get("node_type") or payload.get("node_id")
                return f"{node}: {payload.get('exception_message') or payload.get('exception_type')}"
        return status.get("status_str") or "unknown"

    @staticmethod
    def _raise_for_status(response: httpx.Response, call: str) -> None:
        if response.status_code >= 400:
            raise ComfyError(f"{call}: {response.status_code} {response.text[:300]}",
                             response.status_code)

    @staticmethod
    def _json(response: httpx.Response, call: str) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyError(f"{call}: response was not JSON: {response.text[:200]}") from exc
        if not isinstance(body, dict):
            raise ComfyError(f"{call}: expected a JSON object, got {type(body).__name__}")
        return body
