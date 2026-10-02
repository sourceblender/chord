"""Run an operator-owned ComfyUI video workflow behind the public video API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Callable
from typing import Any

from . import comfy
from .comfy_video_workflow import VideoWorkflow

logger = logging.getLogger(__name__)

T2V = "t2v"
R2V = "r2v"


class VideoError(RuntimeError):
    """The request cannot be rendered, and the reason is the caller's to hear."""


async def render(client: comfy.ComfyClient, workflow: VideoWorkflow, params: dict[str, Any],
                 poll_interval: float = 2.0, timeout: float = 1800.0,
                 on_submit: Callable[[str], None] | None = None,
                 on_settled: Callable[[str], None] | None = None) -> tuple[bytes, str]:
    """Render `params` and return the finished artifact.

    `on_submit(id)` fires once ComfyUI accepts the prompt; `on_settled(id)`
    once nothing of it can still be running there -- ComfyUI finished it, or a
    stop of it settled. A failed stop never reports settled, so the caller's
    record of the prompt survives for a restart to retry (Copilot on #336).

    Blocking by design for the first draft — ComfyUI is already asynchronous
    underneath (submit, poll, retrieve), so the eventual async contract exposes
    these same three steps rather than replacing them.
    """
    import asyncio

    reference = params.get("reference")
    reference_name = None
    if reference:
        reference_name = await client.upload_image(
            reference, params.get("reference_filename") or "reference.png")

    graph = workflow.for_request(
        prompt=params["prompt"],
        width=int(params["width"]),
        height=int(params["height"]),
        seconds=int(params["seconds"]),
        seed=int(params.get("seed", random.getrandbits(63))),
        reference_name=reference_name,
    )
    # Shielded: a cancel landing after ComfyUI accepted /prompt but before its
    # answer arrived used to leave with no id -- nothing stopped the prompt and
    # nothing recorded it (Copilot on #336). The id is waited for, recorded,
    # stopped, and the cancel goes on.
    def settled(pid: str) -> None:
        if on_settled is not None:
            try:
                on_settled(pid)
            except Exception:
                logger.exception("recording prompt %s as settled failed", pid)

    async def stop(pid: str) -> None:
        if await asyncio.shield(client.interrupt(pid)) is True:
            settled(pid)

    submitting = asyncio.ensure_future(client.submit(graph))
    try:
        prompt_id = await asyncio.shield(submitting)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            accepted = await asyncio.shield(submitting)
            try:
                if on_submit is not None:
                    on_submit(accepted)
            finally:
                await stop(accepted)     # whatever the record did (Copilot on #336)
        raise

    started = asyncio.get_running_loop().time()
    try:
        if on_submit is not None:
            # The caller keeps the id so a restart can cancel the orphan
            # (review 2026-09-24 B20). ComfyUI already accepted the prompt, so
            # a callback that fails (a disk error keeping the id) must stop it
            # before the error propagates, or it renders on as an orphan
            # nothing recorded (Copilot on #336).
            try:
                on_submit(prompt_id)
            except Exception:
                await stop(prompt_id)        # a settled stop clears any record the callback made
                raise
        while True:
            status, _progress = await client.poll(prompt_id)
            if status in (comfy.COMPLETED, comfy.FAILED):
                settled(prompt_id)           # ComfyUI is done with it either way
            if status == comfy.COMPLETED:
                return await client.fetch(prompt_id, workflow.output_node_id)
            if status == comfy.FAILED:
                # Let fetch raise: it reads the execution error out of history, so
                # the caller gets the failing node instead of a bare "failed".
                return await client.fetch(prompt_id, workflow.output_node_id)
            elapsed = asyncio.get_running_loop().time() - started
            if elapsed >= timeout:
                await stop(prompt_id)
                raise VideoError(f"render timed out after {timeout:.0f}s (prompt {prompt_id})")
            await asyncio.sleep(min(poll_interval, timeout - elapsed))
    except asyncio.CancelledError:
        # A cancel (DELETE) must stop the WORK, not only chord's wait. The
        # interrupt is sent only while ComfyUI's queue shows THIS prompt
        # running, and names it, so ComfyUI re-checks before stopping; the
        # one-render lock alone could not promise that on a ComfyUI shared
        # with other producers (review 2026-09-24 B20). ComfyUI's own check-then-
        # stop is not atomic, so this narrows the race, it does not close it
        # (ComfyUI cancellation). This runs before VideoBackend's finally releases the
        # lock, so a deleted job cannot keep burning the GPU with the next
        # job submitted behind the orphan. The
        # shield keeps a second cancellation from cancelling the interrupt.
        await stop(prompt_id)
        raise


class VideoBackend:
    """What the API route holds: one call, no ComfyUI vocabulary.

    `videos.py` never imports `comfy`.  It constructs this (or is handed a
    double) and calls `render`.  Everything ComfyUI-shaped — uploads, graphs,
    node ids, polling — stops here.
    """

    def __init__(self, client: comfy.ComfyClient, workflows: dict[str, VideoWorkflow],
                 timeout: float = 1800.0, render_lock: asyncio.Lock | None = None) -> None:
        self._client = client
        self._workflows = workflows
        self._timeout = timeout
        self._one_render: asyncio.Lock | None = render_lock

    @classmethod
    def from_settings(cls, settings: Any,
                      render_lock: asyncio.Lock | None = None) -> "VideoBackend | None":
        """Build from configuration, or None when this deployment has no ComfyUI.

        None rather than a broken instance: a route that is advertised and then
        fails is worse than one that was never advertised, which is the rule
        embeddings already follows for its own backend.
        """
        base_url = getattr(settings, "comfy_base_url", "")
        configured = getattr(settings, "video_workflows", None)
        if not base_url.strip() or configured is None:
            return None
        timeout = float(getattr(settings, "video_timeout_s", 1800.0))
        return cls(comfy.ComfyClient(base_url, timeout=timeout),
                   {kind: workflow.load() for kind, workflow in configured.items()},
                   timeout=timeout,
                   render_lock=render_lock)

    async def _lock(self) -> asyncio.Lock:
        if self._one_render is None:
            self._one_render = asyncio.Lock()
        return self._one_render

    def supports(self, kind: str) -> bool:
        return kind in self._workflows

    async def _acquire(self) -> asyncio.Lock:
        """The GPU lock, with the WAIT bounded by one render's timeout.

        The lock models the single GPU; waiting on it used to be unbounded. An
        image edit is an inline HTTP request, so a deep video queue held the
        caller's connection open for hours, and a video job queued behind it
        pinned its reference bytes (up to 25 MB) in its task closure the whole
        time (review 2026-09-22). Past the bound the job fails through the
        ordinary envelope instead. Cancel-safety was verified against the
        installed 3.12.13 Lock.acquire: every raising branch leaves the lock
        unheld, and a wakeup granted at the deadline is handed to the next
        waiter, so a timeout cannot leak the lock."""
        lock = await self._lock()
        await asyncio.wait_for(lock.acquire(), timeout=self._timeout)
        return lock

    async def render(self, kind: str, params: dict[str, Any],
                     on_start: Callable[[], None] | None = None,
                     on_submit: Callable[[str], None] | None = None,
                     on_settled: Callable[[str], None] | None = None) -> tuple[bytes, str]:
        # One render at a time: the GPU is single. (The interrupt no longer
        # leans on this lock to be safe; it checks whose prompt is running,
        # review 2026-09-24 B20.)
        workflow = self._workflows.get(kind)
        if workflow is None:
            raise VideoError(f"video workflow {kind!r} is not configured")
        lock = await self._acquire()
        try:
            if on_start is not None:
                # The GPU is actually ours now, and only now: a job WAITING on
                # this lock is queued, and the spec's first status describes
                # the render, not the intention (batch 4). An exception from
                # the callback (a deleted row) unwinds through the finally, so
                # the lock never leaks.
                on_start()
            return await render(self._client, workflow, params, timeout=self._timeout,
                                on_submit=on_submit, on_settled=on_settled)
        finally:
            lock.release()

    async def cancel_orphan(self, prompt_id: str) -> bool:
        """Stop a prompt a previous process submitted and never finished
        (review 2026-09-24 B20), and say whether the stop settled. No lock:
        the orphan is not this process's render, and the interrupt only ever
        touches `prompt_id`."""
        return await self._client.interrupt(prompt_id)

    async def aclose(self) -> None:
        await self._client.aclose()
