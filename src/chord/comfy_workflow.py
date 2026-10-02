"""Operator-owned ComfyUI image workflows and their request bindings.

The workflow chooses its model, checkpoint, samplers, and output nodes. Chord
only fills the named inputs; no workflow or model is embedded in the product.
Generation fills a prompt. Edits and variations also fill a source image,
uploaded to ComfyUI first.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import comfy


logger = logging.getLogger(__name__)


class WorkflowError(ValueError):
    """The configured API-format workflow cannot honor an image request."""


@dataclass(frozen=True)
class ComfyImageWorkflow:
    graph: dict[str, dict[str, Any]]
    prompt_node_id: str
    output_node_id: str
    prompt_input_name: str = "text"
    seed_node_id: str | None = None
    seed_input_name: str = "seed"

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        prompt_node_id: str,
        output_node_id: str,
        prompt_input_name: str = "text",
        seed_node_id: str | None = None,
        seed_input_name: str = "seed",
    ) -> ComfyImageWorkflow:
        raw = cls._graph(path)
        workflow = cls(raw, prompt_node_id, output_node_id, prompt_input_name,
                       seed_node_id, seed_input_name)
        workflow._input(prompt_node_id, prompt_input_name)
        if output_node_id not in raw:
            raise WorkflowError(f"image workflow has no output node {output_node_id!r}")
        if seed_node_id is not None:
            workflow._input(seed_node_id, seed_input_name)
        return workflow

    @staticmethod
    def _graph(path: Path) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"cannot load image workflow {path}: {type(exc).__name__}") from None
        if not isinstance(raw, dict) or not raw:
            raise WorkflowError("image workflow must be a nonempty API-format node map")
        if any(
            not isinstance(node_id, str)
            or not isinstance(node, dict)
            or not isinstance(node.get("class_type"), str)
            or not isinstance(node.get("inputs"), dict)
            for node_id, node in raw.items()
        ):
            raise WorkflowError("image workflow must be an API-format node map")
        return raw

    def _input(self, node_id: str, input_name: str) -> None:
        _require_input(self.graph, node_id, input_name)

    def for_request(self, prompt: str, *, seed: int | None = None) -> dict[str, dict[str, Any]]:
        if not prompt.strip():
            raise WorkflowError("image prompt is empty")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise WorkflowError("image seed must be a nonnegative 64-bit integer")
        if seed is not None and self.seed_node_id is None:
            raise WorkflowError("image workflow has no seed binding")
        graph = copy.deepcopy(self.graph)
        graph[self.prompt_node_id]["inputs"][self.prompt_input_name] = prompt
        if seed is not None and self.seed_node_id is not None:
            graph[self.seed_node_id]["inputs"][self.seed_input_name] = seed
        return graph


@dataclass(frozen=True)
class ComfyImageInputWorkflow:
    """A workflow that takes the caller's image: an edit (with a prompt) or a variation.

    A variation has no prompt in the OpenAI contract. If an operator's graph
    needs an instruction, the instruction lives in the graph, not in Chord."""
    graph: dict[str, dict[str, Any]]
    image_node_id: str
    output_node_id: str
    image_input_name: str = "image"
    prompt_node_id: str | None = None
    prompt_input_name: str = "text"
    seed_node_id: str | None = None
    seed_input_name: str = "seed"

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        image_node_id: str,
        output_node_id: str,
        image_input_name: str = "image",
        prompt_node_id: str | None = None,
        prompt_input_name: str = "text",
        seed_node_id: str | None = None,
        seed_input_name: str = "seed",
    ) -> ComfyImageInputWorkflow:
        graph = ComfyImageWorkflow._graph(path)
        workflow = cls(graph, image_node_id, output_node_id, image_input_name,
                       prompt_node_id, prompt_input_name, seed_node_id, seed_input_name)
        _require_input(graph, image_node_id, image_input_name)
        if output_node_id not in graph:
            raise WorkflowError(f"image workflow has no output node {output_node_id!r}")
        if prompt_node_id is not None:
            _require_input(graph, prompt_node_id, prompt_input_name)
        if seed_node_id is not None:
            _require_input(graph, seed_node_id, seed_input_name)
        return workflow

    def for_request(self, image_name: str, *, prompt: str | None = None,
                    seed: int | None = None) -> dict[str, dict[str, Any]]:
        if not image_name:
            raise WorkflowError("image workflow needs the uploaded image name")
        if (prompt is None) != (self.prompt_node_id is None):
            raise WorkflowError("image workflow prompt does not match its prompt binding")
        if prompt is not None and not prompt.strip():
            raise WorkflowError("image prompt is empty")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise WorkflowError("image seed must be a nonnegative 64-bit integer")
        if seed is not None and self.seed_node_id is None:
            raise WorkflowError("image workflow has no seed binding")
        graph = copy.deepcopy(self.graph)
        graph[self.image_node_id]["inputs"][self.image_input_name] = image_name
        if prompt is not None and self.prompt_node_id is not None:
            graph[self.prompt_node_id]["inputs"][self.prompt_input_name] = prompt
        if seed is not None and self.seed_node_id is not None:
            graph[self.seed_node_id]["inputs"][self.seed_input_name] = seed
        return graph


def _require_input(graph: dict[str, dict[str, Any]], node_id: str, input_name: str) -> None:
    node = graph.get(node_id)
    if node is None or input_name not in node["inputs"]:
        raise WorkflowError(f"image workflow has no input {node_id!r}.{input_name}")


class ImageComfyClient(Protocol):
    async def submit(self, graph: dict[str, Any]) -> str: ...
    async def poll(self, prompt_id: str) -> tuple[str, float]: ...
    async def fetch_pngs(self, prompt_id: str, output_node_id: str | None = None) -> list[bytes]: ...
    async def interrupt(self, prompt_id: str) -> bool: ...
    async def aclose(self) -> None: ...


async def _stop_when_submitted(
    submitting: asyncio.Task[str], client: ImageComfyClient
) -> None:
    """Own cleanup independently of the caller's cancelled task."""
    try:
        accepted = await submitting
    except (Exception, asyncio.CancelledError):
        return  # No accepted id is known to stop.
    try:
        await client.interrupt(accepted)
    except Exception:
        logger.exception("could not stop accepted image prompt %s", accepted)


@dataclass
class ComfyImageBackend:
    workflow: ComfyImageWorkflow
    client: ImageComfyClient
    timeout_s: float = 300.0
    poll_interval_s: float = 1.5
    render_lock: asyncio.Lock | None = None
    _cleanup_tasks: set[asyncio.Task[Any]] = field(default_factory=set, init=False, repr=False)

    async def render(self, prompt: str, *, seed: int | None = None) -> tuple[bytes, str]:
        # One deadline covers the GPU queue as well as submission and rendering.
        # An expired request must never submit after finally acquiring the lock.
        try:
            async with asyncio.timeout(self.timeout_s):
                if self.render_lock is None:
                    return await self._render(prompt, seed=seed)
                async with self.render_lock:
                    return await self._render(prompt, seed=seed)
        except TimeoutError:
            raise TimeoutError(f"image workflow timed out after {self.timeout_s:g}s") from None

    async def _render(self, prompt: str, *, seed: int | None = None) -> tuple[bytes, str]:
        """Return one PNG and ComfyUI's prompt id; never silently resubmit."""
        graph = self.workflow.for_request(prompt, seed=seed)
        return await _run_one_png(self.client, graph, self.workflow.output_node_id,
                                  self.poll_interval_s, self._cleanup_tasks)

    async def aclose(self) -> None:
        await self.client.aclose()


class ImageInputComfyClient(ImageComfyClient, Protocol):
    async def upload_image(self, data: bytes, filename: str) -> str: ...


@dataclass
class ComfyImageInputBackend:
    """Edits and variations through an operator's workflow, one PNG per submit.

    Variations are `n` submits of the same workflow with fresh seeds, so the
    graph never has to be rewritten to sample several images. That needs a
    seed binding when `n` is above 1; without one every submit would be the
    same picture."""
    workflow: ComfyImageInputWorkflow
    client: ImageInputComfyClient
    timeout_s: float = 1500.0
    poll_interval_s: float = 1.5
    render_lock: asyncio.Lock | None = None
    _cleanup_tasks: set[asyncio.Task[Any]] = field(default_factory=set, init=False, repr=False)

    @property
    def max_variations(self) -> int:
        return 10 if self.workflow.seed_node_id is not None else 1

    async def edit_image(self, image: bytes, filename: str, prompt: str) -> bytes:
        (png,) = await self._render(image, filename, prompt, count=1)
        return png

    async def vary_images(self, image: bytes, filename: str, n: int) -> list[bytes]:
        if not 1 <= n <= self.max_variations:
            raise WorkflowError(f"this image workflow can return 1 to {self.max_variations} variations")
        return await self._render(image, filename, None, count=n)

    async def _render(self, image: bytes, filename: str, prompt: str | None, *, count: int) -> list[bytes]:
        # One deadline per image covers the GPU queue, the upload and every submit.
        deadline = self.timeout_s * count
        try:
            async with asyncio.timeout(deadline):
                if self.render_lock is None:
                    return await self._render_unlocked(image, filename, prompt, count)
                async with self.render_lock:
                    return await self._render_unlocked(image, filename, prompt, count)
        except TimeoutError:
            raise TimeoutError(f"image workflow timed out after {deadline:g}s") from None

    async def _render_unlocked(self, image: bytes, filename: str, prompt: str | None,
                               count: int) -> list[bytes]:
        if not image:
            raise WorkflowError("an input image is required")
        stored_name = await self.client.upload_image(image, filename)
        pngs = []
        for _ in range(count):
            seed = secrets.randbits(63) if self.workflow.seed_node_id is not None else None
            graph = self.workflow.for_request(stored_name, prompt=prompt, seed=seed)
            png, _ = await _run_one_png(self.client, graph, self.workflow.output_node_id,
                                        self.poll_interval_s, self._cleanup_tasks)
            pngs.append(png)
        return pngs

    async def aclose(self) -> None:
        await self.client.aclose()


async def _run_one_png(client: ImageComfyClient, graph: dict[str, Any], output_node_id: str,
                       poll_interval_s: float, cleanup_tasks: set[asyncio.Task[Any]]) -> tuple[bytes, str]:
    """Submit one filled graph and return its one PNG and ComfyUI's prompt id.

    A cancellation during submission waits for its id before attempting a
    by-id stop. Running-job interruption has ComfyUI's documented race, so
    this narrows rather than guarantees cancellation of GPU work.
    """
    submitting = asyncio.ensure_future(client.submit(graph))
    try:
        prompt_id = await asyncio.shield(submitting)
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(_stop_when_submitted(submitting, client))
        cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(cleanup_tasks.discard)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # A second cancel reaches this task, but cleanup owns the
            # accepted id and continues without it.
            pass
        raise

    try:
        while True:
            status, _ = await client.poll(prompt_id)
            if status == comfy.FAILED:
                raise WorkflowError("image workflow failed on ComfyUI")
            if status == comfy.COMPLETED:
                pngs = await client.fetch_pngs(prompt_id, output_node_id)
                if len(pngs) != 1 or not pngs[0]:
                    raise WorkflowError("image workflow must save exactly one nonempty PNG")
                return pngs[0], prompt_id
            await asyncio.sleep(poll_interval_s)
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(client.interrupt(prompt_id))
        cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(cleanup_tasks.discard)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # The independent stop still owns the accepted prompt id.
            pass
        raise
