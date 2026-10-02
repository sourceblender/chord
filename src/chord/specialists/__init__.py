"""Specialist interface.

A specialist is one async callable:

    async def run(job: Job, ctx: SpecialistContext) -> Result

It owns its prompt and tool sequence. It may not deliver anything, pick a
recipient, change the assistant's identity, or write long-term memory. Files go
through `ctx.artifacts.register(...)`, and the returned descriptor goes in
`Result.artifacts`. Every tool call and check it makes is appended to
`ctx.trace` explicitly (`tool_calls`, `checks`), because nothing records it for you.

Register the callable in `SPECIALISTS` under its registry id (registry.yaml).
Registration makes it *invokable* on the internal eval path. It becomes
*routable* from the public alias only once registry.yaml certifies it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from ..artifacts import ArtifactStore
from ..config import Settings
from ..comfy_workflow import ComfyImageBackend
from ..contract import Job, Result
from ..trace import Trace


@dataclass
class SpecialistContext:
    settings: Settings
    artifacts: ArtifactStore
    trace: Trace
    # model(name) -> a chat model at its configured backend. Use the registry's
    # model name so the trace and certification agree on what ran.
    model: Callable[[str], BaseChatModel]
    # progress(stage): report that something just happened, e.g. "preparing",
    # "submitting", "retrying" (stage names in progress.py). Call it at the
    # real event, never ahead of it. How it's shown is not the worker's
    # concern; an unknown stage is traced, not shown.
    progress: Callable[[str], None] = lambda stage: None
    # The configured provider owns scene enhancement and rendering.
    image_backend: ComfyImageBackend | None = None


Specialist = Callable[[Job, SpecialistContext], Awaitable[Result]]

SPECIALISTS: dict[str, Specialist] = {}


def specialist(capability_id: str):
    """Decorator: `@specialist("image")` registers a callable under its id."""

    def wrap(fn: Specialist) -> Specialist:
        SPECIALISTS[capability_id] = fn
        return fn

    return wrap
