"""Loopback-only evaluation and trace-inspection application."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException
from langchain_core.language_models import BaseChatModel

from . import graph as graph_mod
from .artifacts import ArtifactStore
from .config import Settings
from .comfy_workflow import ComfyImageBackend, ComfyImageInputBackend
from .contract import Job, Result
from .registry import Capability
from .specialists import SPECIALISTS, SpecialistContext
from .trace import Trace, TraceSink
from . import manifest
from .videos import video_tools_available


class InternalDeps(Protocol):
    settings: Settings
    artifacts: ArtifactStore
    traces: TraceSink
    specialists: list[str]
    capabilities: dict[str, Capability]
    model: Callable[[str], BaseChatModel]
    image_backend: ComfyImageBackend | None
    edit_backend: ComfyImageInputBackend | None
    variation_backend: ComfyImageInputBackend | None


def create_app(deps: InternalDeps) -> FastAPI:
    """Build the app that must remain bound to loopback."""
    app = FastAPI(title="chord internal")

    @app.get("/internal/health")
    async def internal_health():
        image = deps.capabilities.get("image")
        configured_image = deps.image_backend is not None
        return {
            "status": "ok",
            "revision": os.environ.get("CHORD_REVISION", "dev"),
            "personas": graph_mod.available_personas(),
            "router_enabled": deps.settings.router_enabled,
            "specialists": deps.specialists,
            "enabled_routes": sorted(deps.settings.enabled_routes),
            "capabilities": {
                "input": {"image": bool(manifest.load()["input"]["image"])},
                "output": {"image": configured_image,
                           "image_edit": deps.edit_backend is not None,
                           "image_variation": deps.variation_backend is not None},
                "image_chat_routable": bool(
                    deps.settings.router_enabled and image
                    and "image" in deps.specialists and configured_image
                ),
                "video": getattr(deps, "video_backend", None) is not None and video_tools_available(),
            },
            "public_artifact_links": bool(deps.settings.public_artifact_base),
        }

    @app.post("/internal/specialists/{capability_id}/invoke")
    async def invoke(capability_id: str, job: Job) -> Result:
        run = SPECIALISTS.get(capability_id)
        if run is None:
            raise HTTPException(404, f"no specialist registered for {capability_id!r}")
        trace = Trace(persona_id=job.persona_id, model_id_requested=f"internal:{capability_id}", path="eval")
        ctx = SpecialistContext(
            settings=deps.settings,
            artifacts=deps.artifacts,
            trace=trace,
            model=deps.model,
            image_backend=deps.image_backend,
        )
        try:
            return await run(job, ctx)
        finally:
            deps.traces.write(trace)

    @app.get("/internal/traces/{trace_id}")
    async def get_trace(trace_id: str):
        # Line by line, never read_text(): the trace directory holds a whole
        # retention window (CHORD_RETENTION_DAYS, 30 days by default, forever
        # at 0), and one whole-corpus allocation per lookup froze both
        # servers on the loop. A torn line (an OOM kill mid-append) is skipped
        # rather than breaking every lookup for every older trace, which the
        # unguarded json.loads used to do -- and this app registers no
        # exception handlers, so that was a bare 500 (review 2026-09-22).
        for path in sorted(Path(deps.settings.trace_dir).glob("*.jsonl"), reverse=True):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict) and record.get("trace_id") == trace_id:
                        return record
        raise HTTPException(404)

    return app
