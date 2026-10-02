"""Public service-health and Chord artifact-extension routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .config import Settings


class ArtifactLocator(Protocol):
    def locate(self, artifact_id: str) -> tuple[Path, str] | None: ...


class SystemDeps(Protocol):
    settings: Settings

    # Read-only member: a writable protocol attribute is INVARIANT, so the
    # concrete ArtifactStore a Deps holds would not satisfy it (pyright batch
    # A). Deps only ever reads, and a plain attribute satisfies a read-only
    # property protocol member.
    @property
    def artifacts(self) -> ArtifactLocator: ...


def register(app: FastAPI, deps: SystemDeps) -> None:
    @app.get("/health")
    async def health(request: Request):
        # Liveness is public; deployment detail requires a recognized client.
        if getattr(request.state, "client_id", None) is None:
            return {"status": "ok"}
        return {"status": "ok", "revision": os.environ.get("CHORD_REVISION", "dev")}

    @app.get("/v1/artifacts/{artifact_id}")
    async def artifact(artifact_id: str):
        found = deps.artifacts.locate(artifact_id)
        if not found:
            raise HTTPException(404, f"artifact {artifact_id!r} not found")
        path, mime = found
        return FileResponse(path, media_type=mime)
