"""OpenAI Models API routes.

This module is intentionally self-contained: model inventory policy and the
wire shapes for list, retrieve, and the declared no-op delete live together.
"""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import graph as graph_mod
from . import manifest
from .config import Settings


class ModelsDeps(Protocol):
    settings: Settings


def served_models(settings: Settings) -> list[str]:
    """Model IDs this deployment can actually answer for."""
    ids = [graph_mod.MODEL_ID]
    if settings.embeddings_base_url:
        ids.append(settings.embeddings_model)
    return ids


def _error(status: int, message: str, code: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code,
        }},
        status_code=status,
    )


def register(app: FastAPI, deps: ModelsDeps) -> None:
    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [manifest.models_entry(model_id) for model_id in served_models(deps.settings)],
        }

    # Model IDs may contain a slash, so the route consumes the rest of the path.
    @app.get("/v1/models/{model_id:path}")
    async def model(model_id: str):
        if model_id not in set(served_models(deps.settings)):
            return _error(
                404,
                f"model {model_id!r} not served; use one of /v1/models",
                "model_not_found",
                "model",
            )
        return manifest.models_entry(model_id)

    @app.delete("/v1/models/{model_id:path}")
    async def delete_model(model_id: str):
        """Answer in the OpenAI shape while deleting nothing.

        Chord does not own fine-tuned models. Harnesses can still walk the
        Models API without turning this intentional no-op into an error path.
        The invariant is explicit: this handler cannot report ``deleted: true``.
        """
        return {"id": model_id, "object": "model", "deleted": False}
