"""Application composition root for Chord's public OpenAI-compatible API."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import (
    app_core,
    audio_api,
    chat_api,
    completions_api,
    embeddings,
    files_api,
    fingerprint,
    images_api,
    internal_api,
    models_api,
    moderations,
    responses,
    stored_chat,
    system_api,
    videos,
)
from .api_errors import upstream_error_body as _upstream_error_body
from .chat_contract import CHAT_SPEC_PARAMS
from .chat_wire import _construct as _construct
from .dependencies import Deps, load_specialists
from .responses_store import ResponseStore

__all__ = [
    "CHAT_SPEC_PARAMS",
    "Deps",
    "FORCED_CALL_CODE",
    "JSONResponse",
    "NOOP_VALIDATORS",
    "_construct",
    "_upstream_error_body",
    "create_app",
    "create_internal_app",
    "load_specialists",
    "time",
]

# Compatibility exports for callers and conformance tests that historically
# imported Chat helpers from the composition root. New code should import the
# owning module directly.
FORCED_CALL_CODE = chat_api.FORCED_CALL_CODE
NOOP_VALIDATORS = chat_api.NOOP_VALIDATORS
error = chat_api.error
_conform_choices = chat_api._conform_choices
_split = chat_api._split
_validate = chat_api._validate
opaque_fingerprint = fingerprint.opaque
create_internal_app = internal_api.create_app


def _sync_chat_compatibility_hooks() -> None:
    """Test seam: legacy tests monkeypatch `chord.server.error`,
    `chord.server._conform_choices`, `chord.server._split`, and
    `chord.server._validate`. The chat-core module reads these names through
    its own globals, so a monkeypatch on `server` would not normally reach
    it. Rebinding `chat_api.<name>` here re-reads `server.<name>` at app
    construction time, which picks up any monkeypatch applied BEFORE
    `create_app()` runs. The order matters: patch first, then call
    `create_app()`, then send the request. Used by
    `tests/test_conformance_schema.py::test_offline_red_the_old_gap_still_fails_the_schema`."""
    chat_api.error = error
    chat_api._conform_choices = _conform_choices
    chat_api._split = _split
    chat_api._validate = _validate


def create_app(deps: Deps) -> FastAPI:
    """Compose the public application from independently registered families."""
    _sync_chat_compatibility_hooks()
    app = FastAPI(title="chord")
    stored = ResponseStore(deps.settings.data_dir / "responses.sqlite3")
    fingerprint.configure(deps.settings.data_dir)
    app_core.register(app, deps)

    system_api.register(app, deps)
    models_api.register(app, deps)
    images_api.register(app, deps, stored)
    completions_api.register(app, deps)
    chat_api.register(app, deps, stored)
    audio_api.register(app, deps)
    responses.register(app, deps, stored)
    stored_chat.register(app, stored)
    files_api.register(app, deps, stored)
    moderations.register(app, deps)
    embeddings.register(app, deps)
    videos.register(app, deps, stored)

    return app
