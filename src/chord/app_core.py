"""Cross-cutting public-app authentication and error contracts."""

from __future__ import annotations

import time
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from .artifact_links import is_signed_request
from .client_keys import identify_client, parse_client_keys
from .client_admission import ClientAdmission
from .config import Settings
from .contract import Outcome
from .trace import Trace, TraceSink


class AppCoreDeps(Protocol):
    settings: Settings
    traces: TraceSink


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        {"error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": code,
        }},
        status_code=status,
    )


class BodyTooLargeError(Exception):
    """Raised from inside receive when a counted body crosses the cap."""


def _too_large(max_bytes: int) -> JSONResponse:
    return _error(413, f"request body is larger than {max_bytes} bytes", "request_too_large")


# The ONLY routes whose multipart declaration earns the upload tier: exactly
# the routes that parse with request.form(), where Starlette's disk spooling
# is the intended mechanism and per-part caps run after the parse. The
# exemption used to trust the client's Content-Type header instead, so a JSON
# body wearing `multipart/form-data` walked past the cap into any
# request.json() route -- none of which check the header (review
# 2026-09-22, #2, reproduced). The list is pinned by a drift test that feeds
# a lying Content-Type to EVERY other POST route and demands the 413: a new
# form route that is not added here deliberately fails that test.
MULTIPART_PATHS = frozenset({
    "/v1/files",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/images/edits",
    "/v1/images/variations",
    "/v1/videos",
})

# Aggregate bound for one upload body, set by the largest legitimate one
# (/v1/files: 512 MB plus slack). Per-part caps belong to the routes; this
# tier exists because Starlette spools the whole body to disk before any
# per-part check can run (#3).
MULTIPART_MAX_BYTES = 512 * 1024 * 1024 + 2 * 1024 * 1024


class JsonBodyLimit:
    """Cap every request body before anything parses it.

    `await request.json()` had no bound at all: a multi-gigabyte chat body was
    parsed and held whole in one process (review 2026-09-22). Two enforcement
    points -- a declared Content-Length is refused without reading a byte, and
    a counter inside receive catches a chunked body that declares nothing (or
    lies small). Two tiers: the JSON cap everywhere, and the upload tier on
    the pinned MULTIPART_PATHS when -- and only when -- the request actually
    declares multipart with a SINGLE Content-Type header (a duplicate header
    is a request that cannot be trusted to describe itself: Starlette reads
    the first, this middleware once kept the last, and the disagreement was
    the bypass). Added LAST, so it sits OUTSIDE the API-key check: an
    unauthenticated flood is refused without being parsed. 0 disables the
    JSON tier; the upload tier always stands."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw = scope.get("headers") or []
        content_types = [v.decode("latin-1") for k, v in raw if k.decode("latin-1").lower() == "content-type"]
        lengths = [v.decode("latin-1") for k, v in raw if k.decode("latin-1").lower() == "content-length"]
        path = (scope.get("path") or "").rstrip("/") or "/"
        exempt = (path in MULTIPART_PATHS
                  and len(content_types) == 1
                  and content_types[0].lower().startswith("multipart/form-data"))
        cap = MULTIPART_MAX_BYTES if exempt else self.max_bytes
        if cap <= 0:
            await self.app(scope, receive, send)
            return
        if len(lengths) == 1:
            try:
                if int(lengths[0]) > cap:
                    await _too_large(cap)(scope, receive, send)
                    return
            except ValueError:
                pass                       # an unparsable declaration: the counter decides
        counted = 0
        response_started = False

        async def capped_receive() -> Any:
            nonlocal counted
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body")
                counted += len(body) if isinstance(body, bytes) else 0
                if counted > cap:
                    raise BodyTooLargeError()
            return message

        async def tracking_send(message: Any) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, capped_receive, tracking_send)
        except* BodyTooLargeError:
            # Answered HERE, not by a registered exception handler: the raise
            # crosses the require_key BaseHTTPMiddleware's task group and
            # re-emerges as an ExceptionGroup ABOVE the router-level handlers,
            # which never see it. except* matches both the bare and grouped
            # forms. If a response already started there is nothing honest
            # left to send; the connection error stands.
            if response_started:
                raise
            await _too_large(cap)(scope, receive, send)


def register(app: FastAPI, deps: AppCoreDeps) -> None:
    clients = parse_client_keys(deps.settings.client_keys_json, deps.settings.service_api_key)
    # Authentication wraps this ASGI middleware. It sees the authenticated
    # client_id and holds an in-flight slot until the last streaming byte.
    app.add_middleware(ClientAdmission,
                       global_inflight=deps.settings.max_global_inflight,
                       client_inflight=deps.settings.max_client_inflight,
                       global_per_minute=deps.settings.max_global_per_minute,
                       client_per_minute=deps.settings.max_client_per_minute,
                       traces=deps.traces)
    @app.exception_handler(StarletteHTTPException)
    async def openai_shaped_http_error(request: Request, exc: StarletteHTTPException):
        routing = exc.detail in ("Not Found", "Method Not Allowed")
        message = f"Invalid URL ({request.method} {request.url.path})" if routing else str(exc.detail)
        return JSONResponse(
            {"error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def openai_shaped_unhandled_error(request: Request, exc: Exception):
        trace = Trace(persona_id="unknown", model_id_requested=None)
        trace.set(
            endpoint=f"{request.method} {request.url.path}",
            result_status=Outcome.failed.value,
            unhandled_exception=repr(exc)[:500],
        )
        deps.traces.write(trace)
        headers = {"x-request-id": trace.trace_id, "x-chord-trace-id": trace.trace_id}
        return JSONResponse(
            {"error": {
                "message": "The service could not complete the request.",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }},
            status_code=500,
            headers=headers,
        )

    @app.middleware("http")
    async def require_key(request: Request, call_next):
        key = deps.settings.service_api_key
        authorization = request.headers.getlist("authorization")
        caller = identify_client(authorization[0] if len(authorization) == 1 else "", key, clients,
                                 legacy_enabled=deps.settings.accept_legacy_client_key)
        if request.url.path == "/health":
            request.state.client_id = caller
            return await call_next(request)
        signer = deps.settings.artifact_signer_key
        signed = bool(signer) and is_signed_request(request, signer, deps.settings.artifact_url_ttl_s)
        if (not signed and key and key != signer
                and time.time() < deps.settings.legacy_artifact_verify_until):
            signed = is_signed_request(request, key, deps.settings.artifact_url_ttl_s)
        if signed:
            request.state.client_id = "signed-link"
            return await call_next(request)
        if caller is None:
            return _error(401, "missing or invalid API key", "invalid_api_key")
        request.state.client_id = caller
        return await call_next(request)

    # Added last, so it wraps the key check: an oversized body is refused
    # before authentication parses anything (see JsonBodyLimit). Its counted
    # path answers from inside the middleware itself -- a raise from receive
    # crosses require_key's task group as an ExceptionGroup that router-level
    # handlers never see.
    app.add_middleware(JsonBodyLimit, max_bytes=deps.settings.max_json_body_bytes)
