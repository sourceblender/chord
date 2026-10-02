"""Shared fail-closed translation of transport and upstream failures."""

from __future__ import annotations

import json
import re
from typing import Protocol

from fastapi.responses import JSONResponse

from .chat_contract import CHAT_SPEC_PARAMS
from .trace import Trace, TraceSink
from .upstream import UpstreamError


class TraceDeps(Protocol):
    traces: TraceSink


def unreachable(deps: TraceDeps, trace: Trace, exc: Exception, headers: dict) -> JSONResponse:
    """Return an OpenAI-shaped 502 without publishing connection details."""
    trace.set(upstream_unreachable=repr(exc)[:300])
    deps.traces.write(trace)
    return JSONResponse(
        {"error": {
            "message": f"model gateway unreachable ({type(exc).__name__})",
            "type": "server_error",
            "param": None,
            "code": "upstream_unreachable",
        }},
        status_code=502,
        headers=headers,
    )


_NUMBER = r"-?\d+(?:\.\d+)?"
_RANGE = re.compile(rf"(?<![\w.])([a-z_]+) must be in \[({_NUMBER}), ({_NUMBER})\], got ({_NUMBER})\.")
_MINIMUM = re.compile(rf"(?<![\w.])([a-z_]+) must be at least ({_NUMBER}), got ({_NUMBER})\.")
_CONTEXT = re.compile(
    r"maximum context length is (\d+) tokens.*?(?:prompt contains at least|messages resulted in) (\d+)",
    re.DOTALL,
)
_UPSTREAM_RENAMES = {"max_tokens": ("max_completion_tokens",)}
_UPSTREAM_DEFAULT = {
    429: ("The model backend is busy. Retry the request later.", "rate_limit_error", "rate_limit_exceeded"),
    500: ("The model backend failed to answer the request.", "server_error", "upstream_error"),
    400: ("The model backend rejected the request.", "invalid_request_error", "invalid_request"),
}


def _sent_as(name: str, sent: dict) -> str | None:
    if name not in CHAT_SPEC_PARAMS:
        return None
    if name in sent:
        return name
    return next((alias for alias in _UPSTREAM_RENAMES.get(name, ()) if alias in sent), None)


def upstream_error_body(exc: UpstreamError, sent: dict) -> dict:
    """Rebuild an OpenAI error from allowlisted facts, never upstream prose."""
    try:
        raw = json.loads(exc.body)
        message = str(((raw.get("error") or {}) if isinstance(raw, dict) else {}).get("message") or "")
    except (json.JSONDecodeError, TypeError, AttributeError):
        message = exc.body if isinstance(exc.body, str) else ""
    status = exc.status

    def body(message: str, type_: str, param: str | None, code: str) -> dict:
        return {"error": {"message": message, "type": type_, "param": param, "code": code}}

    if status in (401, 403):
        return body("The model backend rejected Chord's credentials.",
                    "server_error", None, "upstream_auth_error")

    if status < 500 and status != 429:
        context = _CONTEXT.search(message)
        if context:
            limit, used = context.groups()
            return body(
                f"The input is at least {used} tokens, over the model's context limit of {limit} tokens.",
                "invalid_request_error", "messages", "context_length_exceeded",
            )
        if "ContextWindowExceeded" in message:
            return body(
                "The input is longer than the model's context limit.",
                "invalid_request_error", "messages", "context_length_exceeded",
            )
        ranged = _RANGE.search(message)
        if ranged and (param := _sent_as(ranged.group(1), sent)):
            _, low, high, got = ranged.groups()
            return body(
                f"{param} must be between {low} and {high}; got {got}.",
                "invalid_request_error", param, "invalid_value",
            )
        minimum = _MINIMUM.search(message)
        if minimum and (param := _sent_as(minimum.group(1), sent)):
            _, low, got = minimum.groups()
            return body(
                f"{param} must be at least {low}; got {got}.",
                "invalid_request_error", param, "invalid_value",
            )
    text, type_, code = _UPSTREAM_DEFAULT[429 if status == 429 else 500 if status >= 500 else 400]
    return body(text, type_, None, code)


def upstream_http_status(exc: UpstreamError) -> int:
    """A backend auth refusal is Chord's server failure, never the caller's."""
    return 502 if exc.status in (401, 403) else exc.status
