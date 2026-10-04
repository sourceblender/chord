"""Exercise the configured inference backends from the candidate container.

This is a deployment check, not a benchmark. Each request is small, bounded,
and sent directly to the same address and model the application will use. It
prints only the backend name and a short status; response bodies and credentials
never enter the deploy log.
"""

from __future__ import annotations

import io
import sys
import wave
from typing import Any

import httpx

from .config import Settings, load_settings
from .upstream import embeddings_endpoint

TIMEOUT = httpx.Timeout(20.0, connect=5.0)


def _silence() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(bytes(16_000))  # half a second, 16-bit mono
    return buffer.getvalue()


def _json_ok(response: httpx.Response, field: str) -> bool:
    if response.status_code != 200:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and isinstance(body.get(field), list) and bool(body[field])


def check(settings: Settings, client: httpx.Client) -> list[str]:
    """Return backend names whose small inference request did not succeed."""
    failures: list[str] = []

    def request(name: str, url: str, *, auth: str = "", **kwargs: Any) -> httpx.Response | None:
        headers = {"Authorization": auth} if auth else {}
        try:
            return client.post(url, headers=headers, **kwargs)
        except (httpx.HTTPError, OSError):
            failures.append(name)
            return None

    # Each text writer once per distinct target: the same address can serve
    # different models, or the same model behind different credentials, and each
    # of those is checked. A writer that shares a target with an earlier one
    # shares its result, so the names reported are the same as probing both.
    writers = [("persona", settings.persona_model, settings.persona_base_url),
               ("router", settings.router_model, settings.router_base_url)]
    if settings.fast_model and settings.fast_base_url:  # version 2 names its own fast writer
        writers.append(("fast", settings.fast_model, settings.fast_base_url))
    probed: dict[tuple[str, str, str], bool] = {}
    for name, model, base in writers:
        if not base:
            failures.append(name)
            continue
        key = settings.credential_for(base)
        target = (base.rstrip("/"), model, key)
        if target not in probed:
            response = request(
                name, base.rstrip("/") + "/chat/completions",
                auth=f"Bearer {key}" if key else "",
                json={"model": model, "messages": [{"role": "user", "content": "Reply OK"}],
                      "max_tokens": 8, "stream": False},
            )
            probed[target] = response is not None and _json_ok(response, "choices")
            if response is not None and not probed[target]:
                failures.append(name)
        elif not probed[target] and name not in failures:
            failures.append(name)

    # A version 2 classifier dispatcher gets a lane request, never a chat one:
    # it returns a lane name and cannot write.
    if settings.config_version == 2 and settings.router_enabled and settings.router_backend == "classifier":
        from .router import CLASSIFIER_ROUTES
        response = request("classifier", settings.router_classifier_url, json={"text": "user: Reply OK"})
        try:
            valid = (response is not None and response.status_code == 200
                     and response.json().get("route") in CLASSIFIER_ROUTES)
        except (ValueError, AttributeError):
            valid = False
        if response is not None and not valid:
            failures.append("classifier")

    if settings.stt_base_url:
        key = settings.credential_for(settings.stt_base_url)
        response = request(
            "stt", settings.stt_base_url.rstrip("/") + "/audio/transcriptions",
            auth=f"Bearer {key}" if key else "",
            data={"model": settings.stt_model},
            files={"file": ("probe.wav", _silence(), "audio/wav")},
        )
        try:
            valid = response is not None and response.status_code == 200 and isinstance(response.json().get("text"), str)
        except (ValueError, AttributeError):
            valid = False
        if response is not None and not valid:
            failures.append("stt")

    if settings.tts_base_url:
        key = settings.credential_for(settings.tts_base_url)
        response = request(
            "tts", settings.tts_base_url.rstrip("/") + "/audio/speech",
            auth=f"Bearer {key}" if key else "",
            json={"model": settings.tts_model, "input": "test", "voice": settings.voice_for("generic") or "alloy",
                  "response_format": "wav"},
        )
        if response is not None and not (response.status_code == 200 and response.content.startswith(b"RIFF")):
            failures.append("tts")

    if settings.embeddings_base_url:
        response = request(
            "embeddings", settings.embeddings_base_url.rstrip("/") + embeddings_endpoint(settings.embeddings_base_url),
            auth=settings.embeddings_auth_header(),
            json={"model": settings.embeddings_model, "input": "test", "encoding_format": "float"},
        )
        if response is not None and not _json_ok(response, "data"):
            failures.append("embeddings")

    return failures


def main() -> int:
    try:
        settings = load_settings()
        with httpx.Client(timeout=TIMEOUT, follow_redirects=False) as client:
            failures = check(settings, client)
    except (ValueError, RuntimeError):
        print("backend-preflight: invalid configuration", file=sys.stderr)
        return 1
    if failures:
        print(f"backend-preflight: failed {', '.join(failures)}", file=sys.stderr)
        return 1
    print("backend-preflight: configured inference backends responded")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
