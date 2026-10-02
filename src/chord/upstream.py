"""Raw OpenAI-compatible calls to configured model endpoints.

The voice path does not go through a LangChain model. A LangChain wrapper
would normalise away fields a real model returns (reasoning content, usage,
chunk shape), and "behaves like a real model" is judged against a direct call
to the same deployment. So we forward the client's parameters untouched and
hand back what the model sent.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping

import httpx

# Headers LiteLLM sets naming the deployment that actually answered. These are
# the receipt that a local model, not a fallback, produced the reply.
DEPLOYMENT_HEADERS = ("x-litellm-model-api-base", "x-litellm-model-id", "x-litellm-model-group")


def embeddings_endpoint(base_url: str) -> str:
    """The request path relative to a service root or its /v1 API base."""
    path = httpx.URL(base_url).path.rstrip("/")
    return "/embeddings" if path.endswith("/v1") else "/v1/embeddings"


def deployment(headers: httpx.Headers) -> dict:
    return {h.removeprefix("x-litellm-"): headers.get(h) for h in DEPLOYMENT_HEADERS if headers.get(h)}


class UpstreamError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"upstream {status}: {body[:300]}")
        self.status = status
        self.body = body


def _frame(payload):
    """One parsed SSE frame, or the failure it reports.

    A backend that fails after its 200 reports it IN the stream, as
    `data: {"error": {...}}` (vLLM, LiteLLM). Passed on as a chunk it has no
    `choices`, so the wire builder kept an empty frame, dropped the error, and
    the stream closed with a manufactured `finish_reason: "stop"` -- truncated
    text dressed as a finished answer (review 2026-09-27).
    Raising here routes it through the same `stream_failed` path as a transport
    failure. An explicit error wins even beside partial `choices` (#346):
    a frame that says it failed is never forwarded as content, including an
    empty `{}` or `""` error. Only a null `error` field is not a failure.
    The backend's own text travels only in `body`, which the error envelope
    never relays."""
    if isinstance(payload, dict) and payload.get("error", None) is not None:
        error = payload["error"]
        code = error.get("code") if isinstance(error, dict) else None
        status = code if isinstance(code, int) and 400 <= code < 600 else 502
        raise UpstreamError(status, json.dumps(payload))
    return payload


class UpstreamUnconfiguredError(UpstreamError):
    """Raised when a request needs an upstream capability the deployment has not
    configured (e.g. STT or TTS with no address set).

    Distinct from `UpstreamError` so audio handlers can map it to a 503 with a
    clear "not configured" message, instead of swallowing it into the generic
    502 backend-failure response a real upstream failure gets.
    """

    def __init__(self, body: str) -> None:
        super().__init__(503, body)


class Upstream:
    """One client per PURPOSE, not one per service.

    Chat, speech-to-text and text-to-speech are three different models on three
    different hosts once the gateway is out of the middle. Sharing a single base
    URL between them was invisible while everything went through LiteLLM and became
    a 502 the moment chat moved: audio kept posting /audio/transcriptions to a vLLM
    that serves only chat (measured on prod 7032f11, both audio routes 502).

    `stt_base_url`, `tts_base_url` and `embeddings_base_url` default to the chat one,
    so a deployment that sets none behaves exactly as it did before.

    Embeddings is a fourth purpose and a fourth host: the fleet serves them from a
    text-embeddings-inference service, not from the chat model. Pointing it at the
    chat base URL would 404 for the same reason audio did on prod 7032f11."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 600.0,
                 stt_base_url: str = "", tts_base_url: str = "",
                 embeddings_base_url: str = "", embeddings_auth: str = "",
                 chat_routes: Mapping[str, str] | None = None,
                 keys: Mapping[str, str] | None = None,
                 refuse_unset_audio: bool = False) -> None:
        # `keys` maps a base URL to the credential for THAT address. Absent or empty means
        # the client sends no `Authorization` header at all -- not an empty bearer, no
        # header. `api_key` remains the fallback only when `keys` is None, which keeps
        # every direct unit construction behaving as before; production always passes a
        # complete map and a cell asserts it does. (review HIGH-2, 2026-09-19: one
        # key was applied to every client, so the gateway credential reached four
        # backends that are not the gateway.)
        def client(url: str, authorization: str | None = None) -> httpx.AsyncClient:
            token = api_key if keys is None else keys.get(url.rstrip("/"), "")
            header = authorization if authorization is not None else (f"Bearer {token}" if token else "")
            return httpx.AsyncClient(
                base_url=url.rstrip("/"),
                headers={"Authorization": header} if header else {},
                timeout=httpx.Timeout(timeout, connect=10.0),
            )
        self._client = client(base_url)
        self._chat_routes = None if chat_routes is None else {
            model: self._client if url.rstrip("/") == base_url.rstrip("/") else client(url)
            for model, url in chat_routes.items()
        }
        # Direct unit constructions keep the old single-client fallback. Production
        # (Deps) passes refuse_unset_audio=True: with no STT/TTS address configured,
        # audio must refuse with a clear 503 instead of posting audio routes to the
        # chat backend, which cannot serve them (review of the sanitisation PR,
        # 2026-09-25: an empty gateway default made that fallback reachable).
        unset: httpx.AsyncClient | None = None if refuse_unset_audio else self._client
        self._stt = client(stt_base_url) if stt_base_url else unset
        self._tts = client(tts_base_url) if tts_base_url else unset
        # The shared-inference ingress in front of TEI speaks Basic, not the
        # LiteLLM Bearer every other upstream uses. Inheriting the chat header
        # here passes every fake-upstream test and 401s against the live host —
        # the exact shape of a green that proves nothing, so the credential is a
        # separate seam rather than an assumption.
        self._embeddings = (client(embeddings_base_url, embeddings_auth or None)
                            if embeddings_base_url else self._client)

    def _chat_client(self, body: dict) -> httpx.AsyncClient:
        """Select only an explicitly configured direct chat route.

        Legacy/direct unit callers without ``chat_routes`` retain the single
        client. Production supplies a complete settings-derived map; an
        unknown model then refuses instead of falling through to a gateway.
        """
        if self._chat_routes is None:
            return self._client
        model = body.get("model")
        if not isinstance(model, str):
            raise RuntimeError(f"no direct chat route configured for model {model!r}")
        try:
            return self._chat_routes[model]
        except KeyError as exc:
            raise RuntimeError(f"no direct chat route configured for model {model!r}") from exc

    async def aclose(self) -> None:
        route_clients = () if self._chat_routes is None else tuple(self._chat_routes.values())
        clients = (self._client, self._stt, self._tts, self._embeddings, *route_clients)
        for c in {id(x): x for x in clients if x is not None}.values():
            await c.aclose()

    async def complete(self, body: dict) -> tuple[dict, dict]:
        resp = await self._chat_client(body).post("/chat/completions", json={**body, "stream": False})
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, resp.text)
        return self._json(resp, "chat completion"), deployment(resp.headers)

    @staticmethod
    def _json(resp: httpx.Response, what: str) -> dict:
        """A 200 that is not JSON is the backend's failure, not ours: a 502 in
        the API error vocabulary, never a JSONDecodeError escaping through the
        generic handler as 500 internal_error (review 2026-09-22; the embedding
        token decoder, removed in review 2026-09-24 B1, already guarded its own
        parse -- the unguarded calls beside it were oversight, not choice)."""
        try:
            return resp.json()
        except ValueError as exc:
            raise UpstreamError(502, f"upstream {what} returned a 200 that is not JSON") from exc

    async def transcribe(self, audio: bytes, fmt: str, mime: str, model: str,
                         *, language: str | None = None) -> tuple[str, dict]:
        """OpenAI audio/transcriptions (multipart). Returns (text, deployment)."""
        if self._stt is None:
            raise UpstreamUnconfiguredError("speech-to-text is not configured: set STT_BASE_URL")
        resp = await self._stt.post(
            "/audio/transcriptions",
            data={"model": model, **({"language": language} if language is not None else {})},
            files={"file": (f"voice.{fmt}", audio, mime)},
        )
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, resp.text)
        return self._json(resp, "transcription").get("text", ""), deployment(resp.headers)

    async def speak(self, text: str, voice: str, fmt: str, model: str) -> tuple[bytes, dict]:
        """OpenAI audio/speech. Returns (audio bytes, deployment). The content-type
        header is not trusted (a TTS backend we certified against labels WAV as audio/mpeg); the artifact
        store checks the magic bytes against the format asked for."""
        if self._tts is None:
            raise UpstreamUnconfiguredError("text-to-speech is not configured: set TTS_BASE_URL")
        resp = await self._tts.post(
            "/audio/speech", json={"model": model, "input": text, "voice": voice, "response_format": fmt}
        )
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, resp.text)
        return resp.content, deployment(resp.headers)

    async def embed(self, request: dict) -> tuple[dict, dict]:
        """TEI's OpenAI-compatible `/v1/embeddings`, passed through.

        This used to be a shim. Chord built the envelope, computed the float32
        base64 itself and always requested `float`, because the fleet ran TEI
        **1.2.0** — which accepts `encoding_format: "base64"` and returns floats
        anyway, an option honoured in the signature and not the behaviour.

        The operator upgraded the service to **1.9.4** while we were writing around the
        old one. base64 landed in 1.5, so TEI now does natively everything the
        shim existed to compensate for, and the shim is deleted rather than
        certified.

        The caller's `encoding_format` is forwarded, because the version that
        lied about it is no longer running — confirmed against the containers,
        not the repo, which still pinned 1.2.0 at the time of this change.

        The whole validated request goes through, rather than three fields
        rebuilt here. Reconstructing a subset silently drops anything the route
        accepts and this method forgot — `user` was already lost that way — and
        makes every future spec field a second edit somebody has to remember."""
        # httpx appends request paths to base_url even when they start with '/'.
        # Accept either a service root or the /v1 base used by chat/audio slots.
        endpoint = embeddings_endpoint(str(self._embeddings.base_url))
        resp = await self._embeddings.post(endpoint, json=request)
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, resp.text)
        return self._json(resp, "embeddings"), deployment(resp.headers)

    async def complete_text(self, body: dict) -> tuple[dict, dict]:
        """Legacy Completions. The prompt goes to the model as
        written: no chat template, no base layer."""
        resp = await self._chat_client(body).post("/completions", json={**body, "stream": False})
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, resp.text)
        return self._json(resp, "completion"), deployment(resp.headers)

    async def stream_text(self, body: dict) -> AsyncGenerator[tuple[dict | None, dict], None]:
        """Yields (None, deployment) once, then each parsed SSE chunk."""
        async with self._chat_client(body).stream("POST", "/completions", json={**body, "stream": True}) as resp:
            if resp.status_code >= 400:
                raise UpstreamError(resp.status_code, (await resp.aread()).decode(errors="replace"))
            yield None, deployment(resp.headers)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                yield _frame(json.loads(data)), {}

    async def stream(self, body: dict) -> AsyncGenerator[tuple[dict | None, dict], None]:
        """Yields (None, deployment) once, then each parsed SSE chunk."""
        async with self._chat_client(body).stream("POST", "/chat/completions", json={**body, "stream": True}) as resp:
            if resp.status_code >= 400:
                raise UpstreamError(resp.status_code, (await resp.aread()).decode(errors="replace"))
            yield None, deployment(resp.headers)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                yield _frame(json.loads(data)), {}
