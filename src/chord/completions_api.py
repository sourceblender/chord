"""OpenAI Legacy Completions API validation, shaping, and transport."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Protocol

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import graph as graph_mod
from .api_errors import unreachable as _unreachable
from .api_errors import upstream_error_body as _upstream_error_body
from .api_errors import upstream_http_status as _upstream_http_status
from .config import Settings
from .contract import Outcome
from .fingerprint import opaque as opaque_fingerprint
from .http_transport import ClosingStreamingResponse
from .trace import Trace, TraceSink
from .upstream import UpstreamError


class CompletionUpstream(Protocol):
    async def complete_text(self, body: dict) -> tuple[dict, dict]: ...
    def stream_text(self, body: dict) -> AsyncIterator[tuple[dict | None, dict]]: ...


class CompletionsDeps(Protocol):
    settings: Settings
    # Read-only: a writable protocol attribute is invariant, and the concrete
    # Upstream a Deps holds is a subtype, not the protocol itself (batch A).
    @property
    def upstream(self) -> CompletionUpstream: ...
    traces: TraceSink


def error(status: int, message: str, code: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}},
        status_code=status,
    )


# Legacy Completions (S13b, red team 2026-09-15). The prompt
# goes to the model as written: no chat template, no base layer, no router.
# Every param the pinned CreateCompletionRequest defines is forwarded, refused,
# or dropped; nothing is silently ignored.
COMPLETION_SPEC_PARAMS = frozenset({
    "best_of", "echo", "frequency_penalty", "logit_bias", "logprobs", "max_tokens", "model", "n",
    "presence_penalty", "prompt", "seed", "stop", "stream", "stream_options", "suffix", "temperature",
    "top_p", "user",
})
COMPLETION_FORWARDED = frozenset({
    "echo", "frequency_penalty", "logit_bias", "logprobs", "max_tokens", "n", "presence_penalty", "prompt", "seed",
    "stop", "temperature", "top_p",
})


def _validate_completion(body) -> JSONResponse | None:
    """What we can't honour is refused, never ignored."""
    if not isinstance(body, dict):
        return error(400, "body must be a JSON object", "invalid_body")
    model = body.get("model")
    persona_id = graph_mod.persona_for(model)
    if persona_id is None:
        return error(404, f"model {model!r} not served; use one of /v1/models", "model_not_found", "model")
    prompt = body.get("prompt")

    def tokens(value) -> bool:
        return isinstance(value, list) and bool(value) and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in value)

    # The pinned four forms, all honoured by the tested backend (measured 2026-09-16,
    # S13h): a string, strings, a token array, token arrays.
    if not (isinstance(prompt, str)
            or (isinstance(prompt, list) and prompt and (all(isinstance(p, str) for p in prompt)
                                                         or tokens(prompt)
                                                         or all(tokens(p) for p in prompt)))):
        return error(400, "prompt must be a string, an array of strings, an array of token ids, "
                          "or an array of token-id arrays", "invalid_prompt", "prompt")
    n = body.get("n")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 128):
        return error(400, "n must be an integer between 1 and 128", "invalid_value", "n")
    temperature = body.get("temperature")
    if n is not None and n > 1 and isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and temperature == 0:
        # vLLM: "n must be 1 when using greedy sampling" (measured 2026-09-16). Several
        # choices need sampling; refused by name before the backend's opaque 400.
        return error(400, "n above 1 needs sampling: temperature must be above 0", "invalid_value", "n")
    best_of = body.get("best_of")
    if best_of is not None:
        if isinstance(best_of, bool) or not isinstance(best_of, int) or not 1 <= best_of <= 20:
            return error(400, "best_of must be an integer between 1 and 20", "invalid_value", "best_of")
        if best_of < (n or 1):
            return error(400, "best_of must be at least n", "invalid_value", "best_of")
        if best_of > (n or 1):
            # The backend accepts it, but whether it really keeps the best of the
            # extra candidates isn't observable from its answer: refused, not trusted.
            return error(400, "best_of above n is not supported", "unsupported_parameter", "best_of")
    if body.get("echo") is not None and not isinstance(body["echo"], bool):
        return error(400, "echo must be a boolean", "invalid_value", "echo")
    logprobs = body.get("logprobs")
    if logprobs is not None and (isinstance(logprobs, bool) or not isinstance(logprobs, int) or not 0 <= logprobs <= 5):
        return error(400, "logprobs must be an integer between 0 and 5", "invalid_value", "logprobs")
    if body.get("suffix") is not None:
        # The tested backend itself refused it: "suffix is not currently supported".
        return error(400, "'suffix' is not supported", "unsupported_parameter", "suffix")
    value = body.get("max_tokens")
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        return error(400, "max_tokens must be an integer of at least 1", "invalid_value", "max_tokens")
    # The pinned CreateCompletionRequest types and ranges, checked before the
    # model sees them. bool is never a number or an integer here.
    for param, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                             ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
        value = body.get(param)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            return error(400, f"{param} must be a number between {low} and {high}", "invalid_value", param)
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        return error(400, "seed must be an integer", "invalid_value", "seed")
    stop = body.get("stop")
    if stop is not None and not (isinstance(stop, str)
                                 or (isinstance(stop, list) and 1 <= len(stop) <= 4
                                     and all(isinstance(x, str) for x in stop))):
        return error(400, "stop must be a string or an array of up to 4 strings", "invalid_value", "stop")
    options = body.get("stream_options")
    if options is not None:
        if not isinstance(options, dict) or set(options) - {"include_usage", "include_obfuscation"} or any(
                v is not None and not isinstance(v, bool) for v in options.values()):
            return error(400, "stream_options must be an object with boolean include_usage", "invalid_value", "stream_options")
        if body.get("stream") is not True:
            return error(400, "stream_options needs stream: true", "invalid_value", "stream_options")
    if body.get("user") is not None and not isinstance(body["user"], str):
        return error(400, "user must be a string", "invalid_value", "user")
    bias = body.get("logit_bias")
    if bias is not None and not (isinstance(bias, dict) and all(
            isinstance(k, str) and not isinstance(v, bool) and isinstance(v, int) for k, v in bias.items())):
        return error(400, "logit_bias must be an object mapping tokens to integers", "invalid_value", "logit_bias")
    if "stream" in body and not isinstance(body["stream"], bool):
        return error(400, "stream must be a boolean", "invalid_value", "stream")
    unknown = sorted(k for k in body if k not in COMPLETION_SPEC_PARAMS)
    if unknown:
        return error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unsupported_parameter", unknown[0])
    return None


# The pinned CompletionUsage and its two breakdowns. Closed-default: nothing
# outside these names is published, and bool is not an integer here.
COMPLETION_USAGE = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})
COMPLETION_USAGE_DETAILS = {
    "completion_tokens_details": frozenset({"accepted_prediction_tokens", "audio_tokens", "reasoning_tokens",
                                            "rejected_prediction_tokens", "text_tokens"}),
    "prompt_tokens_details": frozenset({"audio_tokens", "cache_write_tokens", "cached_tokens",
                                        "image_tokens", "text_tokens"}),
}
FINISH_REASONS = frozenset({"stop", "length", "content_filter"})


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamShapeError(f"{value!r} is not a number")
    return value


def _completion_logprobs(logprobs, *, echo: bool = False) -> dict | None:
    """The pinned logprobs object, rebuilt field by field: nothing else crosses."""
    if logprobs is None:
        return None
    if not isinstance(logprobs, dict):
        raise UpstreamShapeError("choice logprobs is neither null nor an object")
    out: dict = {}
    if "text_offset" in logprobs:
        out["text_offset"] = [_count(v) for v in _list(logprobs["text_offset"], "text_offset")]
    if "token_logprobs" in logprobs:
        # With echo the first prompt token has nothing before it, so it has no logprob: vLLM
        # and OpenAI's legacy API both send null there, which the spec text doesn't allow.
        # That one position passes as null when echo is on (never an invented number); nowhere else
        # and never without echo (#208).
        out["token_logprobs"] = [None if echo and i == 0 and v is None else _number(v)
                                 for i, v in enumerate(_list(logprobs["token_logprobs"], "token_logprobs"))]
    if "tokens" in logprobs:
        tokens = _list(logprobs["tokens"], "tokens")
        if not all(isinstance(t, str) for t in tokens):
            raise UpstreamShapeError("logprobs tokens are not strings")
        out["tokens"] = tokens
    if "top_logprobs" in logprobs:
        top = _list(logprobs["top_logprobs"], "top_logprobs")
        rebuilt = []
        for i, entry in enumerate(top):
            if echo and i == 0 and entry is None:    # the echoed first token, as above
                rebuilt.append(None)
                continue
            if not isinstance(entry, dict) or not all(isinstance(k, str) for k in entry):
                raise UpstreamShapeError("a top_logprobs entry is not a string-keyed object")
            rebuilt.append({k: _number(v) for k, v in entry.items()})
        out["top_logprobs"] = rebuilt
    return out


def _list(value, name: str) -> list:
    if not isinstance(value, list):
        raise UpstreamShapeError(f"logprobs {name} is not an array")
    return value


class UpstreamShapeError(RuntimeError):
    """The upstream answered with something the pinned schema can't describe."""


def _count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpstreamShapeError(f"token count {value!r} is not an integer")
    return value


def _completion_usage(usage) -> dict | None:
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise UpstreamShapeError("usage is not an object")
    if not COMPLETION_USAGE <= usage.keys():
        raise UpstreamShapeError("usage is missing a required count")
    out: dict = {name: _count(usage[name]) for name in COMPLETION_USAGE}
    for name, allowed in COMPLETION_USAGE_DETAILS.items():
        details = usage.get(name)
        if details is None:
            continue
        if not isinstance(details, dict):
            raise UpstreamShapeError(f"{name} is not an object")
        out[name] = {k: _count(v) for k, v in details.items() if k in allowed}
    return out


def _completion_payload(data, cid: str, model: str, created: int, *, stream: bool = False, echo: bool = False) -> dict:
    """The pinned CreateCompletionResponse, rebuilt field by field. Anything the
    schema doesn't name is dropped; anything it can't describe fails closed.
    A null finish_reason is the streamed-chunk exception only."""
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
        raise UpstreamShapeError("no choices array")
    choices = []
    for i, choice in enumerate(data["choices"]):
        if not isinstance(choice, dict):
            raise UpstreamShapeError("a choice is not an object")
        index = choice.get("index", i)
        if isinstance(index, bool) or not isinstance(index, int):
            raise UpstreamShapeError(f"choice index {index!r} is not an integer")
        text = choice.get("text")
        if stream and "text" not in choice and choice.get("finish_reason") is not None:
            text = ""   # LiteLLM's streamed finish chunk has no text key at all (live 2026-09-16)
        if not isinstance(text, str):   # an empty string is valid; null, 0 and false are not
            raise UpstreamShapeError(f"choice text {text!r} is not a string")
        finish = choice.get("finish_reason")
        if finish is None and not stream:
            raise UpstreamShapeError("finish_reason is null outside a streamed chunk")
        if finish is not None and finish not in FINISH_REASONS:
            raise UpstreamShapeError(f"finish_reason {finish!r} is not one of {sorted(FINISH_REASONS)}")
        choices.append({"index": index, "text": text,
                        "logprobs": _completion_logprobs(choice.get("logprobs"), echo=echo), "finish_reason": finish})
    payload = {"id": cid, "object": "text_completion", "created": created, "model": model, "choices": choices}
    usage = _completion_usage(data.get("usage"))
    if usage is not None:
        payload["usage"] = usage
    if isinstance(data.get("system_fingerprint"), str):
        payload["system_fingerprint"] = opaque_fingerprint(data["system_fingerprint"])
    return payload


def _completion_shape_error(deps, trace, exc: UpstreamShapeError, headers: dict) -> JSONResponse:
    trace.set(upstream_shape_error=str(exc)[:300], result_status=Outcome.failed.value)
    deps.traces.write(trace)
    return JSONResponse({"error": {"message": "The model backend answered with an unusable completion.",
                                   "type": "server_error", "param": None, "code": "upstream_error"}},
                        status_code=502, headers=headers)


def register(app: FastAPI, deps: CompletionsDeps) -> None:
    @app.post("/v1/completions")
    async def completions(request: Request):
        """Legacy Completions: the prompt reaches the model as
        written. No persona, no base layer, no router, and no extension fields
        in the body; the trace id is the x-chord-trace-id header."""
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return error(400, "body is not JSON", "invalid_json")
        bad = _validate_completion(body)
        if bad:
            return bad
        assert isinstance(body, dict)
        persona_id = graph_mod.persona_for(body["model"])
        assert persona_id is not None
        trace = Trace(persona_id=persona_id, model_id_requested=body["model"])
        trace.set(endpoint="completions", params=sorted(k for k in body if k not in {"model", "prompt"}))
        headers = {"x-chord-trace-id": trace.trace_id, "x-request-id": trace.trace_id}
        stream = bool(body.get("stream"))
        created, cid = int(time.time()), f"cmpl-{trace.trace_id}"
        upstream_body = {k: v for k, v in body.items() if k in COMPLETION_FORWARDED}
        upstream_body["model"] = deps.settings.persona_model
        include_usage = stream and (body.get("stream_options") or {}).get("include_usage") is True
        if stream:
            # ALWAYS ask the backend for usage, whatever the client asked for. The
            # backend only emits its terminal chunk — the one carrying finish_reason —
            # alongside the usage chunk, so requesting usage only when the client wants
            # it silently cost every other client the reason the stream ended
            # (S13h: 0/6 streams carried a terminal finish_reason without
            # include_usage, 6/6 with it, same prompt). What the client asked for still
            # decides whether the usage FRAME is forwarded, below; this only decides
            # what we ask upstream.
            upstream_body["stream_options"] = {"include_usage": True}
        trace.set(persona_model=deps.settings.persona_model)

        if not stream:
            try:
                data, deployment = await deps.upstream.complete_text(upstream_body)
            except UpstreamError as exc:
                trace.set(upstream_error=str(exc), upstream_error_body=exc.body[:8000])
                deps.traces.write(trace)
                return JSONResponse(_upstream_error_body(exc, body), status_code=_upstream_http_status(exc), headers=headers)
            except httpx.HTTPError as exc:
                return _unreachable(deps, trace, exc, headers)
            try:
                payload = _completion_payload(data, cid, body["model"], created, echo=body.get("echo") is True)
            except UpstreamShapeError as exc:
                trace.set(persona_deployment=deployment)
                return _completion_shape_error(deps, trace, exc, headers)
            trace.set(persona_deployment=deployment, result_status=Outcome.completed.value)
            deps.traces.write(trace)
            return JSONResponse(payload, headers=headers)

        try:
            chunks = deps.upstream.stream_text(upstream_body)
            first = await anext(chunks)
        except UpstreamError as exc:
            trace.set(upstream_error=str(exc), upstream_error_body=exc.body[:8000])
            deps.traces.write(trace)
            return JSONResponse(_upstream_error_body(exc, body), status_code=_upstream_http_status(exc), headers=headers)
        except httpx.HTTPError as exc:
            return _unreachable(deps, trace, exc, headers)
        trace.set(persona_deployment=first[1])

        async def sse() -> AsyncIterator[str]:
            failed: str | bool | None = None
            stream_error = False
            usage_frame = None
            try:
                async for chunk, _ in chunks:
                    # The backend's usage chunk carries a text-less pseudo-choice. With
                    # include_usage it becomes the last frame, choices: [] (the spec's
                    # usage chunk); without it, it is not sent (S12 on Chat, S13h here).
                    # Earlier frames carry no usage key: the stream_options text says
                    # null, but the pinned CreateCompletionResponse doesn't allow null.
                    try:
                        if isinstance(chunk, dict) and chunk.get("usage") and all(
                                isinstance(c, dict) and "text" not in c for c in chunk.get("choices") or []):
                            usage = _completion_usage(chunk.get("usage"))
                            if include_usage and usage is not None:
                                usage_frame = {"id": cid, "object": "text_completion", "created": created,
                                               "model": body["model"], "choices": [], "usage": usage}
                            continue
                        payload = _completion_payload(chunk, cid, body["model"], created, stream=True, echo=body.get("echo") is True)
                        payload.pop("usage", None)
                    except UpstreamShapeError as exc:
                        # After headers: nothing unusable is published. The turn ends
                        # with a stream error event, never [DONE], which would read as a
                        # finished reply (as Chat does).
                        failed = str(exc)[:300]
                        break
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:
                # review 2026-09-24 B13: a transport drop or a malformed SSE line after
                # headers ends as a shape failure does -- an in-band error event, never
                # [DONE], and a failed trace -- as Chat's _emit does.
                trace.set(stream_error=repr(exc)[:500])
                stream_error = True
            finally:
                trace.set(upstream_shape_error=failed) if failed else None
                failed = failed or stream_error
                trace.set(result_status=(Outcome.failed if failed else Outcome.completed).value)
                deps.traces.write(trace)
            if failed:
                yield "data: " + json.dumps({"error": {"message": "the response failed while streaming", "type": "server_error",
                                                       "param": None, "code": "stream_failed"}}) + "\n\n"
                return
            if usage_frame is not None:
                yield f"data: {json.dumps(usage_frame)}\n\n"
            yield "data: [DONE]\n\n"

        return ClosingStreamingResponse(sse(), media_type="text/event-stream", headers=headers)
