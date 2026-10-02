"""`POST /v1/embeddings` — a thin door onto the fleet's own embedding service.

The scoreboard read `Embeddings 0/1` for weeks while this answered 404. The model
was already running: the embedding models were moved out of the Musubi container
onto the host so both services could reach them, and the deployment description of that
work is the whole of it — *"It was just shifting from within the docker to bare
metal and that was all. Every other thing that was created is self created beyond
that."*

**This file got smaller on 2026-09-21 and that is the point.** It was written
against TEI **1.2.0**, which accepts `encoding_format: "base64"` and returns
floats anyway — an option honoured in the signature and not in the behaviour. So
Chord built the OpenAI envelope, computed float32 base64 itself, and always
requested `float` from upstream.

The operator upgraded the service to **1.9.4** mid-build. base64 landed in TEI 1.5, so
every one of those compensations became dead weight and was deleted rather than
certified. Less of our code between a caller and the model.

What remains is what TEI does not do for us:

  * refuse `dimensions` — it never reaches the model, and honouring it in the
    signature while ignoring it in the response is the exact defect we just
    removed from our own side.
  * refuse an unknown `model` — this deployment serves one, and echoing back a
    name we did not run would claim provenance the response does not have.
  * refuse empty input before a round trip — `data: []` with a 200 gives a caller
    no way to tell an empty request from an empty answer.
  * refuse a mismatched batch — `index` is how a caller maps vectors back to
    inputs, and if the pairing is unknowable the response is not serveable,
    whoever assembled it.

Where the model lives stays a deployment decision: `EMBEDDINGS_BASE_URL` resolves
through `Settings.base_url_for` like every other upstream, and the shared ingress
authenticates with Basic rather than the LiteLLM Bearer, so it gets its own
client.
"""

from __future__ import annotations

import base64
import binascii
import math
import struct

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .models_api import served_models
from .upstream import UpstreamError

# CreateEmbeddingRequest, from the pinned spec. `input` and `model` are the
# required pair; the rest are optional and pass through.
_ALLOWED = {"input", "model", "encoding_format", "dimensions", "user"}

_TOKEN_ARRAYS_REFUSED = ("token arrays are not supported: token ids are specific to the "
                         "tokenizer that produced them and cannot be mapped to this model's; "
                         "send the text as a string or an array of strings")


def _error(status: int, message: str, code: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error",
                           "param": param, "code": code}},
    )


def _input_items(value: object) -> tuple[list[str] | None, str | None]:
    """Normalize one OpenAI input into batch items and return any validation error.

    A blank string and an empty list are both refused here rather than sent
    upstream. They are not errors the backend is obliged to describe, and a
    caller who gets back `data: []` with a 200 has no way to tell an empty
    request from an empty answer.

    Token arrays are refused by name. Token ids mean something only to the
    tokenizer that produced them, and callers produce them with theirs:
    LangChain's OpenAIEmbeddings sends cl100k ids by default. They used to be
    decoded with the embedding model's own tokenizer (TEI `/decode`, bge-m3),
    which turned those ids into unrelated text, embedded it and answered 200.
    Ids from an unknown tokenizer cannot be mapped back to text, so the honest
    answer is a 400 that says to send strings (review 2026-09-24 B1)."""
    if isinstance(value, str):
        return ([value], None) if value else (None, "input must not be empty")
    if isinstance(value, list):
        if not value:
            return None, "input must not be an empty array"
        # Token arrays are classified before the size limit, so a long one
        # gets the token-array refusal, not a generic size error
        # (Copilot on #331, review 2026-09-24 B1).
        if (all(isinstance(v, int) and not isinstance(v, bool) for v in value)
                or all(isinstance(v, list) for v in value)):
            return None, _TOKEN_ARRAYS_REFUSED
        if len(value) > 2048:
            return None, "input array must contain at most 2048 items"
        if all(isinstance(v, str) for v in value):
            if any(not v for v in value):
                return None, "input array must not contain empty strings"
            return list(value), None
        return None, "input array must contain only strings"
    return None, "input must be a string or an array of strings"


def _embedding_width(value: object, fmt: str) -> int | None:
    if fmt == "float":
        if not isinstance(value, list) or not value:
            return None
        if not all(not isinstance(v, bool) and isinstance(v, (int, float))
                   and math.isfinite(float(v)) for v in value):
            return None
        return len(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return None
    if not raw or len(raw) % 4:
        return None
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    return len(values) if all(math.isfinite(v) for v in values) else None


def _invalid_response(payload: object, count: int, fmt: str) -> str | None:
    if not isinstance(payload, dict) or payload.get("object") != "list":
        return "embeddings upstream returned an invalid envelope"
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != count:
        return "embeddings upstream returned a mismatched batch"
    indices: set[int] = set()
    widths: set[int] = set()
    for item in data:
        if not isinstance(item, dict) or item.get("object") != "embedding":
            return "embeddings upstream returned an invalid data item"
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            return "embeddings upstream returned an invalid index"
        indices.add(index)
        width = _embedding_width(item.get("embedding"), fmt)
        if width is None:
            return "embeddings upstream returned an invalid vector"
        widths.add(width)
    if indices != set(range(count)):
        return "embeddings upstream returned invalid batch indexes"
    if len(widths) != 1:
        return "embeddings upstream returned inconsistent vector widths"
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return "embeddings upstream returned invalid usage"
    prompt = usage.get("prompt_tokens")
    total = usage.get("total_tokens")
    if (isinstance(prompt, bool) or not isinstance(prompt, int) or prompt < 0
            or isinstance(total, bool) or not isinstance(total, int) or total < prompt):
        return "embeddings upstream returned invalid usage"
    return None


def register(app: FastAPI, deps) -> None:
    @app.post("/v1/embeddings")
    async def create_embedding(request: Request):
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object", "invalid_json")

        unknown = sorted(set(body) - _ALLOWED)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}",
                          "unknown_parameter", unknown[0])

        if "input" not in body:
            return _error(400, "you must provide an input parameter", "missing_required_parameter", "input")
        items, reason = _input_items(body["input"])
        if reason:
            return _error(400, reason, "invalid_value", "input")

        fmt = body.get("encoding_format", "float")
        if fmt not in ("float", "base64"):
            return _error(400, f"encoding_format {fmt!r} is not one of 'float' or 'base64'",
                          "invalid_value", "encoding_format")

        if "dimensions" in body:
            return _error(400, "dimensions is not supported by this deployment; "
                               "the model returns its native width",
                          "unsupported_value", "dimensions")

        if "model" not in body:
            return _error(400, "you must provide a model parameter",
                          "missing_required_parameter", "model")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return _error(400, "model must be a non-empty string", "invalid_value", "model")
        if model not in served_models(deps.settings):
            # With EMBEDDINGS_BASE_URL unset, /v1/models does not list the
            # embeddings model and retrieving it is a 404, but this door used
            # to accept it and send it to the persona backend. One answer for
            # "is this model served", so the same 404 (review 2026-09-24 B19).
            return _error(404, f"model {model!r} not served; use one of /v1/models",
                          "model_not_found", "model")
        if model != deps.settings.embeddings_model:
            return _error(404, f"model {model!r} is not served; this deployment serves "
                               f"{deps.settings.embeddings_model!r}",
                          "model_not_found", "model")

        try:
            assert items is not None
            texts = list(items)
            # Everything the caller sent that we validated, with `input`
            # normalised to a list and `model` resolved. Not a rebuilt subset:
            # `user` went missing that way once already.
            upstream_request = {**body, "input": texts, "model": model}
            payload, _deployment = await deps.upstream.embed(upstream_request)
        except UpstreamError as exc:
            # A backend refusing OUR credentials is our 502, never a passed-
            # through 401/403: the caller's SDK reads a 401 as "your Chord key
            # is wrong" and blames the wrong party for a server-side config
            # problem (review 2026-09-22).
            status = 502 if exc.status in (401, 403) else (exc.status if 400 <= exc.status < 600 else 502)
            return JSONResponse(
                status_code=status,
                content={"error": {"message": "embeddings upstream rejected the request",
                                   "type": "server_error", "param": None,
                                   "code": "upstream_error"}},
            )
        except httpx.HTTPError as exc:
            # A backend that is down is a 502 in Chord's unreachable
            # vocabulary, not a 500 internal_error through the generic handler:
            # the identical condition on /v1/completions and /v1/chat/completions
            # answers 502 upstream_unreachable, and a 500 here implies a chord
            # bug while polluting the unhandled-exception trace stream (review
            # 2026-09-22).
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"embeddings upstream unreachable ({type(exc).__name__})",
                                   "type": "server_error", "param": None,
                                   "code": "upstream_unreachable"}},
            )

        if reason := _invalid_response(payload, len(texts), fmt):
            return JSONResponse(
                status_code=502,
                content={"error": {"message": reason,
                                   "type": "server_error", "param": None, "code": "upstream_error"}},
            )

        # The one field we do NOT pass through. Live TEI answers
        # `model: "BAAI/bge-m3"` for a request naming our configured model, and
        # returning that verbatim would break Chord's own namespace three ways:
        # `/v1/models` does not list it, this route 404s it as unserved, and a
        # caller cannot feed our response back to us. Refusing an id on the way
        # in while emitting it on the way out is incoherent, so the alias the
        # caller asked for — and the only one this deployment serves — is what
        # comes back. A test found this by making the fake answer as the live
        # service does instead of echoing the alias.
        return JSONResponse(status_code=200, content={**payload, "model": model})
