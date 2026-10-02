"""`POST /v1/moderations` — served, spec-shaped, and it decides nothing.

The moderation route currently has the spec's shape but no classifier. A real
guardrail requires its own implementation and gate; until then this route must
not be treated as an input safety verdict.

**Nothing here classifies anything.** Every request answers `flagged: false`, every
category false, every score 0.0. That is not a judgement about the input; it is the
absence of one. A caller who reads a `false` from this route as "checked and clean"
has been misled, which is exactly why this docstring and the model id
below all say so in the same words.

A safety-shaped no-op is more dangerous to forget than a performance-shaped one
(`prediction` is refused: its token counts are observable, so a 200 that never
fills them is a false success). Three things make forgetting harder:

  * the `model` we return is **ours**, never `omni-moderation-*`. #121: "Return our
    real model name and policy, never impersonate omni-moderation." A caller that
    pins a real moderation model gets a 404 instead of a comfortable lie.
  * `MODERATION_DECIDES_NOTHING` is imported by the test that pins this behaviour,
    so deleting the flag breaks the suite rather than quietly arming the route.
  * the real design lives in #121 and is gated on external exposure, which is the
    condition under which this file must stop being a no-op.

The 13 categories are written out rather than derived at runtime so the service
never has to read the spec to answer, and `test_moderations.py` fails if the pinned
spec's category set ever differs from this tuple.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import graph as graph_mod
from .trace import Trace

# True while this route classifies nothing. The test suite imports it; flipping it
# without building the guard is a failing test, not a silent change in meaning.
MODERATION_DECIDES_NOTHING = True

# CreateModerationResponse.results[].categories, from the pinned spec.
CATEGORIES = (
    "harassment", "harassment/threatening", "hate", "hate/threatening",
    "illicit", "illicit/violent", "self-harm", "self-harm/instructions",
    "self-harm/intent", "sexual", "sexual/minors", "violence", "violence/graphic",
)

MAX_INPUTS = 100


def _error(status: int, message: str, code: str, param: str | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error",
                                   "param": param, "code": code}}, status_code=status)


def _texts(value) -> list[str] | None:
    """The spec's three input forms, flattened to the strings we would classify.

    A string, an array of strings, or an array of multi-modal parts. We keep the
    COUNT because the response must carry one result per input, and we do not keep
    the content: nothing reads it, and storing what we were asked to moderate would
    be the one genuinely surprising thing a no-op could do.
    """
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        return None
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            out.append(item["text"])
        elif isinstance(item, dict) and item.get("type") == "image_url":
            out.append("")          # counted, never inspected
        else:
            return None
    return out


def _result() -> dict:
    """One spec-shaped result that says nothing was decided."""
    return {
        "flagged": False,
        "categories": {c: False for c in CATEGORIES},
        "category_scores": {c: 0.0 for c in CATEGORIES},
        "category_applied_input_types": {c: ["text"] for c in CATEGORIES},
    }


def register(app: FastAPI, deps) -> None:
    @app.post("/v1/moderations")
    async def create_moderation(request: Request):
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object", "invalid_json")
        unknown = sorted(set(body) - {"input", "model"})
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}",
                          "unknown_parameter", unknown[0])

        # `model` is optional in the spec. If given it must be one we serve: naming a
        # real moderation model here would be a request to impersonate one (#121).
        model = body.get("model")
        if model is not None and graph_mod.persona_for(model) is None:
            return _error(404, f"model {model!r} not served; use one of /v1/models",
                          "model_not_found", "model")

        if "input" not in body:
            return _error(400, "input is required", "missing_required_parameter", "input")
        texts = _texts(body["input"])
        if texts is None:
            return _error(400, "input must be a string, an array of strings, or an array of content parts",
                          "invalid_value", "input")
        if len(texts) > MAX_INPUTS:
            return _error(400, f"input accepts at most {MAX_INPUTS} items", "invalid_value", "input")

        trace = Trace(persona_id=graph_mod.persona_for(model) if model else None, model_id_requested=model)
        trace.set(operation="POST /v1/moderations", inputs=len(texts),
                  decided=not MODERATION_DECIDES_NOTHING)
        deps.traces.write(trace)
        return JSONResponse({
            "id": f"modr-{trace.trace_id}",
            "model": graph_mod.MODEL_ID,
            "results": [_result() for _ in texts],
        }, headers={"x-request-id": trace.trace_id, "x-chord-trace-id": trace.trace_id})
