"""OpenAI Images API backed by an operator-configured ComfyUI workflow."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from . import artifact_links
from . import graph as graph_mod
from .artifacts import ArtifactStore, scrub_png
from .comfy_workflow import ComfyImageBackend
from .config import Settings
from .http_transport import BODY_SLACK, read_body_within_cap
from .render_admission import admission_for
from .trace import Trace, TraceSink


IMAGE_PARAM_VALUES = {
    "n": {1, None},
    "size": {"256x256", "512x512", "1024x1024", "auto", None},
    "response_format": {"b64_json", "url", None},
    "output_format": {"png", None},
    "quality": {"auto", None},
    "background": {"auto", None},
    "moderation": {"auto", None},
}
IMAGE_FREE_PARAMS = {"model", "prompt", "user"}
RENDER_SIDE = 1024
MAX_EDIT_BYTES = 25 * 1024 * 1024
MAX_VARIATION_BYTES = 4 * 1024 * 1024
VARIATION_SIDES = {256, 512, 1024}
logger = logging.getLogger(__name__)


class ImagesDeps(Protocol):
    settings: Settings
    artifacts: ArtifactStore
    traces: TraceSink
    image_backend: ComfyImageBackend | None


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


def _square_side(size: object) -> int:
    return RENDER_SIDE if size in (None, "auto") else int(str(size).split("x")[0])


def _misfiled_text_field(form, keys: tuple[str, ...]) -> str | None:
    """The first scalar field a caller sent as a file part, or None.

    A multipart form value is `str | UploadFile`, so `n` or `response_format` can
    arrive as a file. Used as text, each one failed differently and none of them
    said so: `int(UploadFile)` fell into the ValueError branch and answered "n must
    be between 1 and 10", and a misfiled `response_format` is truthy but never
    equals "url", so it was silently ignored and the caller got b64_json. Rejecting
    the shape names the actual mistake. Only `image` is legitimately a file here.

    Tests for "present and not text" rather than `isinstance(..., UploadFile)`:
    `request.form()` yields `starlette.datastructures.UploadFile`, and
    `fastapi.UploadFile` is a SUBCLASS of it, so the obvious isinstance check is
    False for every real upload. Caught by this function's own test returning 200.
    """
    for key in keys:
        value = form.get(key)
        if value is not None and not isinstance(value, str):
            return key
    return None


def _text(value: object) -> str | None:
    """A form or JSON value narrowed to text; anything else reads as absent."""
    return value if isinstance(value, str) else None


def _can_serve_urls(settings) -> bool:
    return bool(settings.public_artifact_base and settings.artifact_signer_key)


def _url_format_error(response_format: object, settings) -> JSONResponse | None:
    """`url` is a promise this deployment either keeps or refuses BY NAME.

    Without a public artifact route there is no URL to hand out; the old code
    silently substituted a data URI inside the `url` field -- a spec-shaped lie
    the caller could only discover by parsing it (decisions #2 and #4).
    Refused, never
    substituted; the message says what to send instead."""
    if response_format == "url" and not _can_serve_urls(settings):
        return _error(400, "response_format 'url' needs a public artifact route, which this "
                           "deployment does not have; omit response_format for b64_json",
                      "unsupported_value", "response_format")
    return None


def _count_error(value, *, min_n: int, max_n: int) -> JSONResponse | None:
    """Strict `n` for the image doors: must be `None`, an int in `[min_n, max_n]`,
    or a string that parses to one. `bool` and `float` are rejected even though
    Python's `==` would say they equal an integer -- `True` is not a count, and
    `1.0` is not a count we want to silently round (review 2026-09-23).
    Shared by the edits and variations doors so those multipart/JSON paths answer
    the same way for the same input."""
    if value is None:
        return None
    if isinstance(value, bool):
        return _error(400, f"n must be an integer between {min_n} and {max_n}, "
                           f"got {value!r}", "invalid_value", "n")
    if isinstance(value, int):
        if min_n <= value <= max_n:
            return None
        return _error(400, f"n must be between {min_n} and {max_n}", "invalid_value", "n")
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return _error(400, f"n must be an integer between {min_n} and {max_n}, "
                               f"got {value!r}", "invalid_value", "n")
        if not isinstance(parsed, int) or isinstance(parsed, bool):
            return _error(400, f"n must be an integer between {min_n} and {max_n}, "
                               f"got {value!r}", "invalid_value", "n")
        if min_n <= parsed <= max_n:
            return None
        return _error(400, f"n must be between {min_n} and {max_n}", "invalid_value", "n")
    return _error(400, f"n must be an integer between {min_n} and {max_n}, "
                       f"got {value!r}", "invalid_value", "n")


def _effective_format(response_format: str | None, settings) -> str:
    """The format an omitted response_format resolves to: OpenAI's default
    (`url`) where this deployment can serve it, `b64_json` where it cannot.
    Deployment-shaped by necessity -- a static `url` default would 400 every
    omitted request on a deployment with no public route."""
    if response_format in (None, ""):
        return "url" if _can_serve_urls(settings) else "b64_json"
    return response_format


def _image_item(png: bytes, response_format: str | None, deps: ImagesDeps, artifact=None) -> dict:
    """One ImagesResponse entry. `url` uses the same signed artifact link as chat.

    The doors refuse `response_format: "url"` where no public artifact route is
    configured (_url_format_error), so the url branch here always has a base to
    sign against; the data-URI line below is defense for a caller that bypassed
    the doors, never a policy -- a silent url->data substitution was exactly
    the spec-shaped lie this arrangement replaced (review 2026-09-22, decision
    #2). Signed links expire in an hour, which is the spec's lifetime for an
    image URL. The artifact is registered only on the path that returns its id;
    registering before that branch left one unreferenced artifact per request.
    And when the CALLER already registered these exact bytes -- the generations
    path holds the render's own descriptor, or the scaled derivative it just
    registered for the trace -- that artifact is reused: the store has no GC, so
    a second identical file is permanent waste (review 2026-09-22)."""
    if response_format != "url":
        return {"b64_json": base64.b64encode(png).decode()}
    settings = deps.settings
    base = settings.public_artifact_base
    if base and settings.artifact_signer_key:
        if artifact is None:
            artifact = deps.artifacts.register(png, "image/png")
        expires = int(time.time()) + min(3600, settings.artifact_url_ttl_s)
        sig = artifact_links.signature(settings.artifact_signer_key, artifact.id, expires)
        return {"url": f"{base.rstrip('/')}/v1/artifacts/{artifact.id}?expires={expires}&sig={sig}"}
    # The doors refuse `response_format: "url"` when no public route is
    # configured (_url_format_error) before reaching this function, so this
    # branch is unreachable in the supported flow. If a future caller bypasses
    # the doors, a silent url->data substitution is exactly the spec-shaped lie
    # this arrangement replaced (review 2026-09-22). Refuse explicitly rather
    # than substitute.
    raise RuntimeError(
        "images_api._image_item reached with response_format='url' but no "
        "public artifact route; the door should have refused this request"
    )


def _downscale_png(png: bytes, side: int) -> bytes:
    with Image.open(BytesIO(png)) as image:
        output = BytesIO()
        image.resize((side, side), Image.Resampling.LANCZOS).save(output, format="PNG")
    return output.getvalue()


def _validate_request(body: object) -> tuple[str | None, str | None, JSONResponse | None]:
    if not isinstance(body, dict):
        return None, None, _error(400, "body must be a JSON object", "invalid_body")
    model = body.get("model")
    if model is None:
        model = graph_mod.MODEL_ID
    persona_id = graph_mod.persona_for(model)
    if persona_id is None:
        return None, None, _error(
            404, f"model {model!r} not served; use one of /v1/models", "model_not_found", "model"
        )
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, None, _error(400, "prompt must be a non-empty string", "invalid_prompt", "prompt")
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError:
        return None, None, _error(400, "prompt must be valid UTF-8", "invalid_prompt", "prompt")
    for key, value in body.items():
        if key in IMAGE_FREE_PARAMS:
            continue
        allowed = IMAGE_PARAM_VALUES.get(key)
        if allowed is None or not any(value == candidate and type(value) is type(candidate) for candidate in allowed):
            return None, None, _error(
                400,
                f"{key}={value!r} is not supported: this model makes one 1024x1024 PNG, as b64_json or a url",
                "unsupported_parameter",
                key,
            )
    return persona_id, prompt.strip(), None


def register(app: FastAPI, deps: ImagesDeps, store=None) -> None:
    async def configured_comfy_generation(body: dict, persona_id: str, prompt: str) -> JSONResponse:
        backend = deps.image_backend
        if backend is None:
            return _error(503, "the configured image provider is unavailable",
                          "capability_unavailable", "model")
        trace = Trace(persona_id=persona_id, model_id_requested=body.get("model"))
        trace.set(endpoint="images.generations", image_backend="comfyui",
                  params=sorted(key for key in body if key not in {"model", "prompt"}))
        headers = {"x-chord-trace-id": trace.trace_id, "x-request-id": trace.trace_id}
        try:
            try:
                png, prompt_id = await backend.render(prompt)
                with Image.open(BytesIO(png)) as image:
                    image.verify()
                    if image.format != "PNG" or image.size != (RENDER_SIDE, RENDER_SIDE):
                        raise ValueError("image workflow must return a 1024x1024 PNG")
                # ComfyUI can embed the full graph, checkpoint names and
                # expanded prompt in PNG text chunks. Never deliver those.
                png, removed_chunks = scrub_png(png)
                trace.set(image_metadata_removed=removed_chunks)
                source = deps.artifacts.register(png, "image/png")
                trace.artifacts.append(source.model_dump())
                side = _square_side(body.get("size"))
                serving = source
                if side != RENDER_SIDE:
                    png = await asyncio.to_thread(_downscale_png, png, side)
                    serving = deps.artifacts.register(png, "image/png")
                    trace.artifacts.append({**serving.model_dump(), "derived_from": source.id,
                                            "size": f"{side}x{side}"})
                trace.set(result_status="completed", comfy_prompt_id=prompt_id,
                          delivered={"size": f"{side}x{side}",
                                     "sha256": hashlib.sha256(png).hexdigest()})
                data = [_image_item(png, _effective_format(body.get("response_format"), deps.settings),
                                    deps, artifact=serving)]
                return JSONResponse({"created": int(time.time()), "data": data}, headers=headers)
            except Exception as exc:
                # ComfyUI responses can include operator workflow and path
                # details. The public error and trace record only the type.
                trace.set(result_status="failed", image_backend_error=type(exc).__name__)
                return JSONResponse({"error": {
                    "message": "Image generation failed.", "type": "server_error",
                    "param": None, "code": "image_generation_failed",
                }}, status_code=502, headers={**headers, "x-should-retry": "false"})
        finally:
            deps.traces.write(trace)

    @app.post("/v1/images/generations")
    async def images_generations(request: Request):
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        persona_id, prompt, err = _validate_request(body)
        if err:
            return err
        if (bad := _url_format_error(body.get("response_format"), deps.settings)) is not None:
            return bad
        assert isinstance(body, dict) and persona_id is not None and prompt is not None
        if deps.settings.image_workflow is None:
            return _error(503, "no image provider is configured", "capability_unavailable", "model")
        return await admission_for(app).run_inline(
            "image generations", lambda: configured_comfy_generation(body, persona_id, prompt))

    files_dir: Path = deps.settings.data_dir / "files"

    def _edit_source(data: bytes, filename: str) -> tuple[bytes, str] | JSONResponse:
        if len(data) > MAX_EDIT_BYTES:
            return _error(413, "image must be at most 25 MB", "file_too_large", "image")
        if not data:
            return _error(400, "image is empty", "invalid_value", "image")
        try:
            with Image.open(BytesIO(data)) as image:
                image.verify()
        # DecompressionBombError inherits from Exception alone: without it here,
        # a 66-byte PNG claiming 196M pixels escapes as a 500 (review 2026-09-22).
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
            return _error(400, "image must be a readable image", "invalid_value", "image")
        return data, filename

    # Review 2026-09-24 B10: edits and variations wait on the video GPU lock,
    # so they are admitted against the video budget, BEFORE the upload is read.
    admission = admission_for(app)

    @app.post("/v1/images/edits")
    async def images_edits(request: Request):
        return await admission.run_inline("image edits", lambda: _images_edits(request))

    async def _images_edits(request: Request):
        """One source image and a prompt, rendered by the operator's edit workflow.

        The image is a multipart upload or a stored ``file_id``. A URL or a mask
        is refused: the workflow binding has no input for either.
        """
        content_type = request.headers.get("content-type", "")
        filename = "edit.png"
        data: bytes | None = None
        prompt = None
        model = None
        count = None
        response_format = None
        if content_type.startswith("multipart/form-data"):
            # Bound ingress before the form parse spools to disk without a
            # limit (2026-09-22, #3); same envelope as the
            # per-part check below.
            raw = await read_body_within_cap(request, MAX_EDIT_BYTES + BODY_SLACK)
            if raw is None:
                return _error(413, f"image must be at most {MAX_EDIT_BYTES // (1024 * 1024)} MB",
                              "file_too_large", "image")
            request._body = raw
            try:
                form = await request.form()
            except Exception:
                return _error(400, "the multipart body could not be parsed", "invalid_request")
            allowed = {"image", "prompt", "model", "n", "response_format"}
            unknown = sorted(set(form.keys()) - allowed)
            if unknown:
                return _error(400, f"{unknown[0]} is not supported", "unsupported_parameter", unknown[0])
            scalars = ("prompt", "model", "n", "response_format")
            misfiled = _misfiled_text_field(form, scalars)
            if misfiled:
                return _error(400, f"{misfiled} must be a text field, not a file", "invalid_value", misfiled)
            prompt, model, count, response_format = (_text(form.get(key)) for key in scalars)
            upload = form.get("image")
            if upload is None or isinstance(upload, str) or len(form.getlist("image")) != 1:
                return _error(400, "image must be one uploaded image", "invalid_value", "image")
            if upload.size is not None and upload.size > MAX_EDIT_BYTES:
                return _error(413, "image must be at most 25 MB", "file_too_large", "image")
            data = await upload.read(MAX_EDIT_BYTES + 1)
            filename = os.path.basename(upload.filename or "") or "edit.png"
        elif content_type.startswith("application/json"):
            try:
                body = await request.json()
            except (ValueError, RecursionError):
                return _error(400, "body is not JSON", "invalid_json")
            if not isinstance(body, dict):
                return _error(400, "body must be a JSON object", "invalid_json")
            allowed = {"images", "prompt", "model", "n", "response_format"}
            unknown = sorted(set(body) - allowed)
            if unknown:
                return _error(400, f"{unknown[0]} is not supported", "unsupported_parameter", unknown[0])
            prompt, model, count, response_format = (body.get(key) for key in ("prompt", "model", "n", "response_format"))
            images = body.get("images")
            if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
                return _error(400, "images must be one file_id", "invalid_value", "images")
            ref = images[0]
            if "image_url" in ref:
                return _error(400, "image_url is not supported", "unsupported_parameter", "image_url")
            file_id = ref.get("file_id")
            if set(ref) != {"file_id"} or not isinstance(file_id, str) or store is None:
                return _error(400, "images must be one file_id", "invalid_value", "images")
            from .files_api import StoredFileTooLarge, load_stored_file
            try:
                # Bound the load to the edit cap BEFORE the bytes are resident:
                # a stored file may be 512 MB while an edit source may not
                # exceed MAX_EDIT_BYTES, and _edit_source's check downstream
                # can only refuse what is already in memory (review 2026-09-22).
                loaded = load_stored_file(store, files_dir, file_id, max_bytes=MAX_EDIT_BYTES)
            except StoredFileTooLarge:
                return _error(413, f"image must be at most {MAX_EDIT_BYTES // (1024 * 1024)} MB",
                              "file_too_large", "image")
            if loaded is None:
                return _error(404, f"No such File object: {file_id}", "not_found", "file_id")
            file, data = loaded
            filename = file.get("filename") or "edit.png"
        else:
            return _error(400, "image edits are multipart or JSON with a file_id", "invalid_request")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error(400, "prompt must be a non-empty string", "invalid_prompt", "prompt")
        model = model or graph_mod.MODEL_ID
        if graph_mod.persona_for(model) is None:
            return _error(404, f"model {model!r} not served; use one of /v1/models", "model_not_found", "model")
        if (bad := _count_error(count, min_n=1, max_n=1)) is not None:
            return bad
        if response_format not in (None, "", "b64_json", "url"):
            return _error(400, "response_format must be b64_json or url", "unsupported_value", "response_format")
        if (bad := _url_format_error(response_format, deps.settings)) is not None:
            return bad
        checked = _edit_source(data or b"", filename)
        if isinstance(checked, JSONResponse):
            return checked
        data, filename = checked
        backend = getattr(deps, "edit_backend", None)
        if backend is None:
            return JSONResponse(
                {"error": {
                    "message": "image editing is not configured",
                    "type": "server_error",
                    "param": None,
                    "code": "backend_unavailable",
                }},
                status_code=503,
            )
        try:
            png = await backend.edit_image(data, filename, prompt.strip())
            # Scrubbed here, not only on registration: the b64_json door returns
            # these bytes directly, and ComfyUI's save node writes the whole
            # workflow into a tEXt chunk (review 2026-09-24 A4; T01 for edits).
            # Not a PNG we can parse fails closed, as a failed edit.
            png = scrub_png(png)[0] if png else png
        except Exception:
            logger.exception("image edit failed")
            png = b""
        if not png:
            return JSONResponse(
                {"error": {
                    "message": "the image could not be edited right now",
                    "type": "server_error",
                    "param": None,
                    "code": "image_edit_failed",
                }},
                status_code=502,
            )
        return JSONResponse({
            "created": int(time.time()),
            "data": [_image_item(png, _effective_format(response_format, deps.settings), deps)],
        })

    @app.post("/v1/images/variations")
    async def images_variations(request: Request):
        return await admission.run_inline("image variations", lambda: _images_variations(request))

    async def _images_variations(request: Request):
        """Several variations of one square PNG, from the operator's variation workflow.

        The spec has no prompt, and neither does Chord: any instruction lives in
        the operator's graph. Each result is one submit of that workflow with a
        fresh seed, so more than one needs a seed binding.
        """
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            return _error(400, "image variations are uploaded as multipart/form-data", "invalid_request")
        # Bound ingress before the form parse spools to disk (#3);
        # same envelope as the per-part check below.
        raw = await read_body_within_cap(request, MAX_VARIATION_BYTES + BODY_SLACK)
        if raw is None:
            return _error(400, "image must be a square PNG under 4 MB", "invalid_value", "image")
        request._body = raw
        try:
            form = await request.form()
        except Exception:
            return _error(400, "the multipart body could not be parsed", "invalid_request")
        allowed = {"image", "model", "n", "response_format", "size", "user"}
        unknown = sorted(set(form.keys()) - allowed)
        if unknown:
            return _error(400, f"{unknown[0]} is not supported", "unsupported_parameter", unknown[0])
        misfiled = _misfiled_text_field(form, ("model", "n", "response_format", "size", "user"))
        if misfiled:
            return _error(400, f"{misfiled} must be a text field, not a file", "invalid_value", misfiled)
        model = _text(form.get("model"))
        if model not in (None, "", "dall-e-2") and graph_mod.persona_for(model) is None:
            return _error(404, f"model {model!r} not served; use one of /v1/models", "model_not_found", "model")
        raw_n = _text(form.get("n"))
        if (bad := _count_error(raw_n, min_n=1, max_n=10)) is not None:
            return bad
        count = 1 if raw_n in (None, "") else int(raw_n)
        size = _text(form.get("size")) or "1024x1024"
        if size not in {f"{side}x{side}" for side in VARIATION_SIDES}:
            return _error(400, "size must be 256x256, 512x512, or 1024x1024", "invalid_value", "size")
        side = int(str(size).split("x", 1)[0])
        response_format = _text(form.get("response_format"))
        if response_format not in (None, "", "b64_json", "url"):
            return _error(400, "response_format must be b64_json or url", "unsupported_value", "response_format")
        if (bad := _url_format_error(response_format, deps.settings)) is not None:
            return bad
        upload = form.get("image")
        if upload is None or isinstance(upload, str) or len(form.getlist("image")) != 1:
            return _error(400, "image must be one square PNG", "invalid_value", "image")
        data = await upload.read(MAX_VARIATION_BYTES + 1)
        if len(data) > MAX_VARIATION_BYTES:
            return _error(400, "image must be a square PNG under 4 MB", "invalid_value", "image")
        try:
            with Image.open(BytesIO(data)) as incoming:
                image_format, dimensions = incoming.format, incoming.size
        # Same bomb, same reasoning as _edit_source: Pillow raises it from open,
        # and it is neither UnidentifiedImageError nor OSError.
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
            return _error(400, "image must be a square PNG under 4 MB", "invalid_value", "image")
        if image_format != "PNG" or dimensions[0] != dimensions[1]:
            return _error(400, "image must be a square PNG under 4 MB", "invalid_value", "image")
        backend = getattr(deps, "variation_backend", None)
        if backend is None:
            return JSONResponse(
                {"error": {
                    "message": "image variations are not configured",
                    "type": "server_error",
                    "param": None,
                    "code": "backend_unavailable",
                }},
                status_code=503,
            )
        limit = getattr(backend, "max_variations", 10)
        if count > limit:
            return _error(400, f"n must be at most {limit}: the configured variation workflow has no seed binding",
                          "invalid_value", "n")
        filename = os.path.basename(upload.filename or "") or "variation.png"
        try:
            pngs = await backend.vary_images(data, filename, count)
            pngs = [scrub_png(png)[0] for png in pngs]   # as edits (review 2026-09-24 A4)
        except Exception:
            logger.exception("image variation failed")
            pngs = []
        if len(pngs) != count:
            return JSONResponse(
                {"error": {
                    "message": "the image could not be varied right now",
                    "type": "server_error",
                    "param": None,
                    "code": "image_variation_failed",
                }},
                status_code=502,
            )
        # One thread hop for the whole batch: n=10 is ten LANCZOS resizes
        # (batch 4).
        rendered = await asyncio.to_thread(
            lambda: [_downscale_png(png, side) if side else png for png in pngs])
        return JSONResponse({
            "created": int(time.time()),
            "data": [_image_item(png, _effective_format(response_format, deps.settings), deps) for png in rendered],
        })
