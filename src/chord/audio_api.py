"""The Audio API (#139, Phase 4): POST /v1/audio/speech and
POST /v1/audio/transcriptions, non-streaming, in the pinned spec's shapes.

The chat path can use the same configured speech backends; these
are the standalone speech endpoints. Everything
the backend can't do is refused with a 400 naming the parameter, never let
through to a backend 500 (measured 2026-09-16 against the certified TTS backend: aac, an
unknown voice and any speed but 1.0 each return 500; mp3, opus, flac, wav and
pcm work, all labelled audio/mpeg, so the format is checked from the bytes).

Streaming speech and transcription stay on hold (#139). Translations are two
honest steps: the configured STT transcribes, then the chat model translates.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import graph as graph_mod
from .artifacts import looks_like_mp3 as _looks_like_mp3
from .http_transport import BODY_SLACK, read_body_within_cap
from .registry import PROMPTS_DIR
from .trace import Trace
from .upstream import UpstreamUnconfiguredError

MAX_INPUT = 4096
# OpenAI's speech-to-text guide caps uploads at 25 MB.  Starlette spools large
# multipart parts to disk, but calling read() without a bound copies the whole
# part back into this process.  Keep the wire limit and the memory limit the
# same, and read one extra byte so a missing/incorrect multipart size cannot
# bypass the check.
MAX_AUDIO_BYTES = 25 * 1024 * 1024



# response_format -> (content-type, does the body look like it)
SPEECH_FORMATS = {
    "mp3": ("audio/mpeg", _looks_like_mp3),
    "opus": ("audio/ogg", lambda b: b[:4] == b"OggS"),
    "flac": ("audio/flac", lambda b: b[:4] == b"fLaC"),
    "wav": ("audio/wav", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WAVE"),
    "pcm": ("audio/pcm", lambda b: len(b) % 2 == 0),   # raw 24 kHz 16-bit mono: no header to check
}
SPEECH_FIELDS = {"model", "input", "voice", "instructions", "response_format", "speed", "stream_format"}

# What an uploaded file is, from its bytes (the name and content-type are the
# caller's claims): format -> (mime sent upstream, does the body look like it)
TRANSCRIBE_FORMATS = {
    "wav": ("audio/wav", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WAVE"),
    "mp3": ("audio/mpeg", _looks_like_mp3),
    "flac": ("audio/flac", lambda b: b[:4] == b"fLaC"),
    "ogg": ("audio/ogg", lambda b: b[:4] == b"OggS"),
    # What browser recorders produce: Chrome's MediaRecorder writes webm (an EBML
    # header), Safari's writes mp4/m4a (an ISO BMFF `ftyp` box). The tested STT
    # backend decoded both (2026-09-24); they were refused here only.
    "webm": ("audio/webm", lambda b: b[:4] == b"\x1a\x45\xdf\xa3"),
    "m4a": ("audio/mp4", lambda b: b[4:8] == b"ftyp"),
}
TRANSCRIBE_FIELDS = {"file", "model", "language", "prompt", "response_format", "temperature"}
TRANSLATE_FIELDS = {"file", "model", "prompt", "response_format", "temperature"}
# Accepted by the spec, not by this backend: refused by name.
TRANSCRIBE_REFUSED = {"stream", "include", "timestamp_granularities", "chunking_strategy", "languages", "keywords",
                      "known_speaker_names", "known_speaker_references"}


def _error(status: int, message: str, code: str, param: str | None = None, kind: str = "invalid_request_error",
           headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "param": param, "code": code}},
                        status_code=status, headers=headers)


def _model_error(model) -> JSONResponse | None:
    if not isinstance(model, str) or not model:
        return _error(400, "model is required", "invalid_value", "model")
    if graph_mod.persona_for(model) is None:
        return _error(404, f"The model '{model}' does not exist", "model_not_found", "model")
    return None


def known_voices(settings) -> list[str]:
    """Voice identifiers explicitly configured for this installation."""
    return sorted(set(json.loads(settings.tts_voices).values()))


def register(app: FastAPI, deps) -> None:
    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return _error(400, "body is not JSON", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object", "invalid_json")
        unknown = sorted(set(body) - SPEECH_FIELDS)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        if (bad := _model_error(body.get("model"))) is not None:
            return bad
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            return _error(400, "input must be a non-empty string", "invalid_value", "input")
        if len(text) > MAX_INPUT:
            return _error(400, f"input must be at most {MAX_INPUT} characters", "string_above_max_length", "input")
        voices = known_voices(deps.settings)
        voice = body.get("voice")
        if isinstance(voice, dict):
            return _error(400, "custom voices are not supported", "unsupported_value", "voice")
        if voice not in voices:
            return _error(400, f"voice must be one of {voices}", "invalid_value", "voice")
        fmt = body.get("response_format") or "mp3"
        # isinstance first: SPEECH_FORMATS is a dict, `in` hashes, and a list
        # here was an unhashable-type 500 (2026-09-22, #5).
        if not isinstance(fmt, str) or fmt not in SPEECH_FORMATS:
            return _error(400, f"response_format must be one of {sorted(SPEECH_FORMATS)}", "unsupported_value", "response_format")
        speed = body.get("speed")
        if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed != 1):
            return _error(400, "speed other than 1.0 is not supported", "unsupported_value", "speed")
        if body.get("instructions"):
            return _error(400, "instructions are not supported", "unsupported_parameter", "instructions")
        if body.get("stream_format") not in (None, "audio"):
            return _error(400, "stream_format sse is not supported", "unsupported_value", "stream_format")

        persona = graph_mod.persona_for(body["model"])
        assert persona is not None       # _model_error above refused every unserved model
        trace = Trace(persona_id=persona, model_id_requested=body["model"])
        trace.set(operation="POST /v1/audio/speech")
        headers = {"x-chord-trace-id": trace.trace_id, "x-request-id": trace.trace_id}
        mime, looks_right = SPEECH_FORMATS[fmt]
        entry = {"model": deps.settings.tts_model, "voice": voice, "format": fmt, "input_chars": len(text)}
        try:
            with trace.timed("tts"):
                data, deployment = await deps.upstream.speak(text, voice, fmt, deps.settings.tts_model)
        except UpstreamUnconfiguredError as exc:
            # This deployment has no TTS address; refuse honestly with the 503
            # the upstream call raised, not the generic 502 backend failure.
            trace.tts = {**entry, "error": repr(exc)[:300]}
            deps.traces.write(trace)
            return _error(503, exc.body, "speech_not_configured", kind="server_error", headers=headers)
        except Exception as exc:  # noqa: BLE001 - any backend failure is ours, a 502
            trace.tts = {**entry, "error": repr(exc)[:300]}
            deps.traces.write(trace)
            return _error(502, "speech could not be generated", "speech_generation_failed", kind="server_error", headers=headers)
        if not data or not looks_right(data):
            trace.tts = {**entry, "deployment": deployment, "error": f"bytes are not {fmt}"}
            deps.traces.write(trace)
            return _error(502, "speech could not be generated", "speech_generation_failed", kind="server_error", headers=headers)
        trace.tts = {**entry, "deployment": deployment, "output_bytes": len(data)}
        deps.traces.write(trace)
        return Response(content=data, media_type=mime, headers=headers)

    async def heard(request: Request, operation: str, allowed: set, refused_names: set):
        """Parse, validate and transcribe an audio upload. Returns
        (transcript, response_format, headers, trace) or an error response."""
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            return _error(400, "the request must be multipart/form-data", "invalid_request")
        # Bound ingress BEFORE the form parse: Starlette spools file parts to
        # disk without any limit, so the 25 MB per-part check used to fire only
        # after a 50 GB body had already landed on the volume (#3).
        raw = await read_body_within_cap(request, MAX_AUDIO_BYTES + BODY_SLACK)
        if raw is None:
            return _error(413, f"file must be at most {MAX_AUDIO_BYTES // (1024 * 1024)} MB", "file_too_large", "file")
        request._body = raw
        try:
            form = await request.form()
        except Exception:  # noqa: BLE001 - a malformed body is the caller's
            return _error(400, "the multipart body could not be parsed", "invalid_request")
        base = {f[:-2] if f.endswith("[]") else f for f in form.keys()}
        stream_off = str(form.get("stream")).lower() in ("false", "")
        refused = [r for r in sorted(base & refused_names) if not (r == "stream" and stream_off)]
        if refused:
            return _error(400, f"{refused[0]} is not supported", "unsupported_parameter", refused[0])
        unknown = sorted(base - allowed - refused_names)
        if unknown:
            return _error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unknown_parameter", unknown[0])
        if (bad := _model_error(form.get("model"))) is not None:
            return bad
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            return _error(400, "file is required", "invalid_value", "file")
        if upload.size is not None and upload.size > MAX_AUDIO_BYTES:
            return _error(413, "file must be at most 25 MB", "file_too_large", "file")
        audio = await upload.read(MAX_AUDIO_BYTES + 1)
        if len(audio) > MAX_AUDIO_BYTES:
            return _error(413, "file must be at most 25 MB", "file_too_large", "file")
        fmt = next((name for name, (_, looks) in TRANSCRIBE_FORMATS.items() if audio and looks(audio)), None)
        if fmt is None:
            return _error(400, f"file must be audio in one of {sorted(TRANSCRIBE_FORMATS)}", "invalid_value", "file")
        response_format = form.get("response_format") or "json"
        if not isinstance(response_format, str) or response_format not in ("json", "text"):
            return _error(400, "response_format must be json or text", "unsupported_value", "response_format")
        temperature = form.get("temperature")
        if temperature is not None and temperature != "":
            # A multipart file part misfiled as temperature gets the same 400
            # float()'s TypeError produced (which once escaped as a bare 500
            # after the audio had already been read) -- named, not raised.
            if not isinstance(temperature, str):
                return _error(400, "temperature must be between 0 and 1", "invalid_value", "temperature")
            try:
                t = float(temperature)
            except ValueError:
                t = -1.0
            if not 0 <= t <= 1:
                return _error(400, "temperature must be between 0 and 1", "invalid_value", "temperature")
        language = form.get("language") if "language" in allowed else None
        if language is not None and (not isinstance(language, str) or not language.strip()):
            return _error(400, "language must be a nonblank string", "invalid_value", "language")

        model = form.get("model")
        assert isinstance(model, str)    # _model_error refused anything else
        persona = graph_mod.persona_for(model)
        assert persona is not None       # _model_error refused every unserved model
        trace = Trace(persona_id=persona, model_id_requested=model)
        trace.set(operation=operation)
        headers = {"x-chord-trace-id": trace.trace_id, "x-request-id": trace.trace_id}
        mime = TRANSCRIBE_FORMATS[fmt][0]
        # Prompt and temperature are accepted and traced as declared no-ops.
        # Language is a real hint passed to the OpenAI-compatible STT backend.
        trace.set(stt_hints={k: form.get(k) is not None for k in ("prompt", "temperature", "language") if k in allowed})
        try:
            with trace.timed("stt"):
                if language is None:
                    text, deployment = await deps.upstream.transcribe(audio, fmt, mime, deps.settings.stt_model)
                else:
                    text, deployment = await deps.upstream.transcribe(
                        audio, fmt, mime, deps.settings.stt_model, language=language)
        except UpstreamUnconfiguredError as exc:
            # This deployment has no STT address; refuse honestly with the 503
            # the upstream call raised, not the generic 502 backend failure.
            trace.add_stt(model=deps.settings.stt_model, deployment={}, input_bytes=len(audio), transcript_chars=0, error=repr(exc)[:300])
            deps.traces.write(trace)
            return _error(503, exc.body, "transcription_not_configured", kind="server_error", headers=headers)
        except Exception as exc:  # noqa: BLE001
            trace.add_stt(model=deps.settings.stt_model, deployment={}, input_bytes=len(audio), transcript_chars=0, error=repr(exc)[:300])
            deps.traces.write(trace)
            return _error(502, "the audio could not be transcribed right now", "transcription_unavailable", kind="server_error", headers=headers)
        trace.add_stt(model=deps.settings.stt_model, deployment=deployment, input_bytes=len(audio), transcript_chars=len(text))
        return text, response_format, headers, trace

    def answer(text: str, response_format: str, headers: dict):
        if response_format == "text":
            return PlainTextResponse(text, headers=headers)
        return JSONResponse({"text": text}, headers=headers)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        got = await heard(request, "POST /v1/audio/transcriptions", TRANSCRIBE_FIELDS, TRANSCRIBE_REFUSED)
        if not isinstance(got, tuple):
            return got
        text, response_format, headers, trace = got
        deps.traces.write(trace)
        return answer(text, response_format, headers)

    translate_prompt = (PROMPTS_DIR / "translate.md").read_text()

    @app.post("/v1/audio/translations")
    async def translations(request: Request):
        """Two steps (the configured STT transcribes, it does not translate):
        transcribe, then translate the transcript into English with the persona
        model. An English recording comes back as its transcript."""
        got = await heard(request, "POST /v1/audio/translations", TRANSLATE_FIELDS, {"stream"})
        if not isinstance(got, tuple):
            return got
        text, response_format, headers, trace = got
        if not text.strip():
            deps.traces.write(trace)
            return answer("", response_format, headers)
        # The persona backend thinks unless told not to; unasked, reasoning could
        # spend the budget and leave content null (review 2026-09-24 A6).
        call = graph_mod.thinking_switch({
            "model": deps.settings.persona_model, "temperature": 0, "max_completion_tokens": 4096,
            "messages": [{"role": "system", "content": translate_prompt}, {"role": "user", "content": text}]},
            deps.settings.persona_thinking_mode)
        try:
            with trace.timed("translate"):
                data, deployment = await deps.upstream.complete(call)
            english = (data["choices"][0]["message"].get("content") or "").strip()
            if not english:
                # Words were heard (the empty case returned above), so nothing back
                # is the backend failing, not silence: never an empty 200.
                raise ValueError(f"empty translation, finish_reason={data['choices'][0].get('finish_reason')!r}")
        except Exception as exc:  # noqa: BLE001
            trace.set(translate_error=repr(exc)[:300])
            deps.traces.write(trace)
            return _error(502, "the audio could not be translated right now", "translation_unavailable", kind="server_error", headers=headers)
        trace.set(translate={"model": deps.settings.persona_model, "deployment": deployment,
                             "input_chars": len(text), "output_chars": len(english)})
        deps.traces.write(trace)
        return answer(english, response_format, headers)
