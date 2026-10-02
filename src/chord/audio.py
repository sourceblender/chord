"""Audio in: voice messages become text before routing.

An `input_audio` part in a USER message is sent to the STT model and replaced
by a text part marked as transcribed, so the router, specialists and reply
generation all work on words. Nothing else in the graph has to know audio exists.

Rules:
- The newest user message's audio has to transcribe. If STT fails or hears
  nothing, the turn ends as `failed` and the assistant says it could not hear it; it
  never proceeds as though the user said nothing.
- Audio in OLDER messages (clients resend the whole conversation) is looked up
  in a cache keyed by the audio's sha256, so a voice message is transcribed
  once, not on every later turn. If an older one can't be transcribed it
  becomes a marker, and the current turn goes on.
- Every transcription is traced: model, deployment, input sha256, transcript
  length, whether it came from the cache.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg"}
TRANSCRIBED = "[voice message, transcribed] "
UNHEARD = "[voice message that could not be transcribed]"
_CACHE_MAX = 256
# The same 25 MB wire cap audio_api puts on a standalone transcription upload
# (OpenAI's STT guide). A chat request has no global body-size limit, and
# transcribe_messages decodes the whole payload into RAM before hashing it, so
# this per-part cap is the only bound between a caller and that decode. It is
# checked on the ENCODED length, before decoding: measuring by decoding first
# pays the very cost the cap exists to prevent (review 2026-09-22).
MAX_INPUT_AUDIO_BYTES = 25 * 1024 * 1024
MAX_INPUT_AUDIO_B64_CHARS = ((MAX_INPUT_AUDIO_BYTES + 2) // 3) * 4


class AudioError(Exception):
    """The newest voice message could not be turned into words. `backend` is
    True when transcription itself failed (our fault, a 502), False when it
    ran and heard nothing (the caller's audio, a 400)."""

    def __init__(self, message: str, backend: bool = False):
        super().__init__(message)
        self.backend = backend


def check_part(part: dict) -> str | None:
    """Shape check for an input_audio part, run during request validation.
    Returns an error message, or None when the part is well formed."""
    ia = part.get("input_audio")
    if not isinstance(ia, dict):
        return "input_audio must be an object with data and format"
    if not isinstance(ia.get("format"), str) or ia["format"] not in FORMATS:
        return f"input_audio.format must be one of {sorted(FORMATS)}"
    data = ia.get("data")
    if not isinstance(data, str) or not data:
        return "input_audio.data must be a non-empty base64 string"
    if len(data) > MAX_INPUT_AUDIO_B64_CHARS:
        return f"input_audio.data must be at most {MAX_INPUT_AUDIO_BYTES} bytes of audio"
    try:
        base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return "input_audio.data is not valid base64"
    return None


@dataclass
class _Heard:
    text: str
    cached: bool


class Transcriber:
    """STT with a small in-process cache, shared across requests."""

    def __init__(self, upstream, model: str) -> None:
        self._upstream = upstream
        self._model = model
        self._cache: OrderedDict[str, str] = OrderedDict()
        # One transcription per audio IN FLIGHT, not just per audio already heard. The cache is
        # read before the await and written after it, so concurrent turns for the same clip all
        # miss it: `n` above 1 fans its turns out with asyncio.gather, and one voice message cost
        # n paid STT calls and n round trips (bug bounty 2026-09-17). Callers that arrive
        # while a digest is in flight await the same task instead of starting their own.
        self._inflight: dict[str, asyncio.Task] = {}
        # Counted per TASK, not per digest: once a task finishes, its done-callback drops the
        # in-flight entry before its waiters resume, and a new request for the same clip can
        # start a new task in that gap. A per-digest count let the new request reset the old
        # waiters' count, and the second old waiter then KeyErrored in its `finally`, turning
        # a good transcription into a 502 (review 2026-10-01).
        self._waiters: dict[asyncio.Task, int] = {}

    def _drop_inflight(self, task: asyncio.Task) -> None:
        """Remove `task` when IT finishes, not when a waiter leaves.

        A cancelled waiter used to delete the entry from its own `finally` while
        the transcription was still running, so the next caller started a second
        one."""
        for digest, current in list(self._inflight.items()):
            if current is task:
                del self._inflight[digest]
                return

    async def _hear(self, raw: bytes, fmt: str, digest: str, trace) -> _Heard:
        if digest in self._cache:
            self._cache.move_to_end(digest)
            text = self._cache[digest]
            trace.add_stt(model=self._model, deployment={}, input_sha256=digest, transcript_chars=len(text), cached=True)
            return _Heard(text, True)
        task = self._inflight.get(digest)
        # A task that is already finishing, or whose waiter count was removed,
        # is not joinable. Attaching there either KeyErrors or waits on a
        # cancellation that belongs to the previous caller.
        if task is None or task.done() or task not in self._waiters:
            task = asyncio.ensure_future(self._upstream.transcribe(raw, fmt, FORMATS[fmt], self._model))
            self._inflight[digest] = task
            self._waiters[task] = 0
            task.add_done_callback(self._drop_inflight)
            shared = False
        else:
            shared = True
        self._waiters[task] += 1
        try:
            # Shield so cancelling this waiter does not cancel the shared task
            # while another caller is still on it. CancelledError is not an Exception.
            text, deployment = await asyncio.shield(task)
        finally:
            self._waiters[task] -= 1
            if self._waiters[task] <= 0:
                del self._waiters[task]
                # Drop the entry before cancel, still without awaiting, so a new
                # caller starts a fresh transcription instead of joining this one.
                if self._inflight.get(digest) is task:
                    del self._inflight[digest]
                if not task.done():
                    task.cancel()
        text = (text or "").strip()
        # No await between here and the cache write, so nothing can slip in and start a second call.
        trace.add_stt(model=self._model, deployment=deployment, input_sha256=digest,
                      transcript_chars=len(text), cached=False, shared=shared)
        if text:
            self._cache[digest] = text
            if len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)
        return _Heard(text, False)

    async def transcribe_messages(self, messages: list[dict], trace) -> list[dict]:
        """Return the messages with every user input_audio part replaced by text.
        Raises AudioError if the NEWEST user message's audio can't be heard."""
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        out = []
        for i, m in enumerate(messages):
            content = m.get("content")
            if m.get("role") != "user" or not isinstance(content, list) or not any(
                isinstance(p, dict) and p.get("type") == "input_audio" for p in content
            ):
                out.append(m)
                continue
            parts = []
            for p in content:
                if not (isinstance(p, dict) and p.get("type") == "input_audio"):
                    parts.append(p)
                    continue
                raw = base64.b64decode(p["input_audio"]["data"])
                digest = hashlib.sha256(raw).hexdigest()
                backend = False
                try:
                    heard = await self._hear(raw, p["input_audio"]["format"], digest, trace)
                except Exception as exc:  # STT unreachable, 4xx/5xx, bad reply
                    trace.add_stt(model=self._model, deployment={}, input_sha256=digest, transcript_chars=0, cached=False, error=repr(exc)[:200])
                    heard, backend = _Heard("", False), True
                if heard.text:
                    parts.append({"type": "text", "text": TRANSCRIBED + heard.text})
                elif i == last_user:
                    raise AudioError(f"voice message {digest[:16]} could not be transcribed", backend=backend)
                else:
                    parts.append({"type": "text", "text": UNHEARD})
            out.append({**m, "content": parts})
        return out


# --- Audio out ---------------------------------------------------------------
#
# modalities ["text","audio"] + audio:{voice, format}: the assistant's finished
# words are spoken in its configured voice (persona_id -> voice; asking for
# another is a 400). The result is standard message.audio
# {id, data, expires_at, transcript}. The
# bytes are registered as an artifact so the client can check the sha256.
# If speech fails, the words still go out as text, without `audio`, and the
# failure is only in the trace (`tts.error`); the text is never lost to a TTS
# problem. (The `audio_failed` body extension was retired 2026-09-16.)

OUT_FORMATS = {"wav": "audio/wav", "mp3": "audio/mpeg"}
AUDIO_TTL_S = 3600
SENT_EARLIER = "[voice reply you sent earlier]"


def check_audio_request(body: dict, stream: bool, voice: str | None) -> str | None:
    """Validation for an audio-output request. Returns an error, or None."""
    if stream:
        return "audio output is not supported on a streamed response yet; send stream: false"
    params = body.get("audio")
    if not isinstance(params, dict):
        return "modalities includes audio, so audio: {voice, format} is required"
    if not isinstance(params.get("format"), str) or params["format"] not in OUT_FORMATS:
        return f"audio.format must be one of {sorted(OUT_FORMATS)}"
    if voice is None:
        return "this persona has no voice configured"
    asked = params.get("voice")
    # The voice comes from persona_id. A different request voice is refused,
    # not silently replaced; leaving it out selects the configured one.
    if asked is not None and asked != voice:
        return f"this model speaks in voice {voice!r}; send audio.voice {voice!r} or leave it out"
    return None


class Speaker:
    """TTS for a finished reply; remembers what each audio id said, so a client
    replaying `{"role":"assistant","audio":{"id":...}}` gets the spoken words back."""

    def __init__(self, upstream, model: str, artifacts) -> None:
        self._upstream = upstream
        self._model = model
        self._artifacts = artifacts
        self._said: OrderedDict[str, str] = OrderedDict()

    async def speak(self, text: str, voice: str, fmt: str, trace) -> tuple[dict, dict] | None:
        """(message.audio, artifact descriptor), or None if it couldn't be spoken."""
        entry = {"model": self._model, "voice": voice, "format": fmt, "input_chars": len(text)}
        try:
            data, deployment = await self._upstream.speak(text, voice, fmt, self._model)
            entry["deployment"] = deployment
            # The content-type isn't trusted (a certified TTS backend labels WAV audio/mpeg);
            # register checks the magic bytes against the format asked for.
            descriptor = self._artifacts.register(data, OUT_FORMATS[fmt])
        except Exception as exc:  # TTS down, 4xx/5xx, wrong bytes
            trace.tts = {**entry, "error": repr(exc)[:200]}
            return None
        trace.tts = {**entry, "output_sha256": descriptor.sha256, "artifact_id": descriptor.id}
        audio_id = f"audio_{descriptor.id}"
        self._said[audio_id] = text
        if len(self._said) > _CACHE_MAX:
            self._said.popitem(last=False)
        message_audio = {
            "id": audio_id,
            "data": base64.b64encode(data).decode(),
            "expires_at": int(time.time()) + AUDIO_TTL_S,
            "transcript": text,
        }
        return message_audio, descriptor.model_dump()

    def replayed(self, messages: list[dict]) -> list[dict]:
        """Assistant history carrying our `audio` goes back to the model as words."""
        out = []
        for m in messages:
            if m.get("role") != "assistant" or not isinstance(m.get("audio"), dict):
                out.append(m)
                continue
            m = dict(m)
            ref = m.pop("audio")
            if not m.get("content"):
                m["content"] = ref.get("transcript") or self._said.get(ref.get("id"), SENT_EARLIER)
            out.append(m)
        return out
