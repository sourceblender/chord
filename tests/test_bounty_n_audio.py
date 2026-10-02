"""Bug-bounty probe (testing the #242/#244 `n` path, 2026-09-17).

Attack: `n` above 1 with an audio INPUT. The Transcriber has a sha256-keyed cache
"so a voice message is transcribed once, not on every later turn" (audio.py), but
run_chat_n fans the n turns out with asyncio.gather. The cache is written only
AFTER the await, so every concurrent turn misses it.

Controls are asserted first in each test: the cache demonstrably works when the
same audio arrives sequentially, and n=1 costs exactly one STT call. Only then is
the concurrent case measured.
"""
import base64

import pytest

from test_audio_input import HearingUpstream, voice
from test_skeleton import make

WAV_B64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()


def _post(client, n=None, **extra):
    body = {"model": "chord-1-poly", "messages": [voice(WAV_B64)], **extra}
    if n is not None:
        body["n"] = n
        body.setdefault("temperature", 0.7)   # n above 1 needs sampling
    return client.post("/v1/chat/completions", json=body)


def test_control_one_turn_costs_one_stt_and_the_cache_holds_sequentially(tmp_path):
    up = HearingUpstream()
    _, client = make(tmp_path, upstream=up)

    r = _post(client)
    assert r.status_code == 200, r.text
    assert len(up.transcribed) == 1, "control: one turn, one STT call"

    r = _post(client)                      # same audio again, sequentially
    assert r.status_code == 200, r.text
    assert len(up.transcribed) == 1, "control: the sha256 cache serves the second turn"


def test_n_above_one_pays_one_stt_call_per_choice(tmp_path):
    """The finding: the cache gives no protection in the one place n uses it."""
    up = HearingUpstream()
    _, client = make(tmp_path, upstream=up)

    r = _post(client, n=4)
    assert r.status_code == 200, r.text
    assert len(r.json()["choices"]) == 4

    calls = len(up.transcribed)
    assert calls == 1, (
        f"one audio, one transcript, but n=4 made {calls} STT calls: the cache is "
        "checked before the await and written after it, so concurrent turns all miss")


class SlowHearingUpstream(HearingUpstream):
    """A transcribe that actually SUSPENDS, as a network STT call does. The plain
    fake returns without yielding, so the event loop never interleaves the n turns
    and a concurrency race cannot appear."""

    async def transcribe(self, audio_bytes, fmt, mime, model):
        import asyncio
        await asyncio.sleep(0.01)
        return await super().transcribe(audio_bytes, fmt, mime, model)


def test_control_slow_stt_still_caches_sequentially(tmp_path):
    up = SlowHearingUpstream()
    _, client = make(tmp_path, upstream=up)
    assert _post(client).status_code == 200
    assert _post(client).status_code == 200
    assert len(up.transcribed) == 1, "control: a suspending STT still caches across turns"


def test_n_with_a_suspending_stt_does_not_stampede(tmp_path):
    up = SlowHearingUpstream()
    _, client = make(tmp_path, upstream=up)
    r = _post(client, n=4)
    assert r.status_code == 200, r.text
    assert len(r.json()["choices"]) == 4
    calls = len(up.transcribed)
    assert calls == 1, (
        f"one audio, n=4, {calls} STT calls: with a real (suspending) STT the "
        "concurrent turns all miss the cache, which is written only after the await")


@pytest.mark.asyncio
async def test_cancelling_one_caller_does_not_cancel_the_shared_transcription():
    import asyncio

    from chord.audio import Transcriber
    from chord.trace import Trace

    started = asyncio.Event()
    release = asyncio.Event()

    class Upstream:
        calls = 0

        async def transcribe(self, raw, fmt, mime, model):
            self.calls += 1
            started.set()
            await release.wait()
            return "heard", {"backend": "test"}

    upstream = Upstream()
    transcriber = Transcriber(upstream, "stt")
    raw = b"same-clip"

    def trace():
        return Trace(persona_id="generic", model_id_requested=None)

    first = asyncio.create_task(transcriber._hear(raw, "wav", "digest", trace()))
    await started.wait()
    second = asyncio.create_task(transcriber._hear(raw, "wav", "digest", trace()))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    heard = await second
    assert heard.text == "heard"
    assert upstream.calls == 1
    assert transcriber._inflight == {}


@pytest.mark.asyncio
async def test_a_caller_who_arrives_after_the_last_waiter_left_starts_fresh():
    import asyncio

    from chord.audio import Transcriber
    from chord.trace import Trace

    started = asyncio.Event()
    release = asyncio.Event()

    class Upstream:
        calls = 0

        async def transcribe(self, raw, fmt, mime, model):
            self.calls += 1
            started.set()
            await release.wait()
            return "heard", {"backend": "test"}

    upstream = Upstream()
    transcriber = Transcriber(upstream, "stt")
    raw = b"same-clip"

    def trace():
        return Trace(persona_id="generic", model_id_requested=None)

    first = asyncio.create_task(transcriber._hear(raw, "wav", "digest", trace()))
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    started.clear()
    release.set()
    second = asyncio.create_task(transcriber._hear(raw, "wav", "digest", trace()))
    heard = await second
    assert heard.text == "heard"
    assert upstream.calls == 2
