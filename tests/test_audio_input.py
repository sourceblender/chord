"""Audio in: a voice message is transcribed before routing,
marked as transcribed, and a newest message nobody could hear ends the turn as
`failed` in her words, with no model call."""
import asyncio
import base64
import json

from fastapi.testclient import TestClient

from chord import audio
from chord.server import create_internal_app
from chord.upstream import UpstreamError

from test_skeleton import FakeUpstream, make

WAV = b"RIFF" + b"\x00" * 40
WAV_B64 = base64.b64encode(WAV).decode()
OTHER_B64 = base64.b64encode(b"RIFF" + b"\x01" * 40).decode()


class HearingUpstream(FakeUpstream):
    """FakeUpstream plus /audio/transcriptions."""

    def __init__(self, heard="hello there", stt_error: Exception | None = None, **kw):
        super().__init__(**kw)
        self.heard = heard
        self.stt_error = stt_error
        self.transcribed = []
        self.languages = []

    async def transcribe(self, audio_bytes, fmt, mime, model, *, language=None):
        await asyncio.sleep(0)   # a real STT call suspends; without this no concurrent miss can appear
        self.transcribed.append((audio_bytes, fmt, mime, model))
        self.languages.append(language)
        if self.stt_error:
            raise self.stt_error
        return self.heard, {"model-group": model, "model-api-base": "http://stt.local/v1"}


def voice(b64=WAV_B64, fmt="wav", text=None):
    parts = [{"type": "text", "text": text}] if text else []
    return {"role": "user", "content": parts + [{"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}]}


def post(client, messages, stream=False):
    return client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": messages, "stream": stream})


def trace_of(deps, r):
    return TestClient(create_internal_app(deps)).get(f"/internal/traces/{r.headers['x-chord-trace-id']}").json()


def frames(r):
    return [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]


def test_voice_message_reaches_her_as_marked_text(tmp_path):
    up = HearingUpstream()
    deps, client = make(tmp_path, up)
    r = post(client, [voice(text="listen to this")])
    assert r.status_code == 200, r.text
    assert "outcome" not in r.json()["choices"][0]["message"]
    assert up.transcribed == [(WAV, "wav", "audio/wav", "example-stt")]
    user = up.bodies[0]["messages"][-1]
    assert user["content"] == [
        {"type": "text", "text": "listen to this"},
        {"type": "text", "text": audio.TRANSCRIBED + "hello there"},
    ]
    stt = trace_of(deps, r)["stt"]
    assert len(stt) == 1
    assert stt[0]["model"] == "example-stt"
    assert stt[0]["deployment"]["model-api-base"] == "http://stt.local/v1"
    assert stt[0]["transcript_chars"] == len("hello there")
    assert stt[0]["cached"] is False
    assert len(stt[0]["input_sha256"]) == 64


def test_unheard_newest_message_is_an_error_without_a_model_call(tmp_path):
    """Nothing heard is the caller's audio (400); transcription failing is ours
    (502). Either way an error, never a 200 reply the caller can't tell from an
    answer (2026-09-16)."""
    for heard, err, status, code in (("", None, 400, "audio_unintelligible"), ("   ", None, 400, "audio_unintelligible"),
                                     ("x", UpstreamError(500, "stt down"), 502, "transcription_unavailable")):
        for stream in (False, True):
            up = HearingUpstream(heard=heard, stt_error=err)
            deps, client = make(tmp_path, up)
            r = post(client, [voice()], stream=stream)
            assert r.status_code == status, r.text
            assert r.json()["error"]["code"] == code
            assert up.bodies == []  # no router, no persona call
            t = trace_of(deps, r)
            assert t["result_status"] == "failed"
            assert t["stt"][0]["transcript_chars"] == 0  # Present on a failed turn too
            assert ("error" in t["stt"][0]) == (err is not None)


def test_unheard_older_message_becomes_a_marker_and_the_turn_goes_on(tmp_path):
    up = HearingUpstream(heard="")
    _, client = make(tmp_path, up)
    r = post(client, [voice(), {"role": "assistant", "content": "hm?"}, {"role": "user", "content": "never mind"}])
    assert r.status_code == 200, r.text
    sent = up.bodies[0]["messages"]
    assert {"type": "text", "text": audio.UNHEARD} in sent[1]["content"]
    assert sent[-1]["content"] == "never mind"


def test_a_voice_message_is_transcribed_once_across_turns(tmp_path):
    up = HearingUpstream()
    deps, client = make(tmp_path, up)
    post(client, [voice()])
    r = post(client, [voice(), {"role": "assistant", "content": "hi there"}, voice(OTHER_B64)])
    assert r.status_code == 200, r.text
    assert len(up.transcribed) == 2  # the first clip once, the new clip once
    assert [e["cached"] for e in trace_of(deps, r)["stt"]] == [True, False]


def test_audio_outside_a_user_message_is_refused(tmp_path):
    up = HearingUpstream()
    _, client = make(tmp_path, up)
    for role in ("assistant", "system", "tool"):
        r = post(client, [{**voice(), "role": role}, {"role": "user", "content": "hi"}])
        assert r.status_code == 400, role
        assert r.json()["error"]["code"] == "invalid_input_audio"
    assert up.transcribed == [] and up.bodies == []


def test_malformed_audio_is_refused_before_any_call(tmp_path):
    up = HearingUpstream()
    _, client = make(tmp_path, up)
    for part in (
        {"type": "input_audio", "input_audio": {"data": "not base64!!", "format": "wav"}},
        {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "ogg"}},
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
        {"type": "input_audio", "input_audio": "AAAA"},
        {"type": "input_audio"},
    ):
        r = post(client, [{"role": "user", "content": [part]}])
        assert r.status_code == 400, part
        assert r.json()["error"]["code"] == "invalid_input_audio"
    assert up.transcribed == [] and up.bodies == []


def test_upstream_transcribe_sends_openai_multipart():
    """The real client, not the fake: the shape LiteLLM's /audio/transcriptions takes."""
    import asyncio

    import httpx

    from chord.upstream import Upstream

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["ctype"], seen["body"] = request.url.path, request.headers["content-type"], request.read()
        if seen.get("fail"):
            return httpx.Response(503, text="no replica")
        return httpx.Response(200, json={"text": "hello"}, headers={"x-litellm-model-group": "example/stt"})

    up = Upstream("http://chat-host/v1", "k")
    up._stt = httpx.AsyncClient(base_url="http://speech-host/v1", transport=httpx.MockTransport(handler))
    text, dep = asyncio.run(up.transcribe(WAV, "wav", "audio/wav", "example/stt", language="fr"))
    assert (text, dep) == ("hello", {"model-group": "example/stt"})
    assert seen["path"] == "/v1/audio/transcriptions"
    assert seen["ctype"].startswith("multipart/form-data")
    assert b'name="model"\r\n\r\nexample/stt' in seen["body"]
    assert b'name="language"\r\n\r\nfr' in seen["body"]
    assert b'name="file"; filename="voice.wav"\r\nContent-Type: audio/wav\r\n\r\n' + WAV in seen["body"]
    seen["fail"] = True
    up._stt = httpx.AsyncClient(base_url="http://speech-host/v1", transport=httpx.MockTransport(handler))
    try:
        asyncio.run(up.transcribe(WAV, "wav", "audio/wav", "example/stt"))
        raise AssertionError("a 503 must raise")
    except UpstreamError as exc:
        assert exc.status == 503


def test_non_string_format_is_a_400_not_a_500(tmp_path):
    """the #33 finding: an unhashable format crashed the membership check."""
    up = HearingUpstream()
    _, client = make(tmp_path, up)
    for fmt in ([], {}, 1, None):
        r = post(client, [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": WAV_B64, "format": fmt}}]}])
        assert r.status_code == 400, fmt
        assert r.json()["error"]["code"] == "invalid_input_audio"
    assert up.transcribed == []


def test_an_oversized_voice_message_is_refused_before_decoding(tmp_path, monkeypatch):
    """The cap is on the ENCODED length and runs before the base64 decode:
    check_part used to decode the whole payload just to validate it, and
    nothing else bounds a chat body, so a multi-gigabyte input_audio part was
    fully resident before anyone could measure it (review 2026-09-22)."""
    monkeypatch.setattr("chord.audio.MAX_INPUT_AUDIO_B64_CHARS", 16)
    up = HearingUpstream()
    _, client = make(tmp_path, up)
    r = post(client, [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}}]}])
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_input_audio"
    assert up.transcribed == [] and up.bodies == []
