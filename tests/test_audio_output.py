"""Audio out: modalities ["text","audio"] returns the finished
words as message.audio, in the configured persona voice, registered as an artifact.
Speech failing never costs her words."""
import base64
import hashlib
import time

from chord import audio
from chord.config import Settings
from chord.upstream import UpstreamError

from test_audio_input import HearingUpstream, trace_of
from test_skeleton import make
from fastapi.testclient import TestClient
from chord.server import Deps, create_app

SPOKEN_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24
SPOKEN_MP3 = b"\xff\xf3\xc0\xc4" + b"\x00" * 24


class SpeakingUpstream(HearingUpstream):
    def __init__(self, spoken=None, tts_error: Exception | None = None, **kw):
        super().__init__(**kw)
        self.spoken = spoken
        self.tts_error = tts_error
        self.said = []

    async def speak(self, text, voice, fmt, model):
        self.said.append((text, voice, fmt, model))
        if self.tts_error:
            raise self.tts_error
        data = self.spoken if self.spoken is not None else (SPOKEN_WAV if fmt == "wav" else SPOKEN_MP3)
        return data, {"model-group": model, "model-api-base": "http://tts.local/v1"}


def ask(client, messages=None, fmt="wav", voice="alloy", stream=False, **extra):
    body = {"model": "chord-1-poly", "modalities": ["text", "audio"], "audio": {"voice": voice, "format": fmt},
            "messages": messages or [{"role": "user", "content": "hi"}], "stream": stream, **extra}
    return client.post("/v1/chat/completions", json=body)


def test_she_answers_in_her_own_voice(tmp_path):
    up = SpeakingUpstream()
    deps, client = make(tmp_path, up)
    before = int(time.time())
    r = ask(client)
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "hi there"
    a = msg["audio"]
    assert a["id"].startswith("audio_")
    assert base64.b64decode(a["data"]) == SPOKEN_WAV
    assert a["transcript"] == msg["content"]
    assert a["expires_at"] >= before + audio.AUDIO_TTL_S
    # her voice from persona_id, never the client's
    assert up.said == [("hi there", "alloy", "wav", "example-tts")]
    assert "artifacts" not in msg  # the spec's message; the artifact is in our trace
    art = [x for x in trace_of(deps, r)["artifacts"] if x["type"] == "audio"]
    assert len(art) == 1 and art[0]["sha256"] == hashlib.sha256(SPOKEN_WAV).hexdigest()
    assert a["id"] == f"audio_{art[0]['id']}"
    # the model is asked for words only: no `audio`, no audio modality
    sent = up.bodies[0]
    assert "audio" not in sent and sent["modalities"] == ["text"]
    tts = trace_of(deps, r)["tts"]
    assert (tts["voice"], tts["model"], tts["output_sha256"]) == ("alloy", "example-tts", art[0]["sha256"])
    assert "error" not in tts


def test_mp3_is_honoured(tmp_path):
    up = SpeakingUpstream()
    _, client = make(tmp_path, up)
    msg = ask(client, fmt="mp3").json()["choices"][0]["message"]
    assert base64.b64decode(msg["audio"]["data"]) == SPOKEN_MP3
    assert up.said[0][2] == "mp3"


def test_speech_failing_keeps_her_words(tmp_path):
    for up in (SpeakingUpstream(tts_error=UpstreamError(500, "riva down")),
               SpeakingUpstream(spoken=SPOKEN_WAV)):  # asked mp3 below, got wav bytes
        deps, client = make(tmp_path, up)
        r = ask(client, fmt="mp3")
        assert r.status_code == 200, r.text
        msg = r.json()["choices"][0]["message"]
        assert msg["content"] == "hi there" and "audio" not in msg and "audio_failed" not in msg
        t = trace_of(deps, r)
        assert "error" in t["tts"] and not [x for x in t["artifacts"] if x["type"] == "audio"]


def test_refused_before_any_call(tmp_path):
    up = SpeakingUpstream()
    _, client = make(tmp_path, up)
    cases = [
        dict(stream=True),                                   # No streamed audio in this route
        dict(fmt="flac"),
        dict(fmt=None),
        dict(voice=7),
        dict(audio="loud"),
    ]
    for c in cases:
        r = ask(client, **c)
        assert r.status_code == 400, c
        assert r.json()["error"]["code"] == "invalid_audio_request"
    assert up.bodies == [] and up.said == []


def test_a_girl_without_a_voice_is_refused(tmp_path):
    up = SpeakingUpstream()
    deps = Deps(Settings(data_dir=tmp_path, tts_voices="{}"), upstream=up, model=lambda n: None)
    r = ask(TestClient(create_app(deps)))
    assert r.status_code == 400 and "no voice" in r.json()["error"]["message"]
    assert up.bodies == []


def test_no_audio_asked_means_no_speech(tmp_path):
    up = SpeakingUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "audio": {"voice": "x", "format": "wav"},
                                                  "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "audio" not in r.json()["choices"][0]["message"]
    assert up.said == [] and "audio" not in up.bodies[0]


def test_replayed_audio_goes_back_to_her_as_words(tmp_path):
    up = SpeakingUpstream()
    _, client = make(tmp_path, up)
    first = ask(client).json()["choices"][0]["message"]
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": None, "audio": {"id": first["audio"]["id"]}},
               {"role": "assistant", "content": None, "audio": {"id": "audio_unknown"}},
               {"role": "user", "content": "again"}]
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": history})
    assert r.status_code == 200, r.text
    sent = up.bodies[-1]["messages"]
    assistant = [m for m in sent if m["role"] == "assistant"]
    assert [m["content"] for m in assistant] == ["hi there", audio.SENT_EARLIER]
    assert all("audio" not in m for m in sent)
    bad = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [
        {"role": "assistant", "content": None, "audio": "x"}, {"role": "user", "content": "hi"}]})
    assert bad.status_code == 400


def test_upstream_speak_posts_openai_speech():
    """The real client: JSON to /audio/speech, raw bytes back, errors raise."""
    import asyncio
    import json

    import httpx

    from chord.upstream import Upstream

    seen = {}

    def handler(request):
        seen["path"], seen["body"] = request.url.path, json.loads(request.read())
        if seen["body"]["input"] == "fail":
            return httpx.Response(500, text="riva down")
        return httpx.Response(200, content=SPOKEN_WAV, headers={"content-type": "audio/mpeg", "x-litellm-model-group": "example/tts"})

    up = Upstream("http://chat-host/v1", "k")
    up._tts = httpx.AsyncClient(base_url="http://speech-host/v1", transport=httpx.MockTransport(handler))
    data, dep = asyncio.run(up.speak("hi", "alloy", "wav", "example/tts"))
    assert (data, dep) == (SPOKEN_WAV, {"model-group": "example/tts"})
    assert seen == {"path": "/v1/audio/speech", "body": {"model": "example/tts", "input": "hi", "voice": "alloy", "response_format": "wav"}}
    up._tts = httpx.AsyncClient(base_url="http://speech-host/v1", transport=httpx.MockTransport(handler))
    try:
        asyncio.run(up.speak("fail", "alloy", "wav", "example/tts"))
        raise AssertionError("a 500 must raise")
    except UpstreamError as exc:
        assert exc.status == 500


def test_unhashable_or_non_string_shapes_are_400s_with_no_calls(tmp_path):
    """the #37 review: format=[]/{} was a bare 500; a non-string replayed
    transcript became malformed upstream content."""
    up = SpeakingUpstream()
    _, client = make(tmp_path, up)
    for fmt in ([], {}, 1):
        r = ask(client, fmt=fmt)
        assert r.status_code == 400, fmt
        assert r.json()["error"]["code"] == "invalid_audio_request"
    for transcript in (123, {"a": 1}, ["x"]):
        r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [
            {"role": "assistant", "content": None, "audio": {"id": "audio_x", "transcript": transcript}},
            {"role": "user", "content": "hi"}]})
        assert r.status_code == 400, transcript
    assert up.bodies == [] and up.said == []


def test_a_different_voice_is_refused_not_ignored(tmp_path):
    """Regression: a caller's audio.voice was silently replaced. Her voice comes from
    persona_id, so asking for another is a 400 that names hers; leaving it out
    gets hers; asking for hers is fine."""
    up = SpeakingUpstream()
    _, client = make(tmp_path, up)
    r = ask(client, voice="nova")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_audio_request"
    assert "'alloy'" in r.json()["error"]["message"]
    assert up.bodies == [] and up.said == []
    body = {"model": "chord-1-poly", "modalities": ["text", "audio"], "audio": {"format": "wav"},
            "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert ask(client, voice="alloy").status_code == 200
    assert [s[1] for s in up.said] == ["alloy", "alloy"]
