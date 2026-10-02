"""The Audio API (#139): POST /v1/audio/speech and /v1/audio/transcriptions,
non-streaming, in the pinned spec's shapes, driven by the official SDK where a
client would use it. What the tested backend can't do is a named 400, never a
backend 500 passed through."""
import io
import json

import openai
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from qa.conformance.schema import Spec, validate_payload
from chord import audio_api
from chord.config import Settings
from chord.server import Deps, create_app, create_internal_app
from chord.upstream import UpstreamError, UpstreamUnconfiguredError
from test_audio_input import HearingUpstream
from test_audio_output import SPOKEN_MP3, SPOKEN_WAV, SpeakingUpstream

MODEL = "chord-1-poly"


def _wav() -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x01" * 800)
    return buf.getvalue()


WAV = _wav()


class Formats(SpeakingUpstream):
    """Bytes that really are each format (a TTS backend we certified against labels all of them audio/mpeg)."""
    BYTES = {"mp3": SPOKEN_MP3, "wav": SPOKEN_WAV, "opus": b"OggS" + b"\x00" * 30,
             "flac": b"fLaC" + b"\x00" * 30, "pcm": b"\x01\x00" * 20}

    async def speak(self, text, voice, fmt, model):
        self.said.append((text, voice, fmt, model))
        if self.tts_error:
            raise self.tts_error
        return self.BYTES[fmt], {"model-group": model}


def app(tmp_path, upstream=None):
    deps = Deps(Settings(data_dir=tmp_path,
                         persona_model="example-persona", router_model="example-router",
                         persona_base_url="http://persona.test/v1", router_base_url="http://router.test/v1",
                         stt_model="example-stt", tts_model="example-tts",
                         persona_thinking_mode="qwen_chat_template"),
                upstream=upstream or Formats(), model=lambda n: None)
    return deps, TestClient(create_app(deps))


def sdk(client):
    return OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=client)


def _error_message(body) -> str:
    """Pull the OpenAI-shaped error message out of an APIStatusError.body, which
    can be a parsed dict (with or without the {error: {...}} envelope), raw bytes,
    or None depending on the response shape."""
    if isinstance(body, dict):
        if isinstance(body.get("error"), dict):
            return body["error"].get("message", "")
        return body.get("message", "")
    if isinstance(body, (bytes, bytearray)):
        return body.decode(errors="replace")
    return str(body or "")


# --- speech ------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,mime", [("mp3", "audio/mpeg"), ("opus", "audio/ogg"), ("flac", "audio/flac"),
                                      ("wav", "audio/wav"), ("pcm", "audio/pcm")])
def test_speech_returns_the_bytes_in_the_asked_format(tmp_path, fmt, mime):
    deps, client = app(tmp_path)
    audio = sdk(client).audio.speech.create(model=MODEL, input="Good morning.", voice="alloy", response_format=fmt)
    assert audio.content == Formats.BYTES[fmt] and audio.response.headers["content-type"] == mime
    assert deps.upstream.said == [("Good morning.", "alloy", fmt, "example-tts")]
    t = TestClient(create_internal_app(deps)).get(f"/internal/traces/{audio.response.headers['x-request-id']}").json()
    assert t["operation"] == "POST /v1/audio/speech" and t["tts"]["output_bytes"] == len(Formats.BYTES[fmt])


def test_speech_defaults_to_mp3(tmp_path):
    deps, client = app(tmp_path)
    r = client.post("/v1/audio/speech", json={"model": MODEL, "input": "hi", "voice": "alloy"})
    assert r.status_code == 200 and r.content == SPOKEN_MP3 and deps.upstream.said[0][2] == "mp3"


@pytest.mark.parametrize("change,param,status", [
    ({"model": "tts-1"}, "model", 404),
    ({"input": ""}, "input", 400),
    ({"input": "x" * 4097}, "input", 400),
    ({"voice": "nova"}, "voice", 400),                      # the certified TTS backend 500s on a voice it lacks
    ({"voice": {"id": "voice_123"}}, "voice", 400),
    ({"response_format": "aac"}, "response_format", 400),   # the certified TTS backend 500s on aac
    ({"speed": 2.0}, "speed", 400),                         # the certified TTS backend 500s on any speed but 1.0
    ({"instructions": "whisper"}, "instructions", 400),
    ({"stream_format": "sse"}, "stream_format", 400),
    ({"bogus": 1}, "bogus", 400),
])
def test_speech_refuses_what_the_backend_cannot_do_before_calling_it(tmp_path, change, param, status):
    deps, client = app(tmp_path)
    r = client.post("/v1/audio/speech", json={"model": MODEL, "input": "hi", "voice": "alloy", **change})
    assert r.status_code == status, r.text
    assert r.json()["error"]["param"] == param
    assert validate_payload(r.json(), kind="error", spec=Spec(), fields="strict")["verdict"] == "pass"
    assert deps.upstream.said == []


def test_speech_speed_one_and_audio_stream_format_are_fine(tmp_path):
    deps, client = app(tmp_path)
    r = client.post("/v1/audio/speech", json={"model": MODEL, "input": "hi", "voice": "alloy", "speed": 1.0, "stream_format": "audio"})
    assert r.status_code == 200


@pytest.mark.parametrize("upstream", [Formats(tts_error=UpstreamError(500, "down")), SpeakingUpstream()],
                         ids=["backend-error", "wrong-bytes"])
def test_speech_backend_failure_is_a_502_never_bad_bytes(tmp_path, upstream):
    deps, client = app(tmp_path, upstream)
    with pytest.raises(openai.InternalServerError) as exc:   # SpeakingUpstream returns mp3 bytes for opus
        sdk(client).audio.speech.create(model=MODEL, input="hi", voice="alloy", response_format="opus")
    assert exc.value.status_code == 502 and exc.value.code == "speech_generation_failed"


def test_speech_unconfigured_returns_503_not_502(tmp_path):
    """Copilot review of #340 (2026-09-25): the upstream refusal is 503; the handler
    must surface that 503 honestly, not absorb it into the generic 502 backend-failure
    response. test_audio_unset_refuses.py covers Upstream directly; this exercises
    the handler that was masking the status."""
    deps, client = app(tmp_path, Formats(tts_error=UpstreamUnconfiguredError("text-to-speech is not configured: set TTS_BASE_URL")))
    with pytest.raises(openai.APIStatusError) as exc:
        sdk(client).audio.speech.create(model=MODEL, input="hi", voice="alloy", response_format="wav")
    assert exc.value.status_code == 503
    assert exc.value.code == "speech_not_configured"
    assert "TTS_BASE_URL" in _error_message(exc.value.body)


# --- transcriptions ------------------------------------------------------------------

def test_transcription_json_through_the_sdk(tmp_path):
    deps, client = app(tmp_path, HearingUpstream(heard="hello there"))
    t = sdk(client).audio.transcriptions.create(model=MODEL, file=("clip.wav", io.BytesIO(WAV)))
    assert t.text == "hello there"
    assert deps.upstream.transcribed == [(WAV, "wav", "audio/wav", "example-stt")]
    assert validate_payload({"text": t.text}, kind="transcription", spec=Spec(), fields="strict")["verdict"] == "pass"


def test_transcription_text_format(tmp_path):
    deps, client = app(tmp_path, HearingUpstream(heard="hello there"))
    text = sdk(client).audio.transcriptions.create(model=MODEL, file=("clip.wav", io.BytesIO(WAV)), response_format="text")
    assert text.strip() == "hello there"


def test_the_file_format_comes_from_its_bytes_not_its_name(tmp_path):
    deps, client = app(tmp_path, HearingUpstream())
    r = client.post("/v1/audio/transcriptions", data={"model": MODEL}, files={"file": ("clip.mp3", WAV, "audio/mpeg")})
    assert r.status_code == 200 and deps.upstream.transcribed[0][1:3] == ("wav", "audio/wav")


def test_hints_are_accepted_and_traced(tmp_path):
    deps, client = app(tmp_path, HearingUpstream())
    r = client.post("/v1/audio/transcriptions", data={"model": MODEL, "prompt": "names: Ava", "temperature": "0.2", "language": "en"},
                    files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 200
    assert deps.upstream.languages == ["en"]
    t = TestClient(create_internal_app(deps)).get(f"/internal/traces/{r.headers['x-request-id']}").json()
    assert t["stt_hints"] == {"prompt": True, "temperature": True, "language": True}


@pytest.mark.parametrize("data,files,param,status", [
    ({"model": "whisper-1"}, {"file": ("a.wav", WAV, "audio/wav")}, "model", 404),
    ({}, {"model": (None, MODEL)}, "file", 400),
    ({"model": MODEL}, {"file": ("a.wav", b"not audio at all", "audio/wav")}, "file", 400),
    ({"model": MODEL, "response_format": "verbose_json"}, {"file": ("a.wav", WAV, "audio/wav")}, "response_format", 400),
    ({"model": MODEL, "stream": "true"}, {"file": ("a.wav", WAV, "audio/wav")}, "stream", 400),
    ({"model": MODEL, "timestamp_granularities[]": "word"}, {"file": ("a.wav", WAV, "audio/wav")}, "timestamp_granularities", 400),
    ({"model": MODEL, "include[]": "logprobs"}, {"file": ("a.wav", WAV, "audio/wav")}, "include", 400),
    ({"model": MODEL, "temperature": "3"}, {"file": ("a.wav", WAV, "audio/wav")}, "temperature", 400),
    ({"model": MODEL, "language": ""}, {"file": ("a.wav", WAV, "audio/wav")}, "language", 400),
    ({"model": MODEL, "bogus": "1"}, {"file": ("a.wav", WAV, "audio/wav")}, "bogus", 400),
])
def test_transcription_refusals(tmp_path, data, files, param, status):
    deps, client = app(tmp_path, HearingUpstream())
    r = client.post("/v1/audio/transcriptions", data=data, files=files)
    assert r.status_code == status, r.text
    assert r.json()["error"]["param"] == param
    assert deps.upstream.transcribed == []


def test_a_file_part_named_temperature_is_a_400_not_a_500(tmp_path):
    deps, client = app(tmp_path, HearingUpstream())
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": MODEL},
        files={"file": ("a.wav", WAV, "audio/wav"), "temperature": ("t.txt", b"0.2", "text/plain")},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "temperature"
    assert deps.upstream.transcribed == []


def test_stream_false_is_fine(tmp_path):
    deps, client = app(tmp_path, HearingUpstream())
    r = client.post("/v1/audio/transcriptions", data={"model": MODEL, "stream": "false"}, files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 200


def test_transcription_backend_failure_is_a_502(tmp_path):
    deps, client = app(tmp_path, HearingUpstream(stt_error=UpstreamError(500, "down")))
    with pytest.raises(openai.InternalServerError) as exc:
        sdk(client).audio.transcriptions.create(model=MODEL, file=("clip.wav", io.BytesIO(WAV)))
    assert exc.value.code == "transcription_unavailable"


def test_transcription_unconfigured_returns_503_not_502(tmp_path):
    """Mirror of test_speech_unconfigured_returns_503_not_502 for the STT path.
    A missing STT address must surface as 503 'transcription_not_configured',
    not 502 'transcription_unavailable' (which the generic Exception handler
    used to emit, masking the contract change)."""
    deps, client = app(tmp_path, HearingUpstream(stt_error=UpstreamUnconfiguredError("speech-to-text is not configured: set STT_BASE_URL")))
    with pytest.raises(openai.APIStatusError) as exc:
        sdk(client).audio.transcriptions.create(model=MODEL, file=("clip.wav", io.BytesIO(WAV)))
    assert exc.value.status_code == 503
    assert exc.value.code == "transcription_not_configured"
    assert "STT_BASE_URL" in _error_message(exc.value.body)


def test_json_not_multipart_is_refused(tmp_path):
    deps, client = app(tmp_path, HearingUpstream())
    r = client.post("/v1/audio/transcriptions", content=json.dumps({"model": MODEL}), headers={"content-type": "application/json"})
    assert r.status_code == 400


@pytest.mark.parametrize("path", ["transcriptions", "translations"])
def test_audio_upload_is_bounded_before_it_reaches_upstream(tmp_path, monkeypatch, path):
    deps, client = app(tmp_path, Translating())
    monkeypatch.setattr(audio_api, "MAX_AUDIO_BYTES", len(WAV) - 1)

    r = client.post(f"/v1/audio/{path}", data={"model": MODEL},
                    files={"file": ("a.wav", WAV, "audio/wav")})

    assert r.status_code == 413
    assert r.json()["error"]["param"] == "file"
    assert r.json()["error"]["code"] == "file_too_large"
    assert deps.upstream.transcribed == []


# --- translations -----------------------------------------------------------------------

class Translating(HearingUpstream):
    async def complete(self, body):
        self.bodies.append(body)
        return ({"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Good morning, everyone."}}]},
                {"model-group": body["model"]})


def test_translation_is_transcribe_then_translate(tmp_path):
    deps, client = app(tmp_path, Translating(heard="Bonjour à tous."))
    t = sdk(client).audio.translations.create(model=MODEL, file=("clip.wav", io.BytesIO(WAV)))
    assert t.text == "Good morning, everyone."
    assert validate_payload({"text": t.text}, kind="translation", spec=Spec(), fields="strict")["verdict"] == "pass"
    call = deps.upstream.bodies[0]
    assert call["messages"][1]["content"] == "Bonjour à tous." and "into natural English" in call["messages"][0]["content"]
    assert call["temperature"] == 0


def test_translation_text_format_and_empty_audio(tmp_path):
    deps, client = app(tmp_path, Translating(heard=""))
    r = client.post("/v1/audio/translations", data={"model": MODEL, "response_format": "text"}, files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 200 and r.text == "" and deps.upstream.bodies == []     # nothing heard: nothing to translate


@pytest.mark.parametrize("data,param", [({"model": MODEL, "language": "fr"}, "language"),
                                        ({"model": MODEL, "stream": "true"}, "stream"),
                                        ({"model": MODEL, "response_format": "srt"}, "response_format")])
def test_translation_refusals(tmp_path, data, param):
    deps, client = app(tmp_path, Translating())
    r = client.post("/v1/audio/translations", data=data, files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_translation_backend_failure_is_a_502(tmp_path):
    class Fails(Translating):
        async def complete(self, body):
            raise UpstreamError(500, "down")
    deps, client = app(tmp_path, Fails(heard="Hola"))
    with pytest.raises(openai.InternalServerError) as exc:
        sdk(client).audio.translations.create(model=MODEL, file=("clip.wav", io.BytesIO(WAV)))
    assert exc.value.code == "translation_unavailable"


def test_a_list_response_format_is_a_named_400(tmp_path):
    """SPEECH_FORMATS is a dict and `in` hashes: a list where the spec says a
    string was an unhashable-type 500 (review 2026-09-22, #5)."""
    deps, client = app(tmp_path)
    r = client.post("/v1/audio/speech",
                    json={"model": "chord-1-poly", "input": "hi", "voice": "alloy",
                          "response_format": ["wav"]})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "response_format"


def test_translation_asks_the_persona_model_not_to_think(tmp_path):
    """Review 2026-09-24 A6: the persona backend thinks on every turn unless told not to
    (graph.thinking_switch); the translate call never said, so reasoning could spend the
    4096-token budget and leave content null."""
    deps, client = app(tmp_path, Translating(heard="Bonjour à tous."))
    sdk(client).audio.translations.create(model=MODEL, file=("clip.wav", io.BytesIO(WAV)))
    assert deps.upstream.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_speech_that_translates_to_nothing_is_a_502_not_an_empty_200(tmp_path):
    """Review 2026-09-24 A6: words were heard, so an empty translation is the backend
    failing (a spent reasoning budget, a null content), not silence."""
    class Empty(Translating):
        async def complete(self, body):
            self.bodies.append(body)
            return ({"choices": [{"index": 0, "finish_reason": "length",
                                  "message": {"role": "assistant", "content": None}}]}, {"model-group": body["model"]})
    deps, client = app(tmp_path, Empty(heard="Bonjour à tous."))
    r = client.post("/v1/audio/translations", data={"model": MODEL}, files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 502 and r.json()["error"]["code"] == "translation_unavailable"


# --- the MP3 sniff reads a frame header, not two magic bytes (review 2026-09-24 B18) ---

_PAD = b"\x00" * 64


@pytest.mark.parametrize("head", [
    b"ID3\x04\x00",            # ID3v2-tagged file
    b"\xff\xfb\x90\x64",       # MPEG-1 Layer III, no CRC
    b"\xff\xfa\x90\x64",       # MPEG-1 Layer III, with CRC
    b"\xff\xf3\x90\x64",       # MPEG-2 Layer III, no CRC
    b"\xff\xf2\x90\x64",       # MPEG-2 Layer III, with CRC
    b"\xff\xe3\x90\x64",       # MPEG-2.5 Layer III, no CRC
    b"\xff\xe2\x90\x64",       # MPEG-2.5 Layer III, with CRC
], ids=["id3", "mpeg1", "mpeg1-crc", "mpeg2", "mpeg2-crc", "mpeg25", "mpeg25-crc"])
def test_every_valid_mpeg_audio_frame_is_heard_as_mp3(tmp_path, head):
    deps, client = app(tmp_path, HearingUpstream())
    r = client.post("/v1/audio/transcriptions", data={"model": MODEL}, files={"file": ("a.mp3", head + _PAD, "audio/mpeg")})
    assert r.status_code == 200, r.text
    assert deps.upstream.transcribed[0][1:3] == ("mp3", "audio/mpeg")


@pytest.mark.parametrize("head", [
    b"\xff\xea\x90\x64",       # version bits 01: reserved
    b"\xff\xf9\x90\x64",       # layer bits 00: reserved
    b"\xff\xfb\xf0\x64",       # bitrate index 1111: invalid
    b"\xff\xfb\x9c\x64",       # sample-rate index 11: reserved
    b"\xff\x1b\x90\x64",       # sync is 11 bits, not 8
    b"\xff\xfb",               # too short to be a header
], ids=["version-reserved", "layer-reserved", "bitrate-bad", "rate-reserved", "short-sync", "truncated"])
def test_garbage_behind_an_ff_byte_is_still_not_mp3(tmp_path, head):
    deps, client = app(tmp_path, HearingUpstream())
    body = head + _PAD if len(head) > 2 else head
    r = client.post("/v1/audio/transcriptions", data={"model": MODEL}, files={"file": ("a.mp3", body, "audio/mpeg")})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "file"
    assert deps.upstream.transcribed == []


def test_speech_accepts_an_mpeg1_frame_with_crc(tmp_path):
    class CrcMp3(Formats):
        BYTES = {**Formats.BYTES, "mp3": b"\xff\xfa\x90\x64" + _PAD}
    deps, client = app(tmp_path, CrcMp3())
    audio = sdk(client).audio.speech.create(model=MODEL, input="hi", voice="alloy", response_format="mp3")
    assert audio.content == CrcMp3.BYTES["mp3"]


WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 60      # EBML header: Chrome's MediaRecorder
M4A = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 60  # ISO BMFF: Safari's MediaRecorder


@pytest.mark.parametrize("name,data,fmt,mime", [("clip.webm", WEBM, "webm", "audio/webm"),
                                                ("clip.m4a", M4A, "m4a", "audio/mp4")])
def test_browser_recordings_are_transcribed(tmp_path, name, data, fmt, mime):
    """Review 2026-09-24 B18: webm (Chrome) and mp4/m4a (Safari) were a 400, while
    the tested STT backend decodes both (live check 2026-09-24: a spoken clip as webm, m4a and
    mp4 all came back "Hello, this is a test.")."""
    deps, client = app(tmp_path, HearingUpstream(heard="hello there"))
    r = client.post("/v1/audio/transcriptions", data={"model": MODEL}, files={"file": (name, data)})
    assert r.status_code == 200, r.text
    assert deps.upstream.transcribed == [(data, fmt, mime, "example-stt")]


def test_artifact_registration_accepts_every_mp3_the_speech_check_accepts(tmp_path):
    """The speech door and the artifact store sniffed the same TTS bytes with two
    different checks; widening only the first would let an MPEG-1-with-CRC clip pass
    the format check and then fail registration. One shared check (review 2026-09-24 B18)."""
    from chord.artifacts import ArtifactStore
    clip = b"\xff\xfa\x90\x00" + b"\x00" * 64          # MPEG-1 layer III with CRC
    assert ArtifactStore(tmp_path).register(clip, "audio/mpeg").mime == "audio/mpeg"


def test_a_three_byte_body_is_not_an_mpeg_frame():
    """An MPEG audio frame header is four bytes; three valid-looking ones are
    not a frame (Copilot on #331, review 2026-09-24 B18)."""
    from chord.artifacts import looks_like_mp3
    assert looks_like_mp3(b"\xff\xfb\x90") is False
    assert looks_like_mp3(b"\xff\xfb\x90\x64") is True
    assert looks_like_mp3(b"ID3") is True
