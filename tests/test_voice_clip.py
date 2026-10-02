"""Voice messages on the spec's wire. #82b (2026-09-13): "Model acted like it
was going to send a audio clip and didnt." Until 2026-09-16 a routed voice ask got
an <audio> block appended to her text for Open WebUI. The public wire is now the
pinned spec (2026-09-16): a voice reply is message.audio when modalities asks
for audio, and otherwise she says plainly that she can't send one here."""
import base64
import json

import pytest
from fastapi.testclient import TestClient

from chord import voice_clip
from chord.config import Settings
from chord.server import Deps, create_app, load_specialists
from test_audio_output import SPOKEN_MP3, SpeakingUpstream
from test_progress import last_trace

load_specialists()
ASK = "Can you send me a audio clip saying good morning?"


class AudioRouter:
    async def ainvoke(self, msgs):
        class R: content = '{"route": "audio", "intent": "a voice message saying good morning"}'
        return R()


class Replying(SpeakingUpstream):
    REPLY = "Good morning, sunshine! Hope your day starts gently."

    async def complete(self, body):
        self.bodies.append(body)
        return ({"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": self.REPLY}}]}, {})

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for piece in [self.REPLY[:20], self.REPLY[20:]]:
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def app(tmp_path):
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        experimental_routes=frozenset({"audio"}))
    up = Replying()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: AudioRouter()))), up, settings


def post(c, stream=False, messages=None, **extra):
    body = {"model": "chord-1-poly", "stream": stream, "messages": messages or [{"role": "user", "content": ASK}], **extra}
    if not stream:
        r = c.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, r.text
        return r.json()["choices"][0]["message"]
    with c.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        chunks = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]
    return {"content": "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))}


@pytest.mark.parametrize("stream", [False, True])
def test_a_voice_ask_without_audio_output_is_answered_honestly_in_text(tmp_path, stream):
    c, up, settings = app(tmp_path)
    message = post(c, stream)
    assert "<audio" not in message["content"] and "audio" not in message and up.said == []
    note = up.bodies[0]["messages"][0]["content"]
    assert "can't send voice messages or audio clips" in note and "Your reply will be spoken aloud" not in note
    t = last_trace(settings)
    assert t["route_unavailable"] == "audio" and t["reason"] == "audio_output_not_requested"


def test_a_voice_ask_with_audio_output_is_spoken_as_message_audio(tmp_path):
    c, up, settings = app(tmp_path)
    message = post(c, modalities=["text", "audio"], audio={"voice": "alloy", "format": "mp3"})
    assert base64.b64decode(message["audio"]["data"]) == SPOKEN_MP3
    assert message["content"] == Replying.REPLY and "<audio" not in message["content"]
    assert "Your reply will be spoken aloud" in up.bodies[0]["messages"][0]["content"]
    assert last_trace(settings)["route_decision"] == "audio"


def clip_block(b64):  # the form we appended before 2026-09-16, still in old conversations
    return f"\n\n<audio controls>\ndata:audio/mpeg;base64,{b64}\n</audio>\n"


def test_our_old_clip_sent_back_is_a_marker_not_base64(tmp_path):
    c, up, settings = app(tmp_path)
    b64 = base64.b64encode(SPOKEN_MP3).decode()
    history = [{"role": "user", "content": ASK},
               {"role": "assistant", "content": "Morning!" + clip_block(b64)},
               {"role": "user", "content": "aww thank you"}]
    post(c, messages=history)
    sent = json.dumps(up.bodies[0]["messages"])
    assert b64 not in sent and voice_clip.MARKER in sent and "Morning!" in sent


def test_a_user_message_with_an_audio_tag_is_untouched(tmp_path):
    """Only assistant messages are rewritten, as with images."""
    c, up, settings = app(tmp_path)
    text = "look:" + clip_block("QUJD")
    post(c, messages=[{"role": "user", "content": text}])
    assert up.bodies[0]["messages"][-1]["content"] == text


def test_a_client_with_its_own_tts_gets_no_clip_from_us(tmp_path):
    c, up, settings = app(tmp_path)
    tts = {"type": "function", "function": {"name": "tts", "parameters": {"type": "object"}}}
    r = c.post("/v1/chat/completions", json={"model": "chord-1-poly", "tools": [tts],
                                             "messages": [{"role": "user", "content": ASK}]})
    assert "<audio" not in r.json()["choices"][0]["message"]["content"] and up.said == []
    assert last_trace(settings)["router"] == "skipped_client_tools"
