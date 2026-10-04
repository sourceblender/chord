"""#82 (2026-09-13): asked for a voice clip the service can't make, told
so, she still wrote "(audio clip playing)". On turns that deliver nothing,
pretend-delivery narration is removed; ordinary parentheses are not."""
import pytest

from chord.image_caption import FakeMediaNarration


def run(text, chunks=None):
    f = FakeMediaNarration()
    if chunks is None:
        return f.feed(text, final=True), f.removed
    out = "".join(f.feed(c) for c in chunks) + f.feed("", final=True)
    return out, f.removed


@pytest.mark.parametrize("text,expected", [
    ("Sure! Here's a little good morning for you:\n\n*(audio clip playing)*\n\nGood morning! ☀️",
     "Sure! Here's a little good morning for you:\n\n\n\nGood morning! ☀️"),          # operator's exact reply
    ("Here you go (voice note attached) love you", "Here you go  love you"),
    ("[audio: good morning, playing]", ""),
    ("(🔊 playing your clip)", ""),
    ("Here's the video (video attached)!", "Here's the video !"),
    ("(voice message sent)", ""),
    ("*(🎵 song playing)*", ""),
    ("(photos attached)", ""),
])
def test_pretend_delivery_is_removed(text, expected):
    out, n = run(text)
    assert out == expected and n == 1


@pytest.mark.parametrize("text", [
    "I can't send voice clips yet (sorry!), but here it is in words: good morning.",
    "That song (the one from last night) is stuck in my head.",
    "I'd love to play it for you (maybe soon).",
    "Check the docs [1] for details.",
    "a (b) c [d] e",
    "(audio is not playing)",                      # Regression: the honest statements stay
    "(the image was not sent)",
    "(audio playback is unavailable)",
    "*(voice clips can't be sent yet)*",
    "(no audio attached)",
    "(the image represents a cat)",                # Regression: "sent" inside "represents"
    "(audio display settings)",                    # "play" inside "display"
    "(a photo of a record player)",                # "play" inside "player"
    "(my profile was sentimental)",                # "file" in "profile", "sent" in "sentimental"
    "(an eclipse, replaying in my head)",          # "clip" in "eclipse", "play" in "replaying"
])
def test_ordinary_parentheses_pass_through(text):
    out, n = run(text)
    assert out == text and n == 0


@pytest.mark.parametrize("text", [
    "Look: (the image represents a cat) and (audio display settings) and (a photo of a record player).",
    "Morning! *(the image represents a cat)* ok",
])
def test_ordinary_words_survive_every_split(text):
    assert run(text) == (text, 0)
    for i in range(1, len(text)):
        assert run(text, [text[:i], text[i:]]) == (text, 0), i
    assert run(text, list(text)) == (text, 0)


def test_streaming_split_anywhere_gives_the_same_result():
    text = "Sure!\n\n*(audio clip playing)*\n\nGood morning (really)."
    whole, n = run(text)
    for i in range(1, len(text)):
        out, m = run(text, [text[:i], text[i:]])
        assert (out, m) == (whole, n), i
    chars, m = run(text, list(text))
    assert (chars, m) == (whole, n)


@pytest.mark.parametrize("text", ["**(audio clip playing)**", "*(audio clip playing)*", "__[voice note attached]__",
                                  "Here: **(audio clip playing)** Good morning!"])
def test_every_emphasis_split_gives_the_same_result(text):
    """Regression: a split between the closing stars left an orphan '*'."""
    whole, n = run(text)
    assert n == 1 and "*" not in whole and "_" not in whole
    for i in range(1, len(text)):
        assert run(text, [text[:i], text[i:]]) == (whole, n), i


def test_an_unclosed_parenthesis_is_released_not_swallowed():
    out, n = run("It was great (really", chunks=["It was great (", "really"])
    assert out == "It was great (really" and n == 0


# --- the whole graph, operator's exact turn ------------------------------------------


from fastapi.testclient import TestClient  # noqa: E402

from chord.config import Settings  # noqa: E402
from chord.server import Deps, create_app, load_specialists  # noqa: E402
from test_progress import content_of, last_trace, stream_chunks  # noqa: E402
from test_skeleton import AvailableImageBackend, FakeUpstream  # noqa: E402

load_specialists()
FAKED = "Sure! Here's a little good morning for you:\n\n*(audio clip playing)*\n\nGood morning! ☀️"


class AudioRouter:
    async def ainvoke(self, msgs):
        class R: content = '{"route": "audio", "intent": "Create a voice message saying \'good morning\'"}'
        return R()


class FakingUpstream(FakeUpstream):
    """Replies the way she did on 2026-09-13 04:15Z."""

    async def complete(self, body):
        self.bodies.append(body)
        return ({"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": FAKED}}],
                 "usage": {"total_tokens": 9}}, {})

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for piece in [FAKED[:40], FAKED[40:52], FAKED[52:]]:   # the aside split across chunks
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def audio_client(tmp_path):
    settings = Settings(data_dir=tmp_path, router_enabled=True)
    up = FakingUpstream()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: AudioRouter()))), up, settings


@pytest.mark.parametrize("stream", [False, True])
def test_an_unavailable_voice_clip_is_never_narrated_as_playing(tmp_path, stream):
    client, up, settings = audio_client(tmp_path)
    ask = "Can you send me a audio clip saying good morning?"
    if stream:
        text = content_of(stream_chunks(client, ask))
    else:
        r = client.post("/v1/chat/completions", json={"model": "chord-1-poly",
                                                     "messages": [{"role": "user", "content": ask}]})
        text = r.json()["choices"][0]["message"]["content"]
    assert "audio clip playing" not in text and "Good morning!" in text
    note = up.bodies[0]["messages"][0]["content"]
    assert "you can't send voice messages or audio clips" in note and "Don't pretend it is playing" in note
    t = last_trace(settings)
    assert t["route_unavailable"] == "audio" and t["fake_media_narration_removed"] == 1


def test_a_completed_picture_turn_is_not_touched_by_the_fake_filter(tmp_path, monkeypatch):
    """The filter runs only when nothing is delivered."""
    from chord import specialists
    from chord.contract import Outcome, Result
    from test_progress import FixedRouter
    from test_skeleton import PNG

    async def made(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a mug")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", made)
    settings = Settings(data_dir=tmp_path, router_enabled=True, enabled_routes=frozenset({"image"}))
    TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                               image_backend=AvailableImageBackend()))).post(
        "/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a mug"}]})
    assert "fake_media_narration_removed" not in last_trace(settings)


class VideoRouter:
    async def ainvoke(self, msgs):
        class R: content = '{"route": "video", "intent": "a video of her dancing"}'
        return R()


def test_a_video_ask_is_said_plainly_unavailable_not_made_into_a_picture(tmp_path):
    """2026-09-13: "make me a video" routed to image and made a still. Video
    is out of scope, so the ask is named and answered honestly (#90)."""
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        enabled_routes=frozenset({"image"}))
    up = FakeUpstream()
    TestClient(create_app(Deps(settings, upstream=up, model=lambda n: VideoRouter()))).post(
        "/v1/chat/completions", json={"model": "chord-1-poly",
                                      "messages": [{"role": "user", "content": "make me a video of you dancing"}]})
    t = last_trace(settings)
    assert t["route_unavailable"] == "video" and "specialist" not in t
    assert "you can't make videos, GIFs or animations" in up.bodies[0]["messages"][0]["content"]
