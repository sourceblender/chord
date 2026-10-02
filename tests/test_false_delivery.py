"""Derived #80 false-delivery fixture: turn 2 needed an answer, but the model
claimed an image was ready by copying turn 1's reply shape. The private capture
was replayed on the chat backend: 10/10 with the note, 7/10 with a stronger one.
On a turn that delivers nothing, hold the whole reply and replace a delivery
claim with a plain line that asks the real question."""
import json

import pytest
from fastapi.testclient import TestClient

from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.image_caption import claims_delivery
from chord.server import Deps, create_app, load_specialists
from test_progress import FixedRouter, last_trace
from test_skeleton import AvailableImageBackend, FakeUpstream

load_specialists()
LIE = ("Here you go — Ava putting in some real work at the gym. There's something focused about that kind of energy, "
       "the kind where it's just you, the weights, and whatever you're listening to.\n\nI pictured her somewhere "
       "mid-set, really engaged in it. Want me to change anything?")               # Derived from the private response-2
HONEST = "I haven't started on that one yet! What should it show — Ava lifting, or on the treadmill?"
Q = "What should this new picture show?"


class Saying(FakeUpstream):
    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    async def complete(self, body):
        self.bodies.append(body)
        return ({"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": self.reply}}]}, {})

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for i in range(0, len(self.reply), 7):
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": self.reply[i:i + 7]}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def app(tmp_path, monkeypatch, reply, status=Outcome.needs_clarification, router=FixedRouter, routes=("image",)):
    async def image(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=status, question=Q, summary="no render")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    settings = Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset(routes))
    return TestClient(create_app(Deps(settings, upstream=Saying(reply), model=lambda n: router(),
                                      image_backend=AvailableImageBackend() if "image" in routes else None))), settings


def ask(c, stream):
    body = {"model": "chord-1-poly", "stream": stream, "messages": [{"role": "user", "content": "draw Ava at the gym"}]}
    if not stream:
        return c.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["content"], None
    with c.stream("POST", "/v1/chat/completions", json=body) as r:
        chunks = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]
    parts = [c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices")]
    return "".join(parts), parts


@pytest.mark.parametrize("stream", [False, True])
def test_the_incident_reply_is_replaced_and_never_streamed(tmp_path, monkeypatch, stream):
    c, settings = app(tmp_path, monkeypatch, LIE)
    text, parts = ask(c, stream)
    assert text == f"I haven't made that one yet. {Q}"
    if parts is not None:
        assert not any("Here you go" in p or "Ava" in p for p in parts)     # no fragment of the lie ever left
    assert last_trace(settings)["false_delivery_claim_replaced"] is True


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("honest", [HONEST,
    "I'm not sending it now. Which outfit should I use?",
    "I haven't made it. You saw 'Here you go' in my previous reply; which outfit do you want?",
    "Here's your question back: which outfit should Ava wear?",
    "You asked me to say `Here you go`, but I haven't made it yet.",                          # #105 round 3
    "> Here you go — Ava at the gym.\n\nThat was the last one. I haven't made this one yet: gym or park?"])
def test_an_honest_reply_is_kept_in_her_words(tmp_path, monkeypatch, stream, honest):
    HONEST_ = honest
    c, settings = app(tmp_path, monkeypatch, HONEST_)
    text, _ = ask(c, stream)
    assert text == HONEST_ and last_trace(settings)["false_delivery_claim_replaced"] is False



@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_render_claiming_delivery_is_replaced(tmp_path, monkeypatch, stream):
    c, settings = app(tmp_path, monkeypatch, "Here it is! Ava mid-squat.", status=Outcome.failed)
    text, _ = ask(c, stream)
    assert text == "That one didn't come through this time, so there's nothing attached."


class AudioRouter:
    async def ainvoke(self, msgs):
        class R: content = '{"route": "audio", "intent": "a voice note"}'
        return R()


def test_an_unavailable_capability_claiming_delivery_is_replaced(tmp_path, monkeypatch):
    c, settings = app(tmp_path, monkeypatch, "Here you go, I've sent it! Good morning!", router=AudioRouter, routes=("image",))
    text, _ = ask(c, False)
    assert text == "I can't send voice messages or audio clips yet, so there's nothing attached this time."


def test_a_completed_picture_is_not_guarded(tmp_path, monkeypatch):
    from test_skeleton import PNG

    async def made(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="Ava")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", made)
    settings = Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset({"image"}))
    c = TestClient(create_app(Deps(settings, upstream=Saying("Here you go — Ava at the gym."),
                                   model=lambda n: FixedRouter(), image_backend=AvailableImageBackend())))
    text, _ = ask(c, False)
    assert text.startswith("Here you go — Ava at the gym.") and "false_delivery_claim_replaced" not in last_trace(settings)


@pytest.mark.parametrize("text,claims", [
    ("Here you go — Ava at the gym.", True),
    ("Here it is!", True),
    ("Here's your picture.", True),
    ("I pictured her mid-set.", True),
    ("I've attached it below.", True),
    ("I made one for you.", True),
    ("Here's Ava putting in the work at the gym.", True),               # 5/40 replay lies were this shape
    ("here's what I think we should do", False),
    ("Here's hoping! Who should it show?", False),
    ("I'm not sending it now. Which outfit should I use?", False),                               # #105
    ("I haven't made it. You saw 'Here you go' in my previous reply; which outfit do you want?", False),
    ("Here's your question back: which outfit should Ava wear?", False),
    ("Here's your picture! I haven't made the next one.", True),     # a later negation doesn't excuse an earlier claim
    # A negation only voids a claim in its own clause: an idiom before a comma or dash
    # ("no problem", "couldn't resist") is not a denial (#105).
    ("No problem, here's your picture of Ava at the gym!", True),
    ("No worries — here you go, Ava mid-set!", True),
    ("Couldn't resist, here it is: Ava at the gym.", True),
    ("I didn't want to keep you waiting, so here's your picture!", True),
    ("Not gonna lie, here's Ava looking amazing.", True),
    ("I haven't sent anything yet. Here's my question: gym or park?", False),
    ("Nothing's attached yet — what should it show?", False),
    ("I'm not re-sending it now, which one did you mean?", False),  # a hyphen is not a clause break
    # ...but a denial OF delivery still covers the whole sentence, across a comma.
    ("I can't send it yet, here it is in words: Ava mid-lift.", False),
    ("I haven't made it, here you go with a question instead: gym or park?", False),
    ("It isn't ready yet — here's the idea: Ava at the gym?", False),
    # Markdown quoting is quoting (#105 round 3): inline code and blockquote lines are not her claim.
    ("You asked me to say `Here you go`, but I haven't made it yet.", False),
    ("> Here you go — Ava at the gym.\n\nThat was the last one. I haven't made this one yet: gym or park?", False),
    ("Here you go!\n> (your last request)", True),                   # a quote after her own claim doesn't excuse it
    ("Here's the thing, I haven't made it yet.", False),
    ("I made a note of that. What should it show?", False),
    ("I haven't made it yet — who should it show?", False),
])
def test_claims_delivery(text, claims):
    assert claims_delivery(text) is claims


@pytest.mark.parametrize("size", [1, 2, 3, 5, 11, 40])
def test_held_reply_is_the_same_whatever_the_chunking(tmp_path, monkeypatch, size):
    reply = "I haven't made it. You saw 'Here you go' in my previous reply; which outfit do you want?"

    class Chunked(Saying):
        async def stream(self, body):
            yield None, {}
            for i in range(0, len(reply), size):
                yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": reply[i:i + size]}, "finish_reason": None}]}, {}
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}

    async def image(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.needs_clarification, question=Q)
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    settings = Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset({"image"}))
    c = TestClient(create_app(Deps(settings, upstream=Chunked(reply), model=lambda n: FixedRouter(),
                                   image_backend=AvailableImageBackend())))
    text, parts = ask(c, True)
    assert text == reply
