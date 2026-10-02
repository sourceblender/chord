"""#109: the false-delivery guard, structurally. The phrase list missed contrast
and time-change wording; on turns where nothing was made the router model now
reads the held reply for a delivery claim however it is worded."""
import json

import pytest
from fastapi.testclient import TestClient

from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app, create_internal_app, load_specialists
from test_skeleton import AvailableImageBackend, FakeUpstream

load_specialists()
SLIPPED = [  # #105's named limit: each passed the phrase list
    "I couldn't wait to send it — take a look!",
    "I wasn't done before, but it's ready for you now!",
    "I never expected to make this, but enjoy the view!",
]


class Model:
    """Routes to image; as the delivery checker, answers from a fixed verdict."""
    def __init__(self, verdict):
        self.verdict, self.checked = verdict, []

    async def ainvoke(self, msgs):
        class R: pass
        if "You check one reply" in msgs[0].content:
            self.checked.append(msgs[1].content)
            if isinstance(self.verdict, Exception):
                raise self.verdict
            R.content = json.dumps({"claims_delivery": self.verdict})
        else:
            R.content = '{"route": "image", "intent": "a mug"}'
        return R()


def run(tmp_path, monkeypatch, reply, verdict, stream, status=Outcome.failed):
    async def image(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=status, summary="render failed")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)

    class Voice(FakeUpstream):
        async def complete(self, body):
            data, dep = await super().complete(body)
            data["choices"][0]["message"]["content"] = reply
            return data, dep

        async def stream(self, body):
            yield None, {}
            for piece in (reply[:10], reply[10:]):
                yield {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}, {}
            yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}

    model = Model(verdict)
    deps = Deps(Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset({"image"})),
                upstream=Voice(), model=lambda n: model)
    client = TestClient(create_app(deps))
    body = {"model": "chord-1-poly", "stream": stream, "messages": [{"role": "user", "content": "draw a mug"}]}
    if stream:
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            chunks = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: {")]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        rid = r.headers["x-request-id"]
    else:
        r = client.post("/v1/chat/completions", json=body)
        text, rid = r.json()["choices"][0]["message"]["content"], r.headers["x-request-id"]
    return text, TestClient(create_internal_app(deps)).get(f"/internal/traces/{rid}").json(), model


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply", SLIPPED)
def test_a_claim_the_phrases_miss_is_caught_by_the_check(tmp_path, monkeypatch, reply, stream):
    text, trace, model = run(tmp_path, monkeypatch, reply, True, stream)
    assert text != reply and "nothing attached" in text
    assert trace["false_delivery_check"] == "model" and trace["false_delivery_claim_replaced"] is True
    assert model.checked == [reply]


@pytest.mark.parametrize("stream", [False, True])
def test_an_honest_reply_goes_out_unchanged(tmp_path, monkeypatch, stream):
    reply = "The render failed this time, sorry. Want me to try again?"
    text, trace, model = run(tmp_path, monkeypatch, reply, False, stream)
    assert text == reply and trace["false_delivery_claim_replaced"] is False


def test_the_phrase_list_still_answers_first_without_a_model_call(tmp_path, monkeypatch):
    text, trace, model = run(tmp_path, monkeypatch, "Here you go!", False, False)
    assert "nothing attached" in text and trace["false_delivery_check"] == "phrase" and model.checked == []


def test_a_check_that_fails_changes_nothing(tmp_path, monkeypatch):
    text, trace, model = run(tmp_path, monkeypatch, SLIPPED[0], RuntimeError("router down"), False)
    assert text == SLIPPED[0] and trace["false_delivery_check"].startswith("unavailable")


def test_a_completed_turn_is_never_checked(tmp_path, monkeypatch):
    from test_skeleton import PNG

    async def image(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")], summary="a mug")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    model = Model(True)
    client = TestClient(create_app(Deps(Settings(data_dir=tmp_path, router_enabled=True,
                                                 experimental_routes=frozenset({"image"})), upstream=FakeUpstream(),
                                      model=lambda n: model, image_backend=AvailableImageBackend())))
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a mug"}]})
    assert model.checked == []
