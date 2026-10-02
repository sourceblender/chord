"""An HTTP lane classifier as chord's router backend (ROUTER_BACKEND=classifier).

It returns a lane only: chat turns never wait on the router model, and on a
specialist lane the router model still writes the brief. A classifier that can't
be reached must not cost a turn its route.
"""
import functools
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from chord import router as router_mod
from chord import specialists
from chord.config import ConfigurationError, Settings
from chord.server import Deps, create_app, load_specialists

from test_skeleton import AvailableImageBackend, FakeUpstream

load_specialists()
URL = "http://router.test:8088/v1/route"


class BriefRouter:
    """The router model. Asked for a brief only; its own route must never win."""
    calls = 0
    route = "chat"          # deliberately disagrees with the classifier's image lane
    fail = False

    async def ainvoke(self, msgs):
        if "You check one reply" in str(msgs[0].content):   # the #109 delivery check, not routing
            class N: content = '{"claims_delivery": false}'
            return N()
        BriefRouter.calls += 1
        if BriefRouter.fail:
            raise httpx.ConnectError("router model down")
        class R: content = json.dumps({"route": BriefRouter.route, "intent": "a mug on a desk",
                                       "constraints": ["blue"], "latitude": "style is hers"})
        return R()


def classifier_service(reply):
    """An httpx transport standing in for the classifier service. Records the text it was sent."""
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["text"])
        return reply() if callable(reply) else reply
    return httpx.MockTransport(handle), seen


def app(tmp_path, monkeypatch, transport):
    started = []

    async def image(job, ctx):
        started.append(job)
        raise RuntimeError("stop after the handoff")      # the test only needs the job it was given

    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    monkeypatch.setattr(router_mod, "classify", functools.partial(router_mod.classify, transport=transport))
    BriefRouter.calls, BriefRouter.route, BriefRouter.fail = 0, "chat", False
    settings = Settings(data_dir=tmp_path, router_enabled=True, router_backend="classifier",
                        router_classifier_url=URL, experimental_routes=frozenset({"image"}))
    up = FakeUpstream()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: BriefRouter(),
                                      image_backend=AvailableImageBackend()))), settings, started


def last_trace(settings):
    lines = [l for p in sorted(Path(settings.trace_dir).glob("*.jsonl")) for l in p.read_text().splitlines()]
    return json.loads(lines[-1])


def say(client, text, history=()):
    msgs = [*history, {"role": "user", "content": text}]
    return client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": msgs})


def lane(route, p=0.97):
    return httpx.Response(200, json={"route": route, "probabilities": {route: p}, "engine_sha256": "03063c27" + "0" * 56})


def test_a_chat_lane_never_waits_on_the_router_model(tmp_path, monkeypatch):
    transport, seen = classifier_service(lane("chat"))
    client, settings, started = app(tmp_path, monkeypatch, transport)
    assert say(client, "any tips for drawing hands?").status_code == 200
    assert BriefRouter.calls == 0 and started == []       # the shortcut this guards: a brief call on every turn
    t = last_trace(settings)
    assert t["route_decision"] == "chat" and t["router_backend"] == "classifier"
    assert t["classifier_route"] == "chat" and t["classifier_confidence"] == 0.97
    assert t["classifier_engine_sha256"].startswith("03063c27")


def test_a_specialist_lane_is_the_classifiers_and_the_brief_is_the_router_models(tmp_path, monkeypatch):
    transport, seen = classifier_service(lane("image"))
    client, settings, started = app(tmp_path, monkeypatch, transport)
    say(client, "draw me a blue mug")
    assert BriefRouter.calls == 1
    assert len(started) == 1                              # the router model said chat; the classifier's lane won
    job = started[0]
    assert (job.intent, job.constraints, job.latitude) == ("a mug on a desk", ["blue"], "style is hers")
    t = last_trace(settings)
    assert t["route_decision"] == "image" and t["brief_router_route"] == "chat"


def test_a_failed_brief_still_starts_the_specialist_with_her_words(tmp_path, monkeypatch):
    transport, _ = classifier_service(lane("image"))
    client, settings, started = app(tmp_path, monkeypatch, transport)
    BriefRouter.fail = True
    say(client, "draw me a blue mug")
    assert len(started) == 1 and started[0].intent == "draw me a blue mug"
    assert last_trace(settings)["router_call_error"].startswith("router call failed")


@pytest.mark.parametrize("reply,error", [
    (httpx.Response(503, json={"error": "warming"}), "classifier call failed (HTTPStatusError)"),
    (httpx.Response(200, json={"route": "clarify"}), "unknown classifier route 'clarify'"),
    (httpx.Response(200, text="not json"), "classifier call failed (JSONDecodeError)"),
    # Unhashable routes: a frozenset membership test raises TypeError on these (Copilot, #327).
    (httpx.Response(200, json={"route": []}), "unknown classifier route []"),
    (httpx.Response(200, json={"route": {}}), "unknown classifier route {}"),
    (httpx.Response(200, json=["image"]), "unknown classifier route None"),
])
def test_an_unusable_classifier_hands_the_turn_to_the_router_model(tmp_path, monkeypatch, reply, error):
    transport, _ = classifier_service(reply)
    client, settings, started = app(tmp_path, monkeypatch, transport)
    BriefRouter.route = "image"
    say(client, "draw me a blue mug")
    assert BriefRouter.calls == 1 and len(started) == 1    # routed, not dropped to chat
    t = last_trace(settings)
    assert t["classifier_error"] == error and t["router_fallback"].startswith("router model")
    assert t["route_decision"] == "image"


def test_an_unreachable_classifier_hands_the_turn_to_the_router_model(tmp_path, monkeypatch):
    def refuse():
        raise httpx.ConnectError("connection refused")
    transport, _ = classifier_service(refuse)
    client, settings, started = app(tmp_path, monkeypatch, transport)
    BriefRouter.route = "image"
    say(client, "draw me a blue mug")
    t = last_trace(settings)
    assert t["classifier_error"] == "classifier call failed (ConnectError)" and len(started) == 1


def test_the_classifier_reads_exactly_the_router_transcript(tmp_path, monkeypatch):
    """Its training rows were built with this format (router-train/build.py copies _transcript)."""
    transport, seen = classifier_service(lane("chat"))
    client, _, _ = app(tmp_path, monkeypatch, transport)
    # Mixed case and punctuation on purpose: an all-lowercase fixture let a .lower() drift pass (red-proof, 2026-09-23).
    history = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "Hi Mei!"},
               {"role": "assistant", "content": "Hey you."}]
    say(client, "Any tips for drawing HANDS?", history)
    assert seen == ["user: Hi Mei!\nassistant: Hey you.\nuser: Any tips for drawing HANDS?"]


def test_classifier_backend_config_is_checked_at_startup():
    base = dict(router_enabled=True)
    with pytest.raises(ConfigurationError) as raised:
        Settings(**base, router_backend="classifier", router_classifier_url="").validate_startup()
    assert "ROUTER_BACKEND=classifier needs ROUTER_CLASSIFIER_URL" in str(raised.value)
    with pytest.raises(ConfigurationError) as raised:
        Settings(**base, router_backend="remote").validate_startup()
    assert "ROUTER_BACKEND must be llm or classifier" in str(raised.value)
    with pytest.raises(ConfigurationError) as raised:
        Settings(**base, router_backend="classifier", router_classifier_url="router.example.test").validate_startup()
    assert "ROUTER_CLASSIFIER_URL must be an absolute http(s) URL" in str(raised.value)
