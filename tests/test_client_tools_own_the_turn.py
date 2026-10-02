"""S04 (red team pass 1; 2026-09-16): in Chat Completions the caller's
`tools` are the caller's. Under `auto` (or no tool_choice, or "none") the model
answers or calls THEIR functions; the spec defines no hidden server tool except
search, and that only when `web_search_options` asks. So a request that
declares tools never runs a service specialist, whatever the router would have
said, and any calls the model returns reach the caller exactly as sent."""
import json

import pytest
from fastapi.testclient import TestClient

from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app, load_specialists
from test_progress import last_trace
from test_skeleton import AvailableImageBackend, FakeUpstream

load_specialists()

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
LEGACY = {"name": "get_weather", "parameters": {"type": "object"}}
CALLS = [
    {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
    {"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}},
]


def router_for(route):
    class Router:
        async def ainvoke(self, msgs):
            class R: content = json.dumps({"route": route, "intent": "the weather in Paris and Oslo"})
            return R()
    return Router


class Calling(FakeUpstream):
    """The model calls the client's function twice, in parallel."""

    async def complete(self, body):
        self.bodies.append(body)
        message = {"role": "assistant", "content": None, "tool_calls": [{k: v for k, v in c.items() if k != "index"} for c in CALLS]}
        return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]}, {}

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for call in CALLS:
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, {}


def app(tmp_path, monkeypatch, route):
    ran = []

    async def specialist(job, ctx):
        ran.append(job)
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.failed, summary="should not run")

    for cap in ("image", "search", "audio"):
        monkeypatch.setitem(specialists.SPECIALISTS, cap, specialist)
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        experimental_routes=frozenset({"image", "search", "audio"}))
    up = Calling()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: router_for(route)(),
                                      image_backend=AvailableImageBackend()))), up, settings, ran


def returned_calls(client, stream, body):
    if not stream:
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        return choice["message"].get("tool_calls"), choice["message"].get("content"), [choice["finish_reason"]]
    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
        assert r.status_code == 200
        chunks = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]
    choices = [c for chunk in chunks for c in chunk.get("choices") or []]
    calls = [call for c in choices for call in (c.get("delta") or {}).get("tool_calls") or []]
    content = "".join((c.get("delta") or {}).get("content") or "" for c in choices)
    return calls, content or None, [c["finish_reason"] for c in choices if c.get("finish_reason")]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("route", ["image", "search", "audio", "clarify"])
@pytest.mark.parametrize("offer", [
    {"tools": [WEATHER]},
    {"tools": [WEATHER], "tool_choice": "auto"},
    {"functions": [LEGACY]},
], ids=["tools", "auto", "legacy-functions"])   # tool_choice "none" has its own test below: its tools are not forwarded
def test_declared_client_tools_own_the_turn(tmp_path, monkeypatch, stream, route, offer):
    client, up, settings, ran = app(tmp_path, monkeypatch, route)
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "weather in Paris and Oslo?"}], **offer}
    calls, content, finishes = returned_calls(client, stream, body)
    assert ran == []                                               # no specialist, whatever the router thinks
    t = last_trace(settings)
    assert t["route_decision"] == "chat" and t["router"] == "skipped_client_tools"
    sent = up.bodies[-1]
    for key, value in offer.items():
        assert sent[key] == value                                  # the client's tools reach the model untouched
    assert sent["messages"][-1] == body["messages"][-1]            # no service note in the user's turn
    expected = CALLS if stream else [{k: v for k, v in c.items() if k != "index"} for c in CALLS]
    assert calls == expected and content is None and finishes == ["tool_calls"]   # both calls, byte for byte, unwrapped


def test_without_declared_tools_the_router_still_decides(tmp_path, monkeypatch):
    client, up, settings, ran = app(tmp_path, monkeypatch, "image")
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a mug"}]})
    assert len(ran) == 1 and last_trace(settings)["route_decision"] == "image"


def test_web_search_options_is_the_one_server_tool_and_still_searches_beside_client_tools(tmp_path, monkeypatch):
    client, up, settings, ran = app(tmp_path, monkeypatch, "chat")
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "tools": [WEATHER], "web_search_options": {},
                                              "messages": [{"role": "user", "content": "news today?"}]})
    assert last_trace(settings)["router"] == "forced_by_web_search_options" and len(ran) == 1


# S-cf-045 (independent S04 gate on prod 0c0fb79, 2026-09-17): tool_choice "none" with tools in the
# prompt came back EMPTY: the backend wrote a tool call anyway and its parser removed it.
@pytest.mark.parametrize("extra", [{"tool_choice": "none"}, {"function_call": "none", "functions": [{"name": "f", "parameters": {"type": "object"}}]}])
def test_tool_choice_none_forwards_no_tools_and_is_still_never_routed(tmp_path, extra):
    from fastapi.testclient import TestClient
    from chord.config import Settings
    from chord.server import Deps, create_app
    from test_skeleton import FakeUpstream

    class Router:
        async def ainvoke(self, msgs):
            raise AssertionError("a request that declares tools is never routed")

    up = FakeUpstream()
    client = TestClient(create_app(Deps(Settings(data_dir=tmp_path, router_enabled=True,
                                                 experimental_routes=frozenset({"search", "image"})), upstream=up, model=lambda n: Router())))
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}], **extra}
    if "functions" in extra:
        body.pop("tools")
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    sent = up.bodies[-1]
    assert not ({"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls"} & sent.keys()), sorted(sent)
