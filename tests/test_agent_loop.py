"""#84 and #83: a client that runs its own agent loop (OpenClaw) gets no service
notes in the middle of it. Red-team run b2 (2026-09-13): a service "ask the user"
note on her first request put third-person planning in her reply, and the planning
in her history made her narrate after a send instead of NO_REPLY. The router also
ran again on her tool-result continuation and truncated its JSON."""
import pytest
from fastapi.testclient import TestClient

from chord import router as router_mod
from chord import specialists
from chord.config import Settings
from chord.server import Deps, create_app, load_specialists
from test_progress import last_trace
from test_skeleton import AvailableImageBackend, FakeUpstream

load_specialists()

MSG_TOOL = {"type": "function", "function": {"name": "message", "parameters": {"type": "object"}}}
CONTINUATION = [
    {"role": "user", "content": "Send me a selfie?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                         "function": {"name": "memory_search", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "{\"results\": []}"},
]


class CountingRouter:
    calls = 0

    async def ainvoke(self, msgs):
        if "You check one reply" in str(msgs[0].content):   # the #109 delivery check, not routing
            class N: content = '{"claims_delivery": false}'
            return N()
        CountingRouter.calls += 1
        class R: content = '{"route": "image", "intent": "a selfie"}'
        return R()


class ClarifyRouter:
    async def ainvoke(self, msgs):
        class R: content = '{"route": "clarify", "question": "Which look do you want?"}'
        return R()


def app(tmp_path, monkeypatch, router):
    renders = []

    async def image(job, ctx):
        renders.append(job)
        raise AssertionError("a specialist started")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        experimental_routes=frozenset({"image"}))
    up = FakeUpstream()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: router(),
                                     image_backend=AvailableImageBackend()))), up, settings, renders


def post(c, messages, stream=False, **extra):
    body = {"model": "chord-1-poly", "messages": messages, "stream": stream, **extra}
    if stream:
        with c.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            list(r.iter_lines())
    else:
        assert c.post("/v1/chat/completions", json=body).status_code == 200


@pytest.mark.parametrize("stream", [False, True])
def test_a_tool_result_continuation_is_not_routed(tmp_path, monkeypatch, stream):
    CountingRouter.calls = 0
    c, up, settings, renders = app(tmp_path, monkeypatch, CountingRouter)
    post(c, CONTINUATION, stream, tools=[MSG_TOOL])
    assert CountingRouter.calls == 0 and renders == []
    t = last_trace(settings)
    assert t["router"] == "skipped_tool_continuation" and t["route_decision"] == "chat"
    assert up.bodies[0]["messages"][-1]["role"] == "tool"       # her loop reaches her untouched


def test_the_first_request_of_the_same_turn_is_the_clients_too(tmp_path, monkeypatch):
    """S04 (2026-09-16): a request that declares tools is never routed; without
    tools the same ask is (control)."""
    CountingRouter.calls = 0
    c, up, settings, renders = app(tmp_path, monkeypatch, CountingRouter)
    post(c, CONTINUATION[:1], tools=[MSG_TOOL])
    assert CountingRouter.calls == 0 and renders == []
    assert last_trace(settings)["router"] == "skipped_client_tools"
    post(c, CONTINUATION[:1])
    assert CountingRouter.calls == 1 and len(renders) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_a_router_clarify_is_never_injected_into_a_tool_client(tmp_path, monkeypatch, stream):
    c, up, settings, renders = app(tmp_path, monkeypatch, ClarifyRouter)
    post(c, CONTINUATION[:1], stream, tools=[MSG_TOOL])
    system = up.bodies[0]["messages"][0]["content"]
    assert "Ask the user" not in system and "No picture or file" not in system
    assert last_trace(settings)["router"] == "skipped_client_tools"


def test_without_tools_the_clarify_question_is_still_asked(tmp_path, monkeypatch):
    """Control: Open WebUI sends no tools; the service asks the question."""
    c, up, settings, renders = app(tmp_path, monkeypatch, ClarifyRouter)
    post(c, CONTINUATION[:1])
    messages = up.bodies[0]["messages"]
    assert "Ask the user" in messages[0]["content"]
    # The question itself is user-shaped: fenced after the conversation, not in the
    # system message (review 2026-09-27, #7).
    assert messages[-1]["role"] == "user" and "Which look do you want?" in messages[-1]["content"]
    assert "route_clarify_deferred_to_client" not in last_trace(settings)


# --- #83: the router asks for a JSON object ------------------------------------

class BindableRouter:
    bound = None

    def bind(self, **kw):
        BindableRouter.bound = kw
        return self

    async def ainvoke(self, msgs):
        class R: content = '{"route": "chat"}'
        return R()


def test_the_router_call_asks_for_json_mode():
    import asyncio
    from chord.registry import load
    BindableRouter.bound = None
    decision, _ = asyncio.run(router_mod.route([{"role": "user", "text": "hi"}], load(), BindableRouter()))
    assert BindableRouter.bound == {"response_format": {"type": "json_object"}}
    assert decision.route == "chat"


def test_the_b2_truncated_clarify_is_what_json_mode_prevents():
    """The exact SXBBN0M4 shape: an object missing its closing brace still fails
    the parser (we don't guess), which is why the call itself asks for JSON."""
    raw = '{"route": "clarify",\n  "intent": "x",\n  "question": "Which look?"'
    assert router_mod.parse(raw, {"image"}).parse_error == "no JSON object in router output"


@pytest.mark.parametrize("stream", [False, True])
def test_web_search_options_do_not_search_again_on_a_tool_continuation(tmp_path, monkeypatch, stream):
    """Review 2026-09-24 A3: a harness that sends the same params on every request
    of its loop (OpenClaw, SDK agents) carries `web_search_options` onto its tool-
    result continuations. The search was forced before the continuation skip, so
    every step of her loop searched again and got the citation contract injected
    into her system message -- #84's shape through a different door."""
    CountingRouter.calls = 0
    c, up, settings, renders = app(tmp_path, monkeypatch, CountingRouter)

    async def search(job, ctx):
        raise AssertionError("a search started on a tool continuation")

    monkeypatch.setitem(specialists.SPECIALISTS, "search", search)
    post(c, CONTINUATION, stream, tools=[MSG_TOOL], web_search_options={})
    t = last_trace(settings)
    assert t["router"] == "skipped_tool_continuation" and t["route_decision"] == "chat"
    assert up.bodies[0]["messages"][-1]["role"] == "tool"
