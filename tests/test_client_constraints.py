"""R1a (red team pass 1, 2026-09-15). A request that constrains its own answer
never goes to a specialist that would substitute for it:
- S03: tool_choice required / named (and a forced legacy function_call) came
  back as plain text or a search answer;
- S05: a strict json_schema request was routed to search and came back as
  prose, crashing the SDK's parse;
- S06: a json_schema request also ran the image specialist;
- S-cf-060: a returned client tool call had its content replaced by our
  citation-failure line."""
import json

import pytest
from fastapi.testclient import TestClient

from chord import specialists, web_search
from chord.specialists import search as SS
from chord.config import Settings
from chord.graph import client_constraint
from chord.server import Deps, create_app, load_specialists
from test_progress import last_trace
from test_search import app as search_app
from test_skeleton import AvailableImageBackend, FakeUpstream

load_specialists()

TOOL = {"type": "function", "function": {"name": "book_table", "parameters": {"type": "object"}}}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}}


class ObedientUpstream(FakeUpstream):
    """Her model honouring a forced call: the forced function, else the first
    declared one. Unforced, she answers in prose. Routing tests use this so a
    forced request is judged on where it went, not on the S03 enforcement."""

    def forced_name(self, body):
        choice, legacy = body.get("tool_choice"), body.get("function_call")
        if isinstance(choice, dict):
            return (choice.get("function") or {}).get("name")
        if isinstance(legacy, dict):
            return legacy.get("name")
        if choice == "required":
            return ((body.get("tools") or [{}])[0].get("function") or {}).get("name")
        return None

    async def complete(self, body):
        data, dep = await super().complete(body)
        name = self.forced_name(body)
        if name:
            key = "function_call" if isinstance(body.get("function_call"), dict) else "tool_calls"
            call = {"name": name, "arguments": "{}"}
            data["choices"][0]["message"] = {"role": "assistant", "content": None,
                                             key: call if key == "function_call" else [{"id": "call_1", "type": "function", "function": call}]}
            data["choices"][0]["finish_reason"] = "function_call" if key == "function_call" else "tool_calls"
        return data, dep

    async def stream(self, body):
        name = self.forced_name(body)
        if not name:
            async for item in super().stream(body):
                yield item
            return
        self.bodies.append(body)
        yield None, {}
        if isinstance(body.get("function_call"), dict):
            delta = {"function_call": {"name": name, "arguments": "{}"}}
        else:
            delta = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}}]}
        yield {"choices": [{"index": 0, "delta": delta}]}, {}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, {}


def router_for(route):
    class R:
        async def ainvoke(self, msgs):
            class A: content = json.dumps({"route": route, "intent": "do the thing"})
            return A()
    return R


def client(tmp_path, monkeypatch, route):
    calls = []

    async def spy(job, ctx):
        calls.append(job)
        raise AssertionError("a specialist ran")

    for cap in ("image", "search", "audio"):
        monkeypatch.setitem(specialists.SPECIALISTS, cap, spy)
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        enabled_routes=frozenset({"image", "search", "audio"}))
    up = ObedientUpstream()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: router_for(route)(),
                                      image_backend=AvailableImageBackend()))), up, settings, calls


@pytest.mark.parametrize("params,expected", [
    ({"tools": [TOOL], "tool_choice": "required"}, "forced_tool"),
    ({"tools": [TOOL], "tool_choice": {"type": "function", "function": {"name": "book_table"}}}, "forced_tool"),
    ({"functions": [TOOL["function"]], "function_call": {"name": "book_table"}}, "forced_tool"),
    ({"response_format": SCHEMA}, "response_format"),
    ({"response_format": {"type": "json_object"}}, "response_format"),
    ({"response_format": {"type": "text"}}, None),
    ({"tools": [TOOL], "tool_choice": "auto"}, None),
    ({"tools": [TOOL], "tool_choice": "none"}, None),
    ({}, None),
])
def test_client_constraint(params, expected):
    assert client_constraint(params) == expected


CONSTRAINED = {
    "S03 required": {"tools": [TOOL], "tool_choice": "required"},
    "S03 named": {"tools": [TOOL], "tool_choice": {"type": "function", "function": {"name": "book_table"}}},
    "S03 legacy function_call": {"functions": [TOOL["function"]], "function_call": {"name": "book_table"}},
    "S05/S06 json_schema": {"response_format": SCHEMA},
    "json_object": {"response_format": {"type": "json_object"}},
}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("route", ["search", "image", "audio"])
@pytest.mark.parametrize("case", list(CONSTRAINED))
def test_a_constrained_request_never_reaches_a_specialist(tmp_path, monkeypatch, case, route, stream):
    c, up, settings, calls = client(tmp_path, monkeypatch, route)
    body = {"model": "chord-1-poly", "stream": stream,
            "messages": [{"role": "user", "content": "Please book a table for two at 7pm."}], **CONSTRAINED[case]}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    assert calls == []                                   # no specialist substituted
    t = last_trace(settings)
    assert t["router"] == "skipped_client_constraint" and t["route_decision"] == "chat"
    sent = up.bodies[-1]
    for k, v in CONSTRAINED[case].items():               # the constraint reached the model as sent
        assert sent[k] == v


def test_an_unconstrained_request_still_routes(tmp_path, monkeypatch):
    """Control: the gate is the constraint, not the presence of a tool."""
    c, up, settings, calls = client(tmp_path, monkeypatch, "image")
    c.post("/v1/chat/completions", json={"model": "chord-1-poly",
                                         "messages": [{"role": "user", "content": "draw a cat"}]})
    assert len(calls) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case,param", [
    ("S03 required", "tool_choice"), ("S03 named", "tool_choice"),
    ("S03 legacy function_call", "function_call"),
    ("S05/S06 json_schema", "response_format"), ("json_object", "response_format"),
])
def test_web_search_options_with_a_client_constraint_is_refused(tmp_path, monkeypatch, case, param, stream):
    """A search answer is our cited prose: never the forced call or the JSON
    (the composition hole on #156: required + web_search_options ran search)."""
    c, up, settings, calls = client(tmp_path, monkeypatch, "search")
    r = c.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
                                             "web_search_options": {}, **CONSTRAINED[case],
                                             "messages": [{"role": "user", "content": "Book a table."}]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_parameter" and err["param"] == param
    assert calls == [] and up.bodies == []


def test_web_search_options_with_unforced_tools_still_searches(tmp_path, monkeypatch):
    """Control: auto tools are not a constraint, so the explicit search runs."""
    hits = [SS.Hit("Source", "https://source.example/x", "A fact.")]
    c, up, settings, searched = search_app(tmp_path, monkeypatch, hits)
    r = c.post("/v1/chat/completions", json={"model": "chord-1-poly", "web_search_options": {},
                                             "tools": [TOOL], "tool_choice": "auto",
                                             "messages": [{"role": "user", "content": "Book a table."}]})
    assert r.status_code == 200 and searched.calls == 1


# S-cf-060: auto tools, the router chose search, and the model answered with the
# client's tool call. The call's content must not become our citation line.
CALL = {"id": "call_1", "type": "function", "function": {"name": "book_table", "arguments": "{}"}}


class CallingUpstream(FakeUpstream):
    """Her model returning the client's call: modern tool_calls, or a legacy
    function_call (the legacy hole on #156)."""
    def __init__(self, content, legacy=False):
        super().__init__()
        self.content, self.legacy = content, legacy

    async def complete(self, body):
        data, dep = await super().complete(body)
        call = {"function_call": CALL["function"]} if self.legacy else {"tool_calls": [CALL]}
        data["choices"][0]["message"] = {"role": "assistant", "content": self.content, **call}
        data["choices"][0]["finish_reason"] = "function_call" if self.legacy else "tool_calls"
        return data, dep

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        if self.content is not None:
            yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": self.content}}]}, {}
        if self.legacy:
            delta = {"function_call": {"name": "book_table", "arguments": "{}"}}
        else:
            delta = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": "book_table", "arguments": "{}"}}]}
        yield {"choices": [{"index": 0, "delta": delta}]}, {}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "function_call" if self.legacy else "tool_calls"}]}, {}


@pytest.mark.parametrize("legacy", [False, True], ids=["tool_calls", "function_call"])
@pytest.mark.parametrize("content", [None, "", "Booking that now."])
@pytest.mark.parametrize("stream", [False, True])
def test_a_returned_client_call_is_not_wrapped_in_the_citation_line(tmp_path, monkeypatch, stream, content, legacy):
    hits = [SS.Hit("Source", "https://source.example/x", "A fact.")]
    c, up, settings, searched = search_app(tmp_path, monkeypatch, hits, upstream=CallingUpstream(content, legacy))
    offered = {"functions": [TOOL["function"]], "function_call": "auto"} if legacy else {"tools": [TOOL]}
    body = {"model": "chord-1-poly", "stream": stream, **offered, "web_search_options": {},
            "messages": [{"role": "user", "content": "Please book a table for two people at 7pm."}]}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    # The one server tool beside client tools is search the caller asked for;
    # the router never chooses it for a tool client (S04, 2026-09-16).
    assert searched.calls == 1
    key = "function_call" if legacy else "tool_calls"
    if stream:
        chunks = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
        deltas = [ch.get("delta") or {} for k in chunks for ch in k.get("choices") or []]
        text = "".join(d.get("content") or "" for d in deltas)
        calls = [d[key] for d in deltas if d.get(key)]
    else:
        msg = r.json()["choices"][0]["message"]
        text, calls = msg.get("content") or "", [msg.get(key)]
    name = calls[0]["name"] if legacy else calls[0][0]["function"]["name"]
    assert name == "book_table"                                         # reached the client
    assert web_search.CITATION_FAILED_LINE not in text                  # and was not wrapped
    assert text == (content or "")                                      # nor its words replaced
