"""The Responses API (#146): create (non-stream and stream), retrieve, delete,
input_items and previous_response_id chaining, through the official SDK, with
every object and stream event validated strict against the pinned spec."""
import base64
import json
import sqlite3

import openai
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from openai import OpenAI

from qa.conformance.schema import Spec, validate_payload
from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app, create_internal_app, load_specialists
from test_client_tools_own_the_turn import CALLS, Calling
from test_progress import FixedRouter
from test_sdk_parse import undeclared
from test_skeleton import AvailableImageBackend, PNG, FakeUpstream

load_specialists()
MODEL = "chord-1-poly"
SPEC = Spec()


def strict(payload, kind):
    row = validate_payload(payload, kind=kind, spec=SPEC, fields="strict")
    assert row["verdict"] == "pass", json.dumps(row["evidence"], indent=1)[:2500]


def make(tmp_path, upstream=None, router=None, image_backend=None, **settings):
    settings.setdefault("persona_thinking_mode", "qwen_chat_template")
    s = Settings(data_dir=tmp_path, **settings)
    deps = Deps(s, upstream=upstream or FakeUpstream(), model=lambda n: router() if router else None,
                image_backend=image_backend)
    client = TestClient(create_app(deps))
    return deps, client, OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=client)


def trace(deps, request_id):
    return TestClient(create_internal_app(deps)).get(f"/internal/traces/{request_id}").json()


# --- create, retrieve, delete, input_items ------------------------------------------------

def test_create_returns_a_strict_response_and_stores_it(tmp_path):
    deps, client, sdk = make(tmp_path)
    r = client.post("/v1/responses", json={"model": MODEL, "input": "hello"})
    assert r.status_code == 200, r.text
    body = r.json()
    strict(body, "response")
    assert body["id"].startswith("resp_") and body["status"] == "completed"
    [msg] = body["output"]
    assert msg["type"] == "message" and msg["content"][0] == {"type": "output_text", "text": "hi there", "annotations": [], "logprobs": []}
    assert body["usage"] == {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens": 2,
                             "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 7}
    sent = deps.upstream.bodies[0]
    assert sent["messages"][-1] == {"role": "user", "content": "hello"} and "max_completion_tokens" not in sent

    parsed = sdk.responses.retrieve(body["id"])
    assert parsed.output_text == "hi there" and undeclared(parsed) == []
    assert client.get(f"/v1/responses/{body['id']}").json() == body

    items = client.get(f"/v1/responses/{body['id']}/input_items").json()
    strict(items, "response-items")
    assert [i["content"][0]["text"] for i in items["data"]] == ["hello"]
    assert trace(deps, r.headers["x-request-id"])["response_id"] == body["id"]

    sdk.responses.delete(body["id"])
    with pytest.raises(openai.NotFoundError):
        sdk.responses.retrieve(body["id"])
    assert client.delete(f"/v1/responses/{body['id']}").status_code == 404


def test_store_false_keeps_nothing(tmp_path):
    deps, client, sdk = make(tmp_path)
    resp = sdk.responses.create(model=MODEL, input="hello", store=False)
    assert client.get(f"/v1/responses/{resp.id}").status_code == 404


def test_early_chat_disconnect_returns_499_without_parsing_or_storing(tmp_path):
    _, client, _ = make(tmp_path)

    async def disconnected(*_args, **_kwargs):
        return Response(status_code=499)

    client.app.state.run_chat = disconnected
    result = client.post("/v1/responses", json={"model": MODEL, "input": "hello", "stream": True})
    assert result.status_code == 499
    assert result.content == b""
    with sqlite3.connect(tmp_path / "responses.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_previous_response_id_continues_without_resending_and_instructions_do_not_carry(tmp_path):
    deps, client, sdk = make(tmp_path)
    first = sdk.responses.create(model=MODEL, input="my name is Ava", instructions="Be brief.")
    second = sdk.responses.create(model=MODEL, input="what is my name?", previous_response_id=first.id)
    assert second.previous_response_id == first.id and second.instructions is None
    sent = deps.upstream.bodies[1]["messages"]
    conversation = [(m["role"], m["content"]) for m in sent if m["role"] != "system"]
    assert conversation == [("user", "my name is Ava"), ("assistant", "hi there"), ("user", "what is my name?")]
    assert "Be brief." not in json.dumps(sent)                   # first turn's instructions are not carried
    assert "Be brief." in json.dumps(deps.upstream.bodies[0]["messages"])


def test_an_unknown_previous_response_is_a_400(tmp_path):
    deps, client, sdk = make(tmp_path)
    with pytest.raises(openai.BadRequestError) as exc:
        sdk.responses.create(model=MODEL, input="x", previous_response_id="resp_nope")
    assert exc.value.param == "previous_response_id" and deps.upstream.bodies == []


def test_input_items_pagination(tmp_path):
    deps, client, sdk = make(tmp_path)
    resp = sdk.responses.create(model=MODEL, input=[{"role": "user", "content": t} for t in ("a", "b", "c")])
    desc = client.get(f"/v1/responses/{resp.id}/input_items?limit=2").json()
    assert [i["content"][0]["text"] for i in desc["data"]] == ["c", "b"] and desc["has_more"] is True
    rest = client.get(f"/v1/responses/{resp.id}/input_items?limit=2&after={desc['last_id']}").json()
    assert [i["content"][0]["text"] for i in rest["data"]] == ["a"] and rest["has_more"] is False
    asc = sdk.responses.input_items.list(resp.id, order="asc")
    assert [i.content[0].text for i in asc.data] == ["a", "b", "c"]
    assert client.get(f"/v1/responses/{resp.id}/input_items?limit=0").status_code == 400


# --- request translation ---------------------------------------------------------------

def test_fields_map_onto_the_chat_core(tmp_path):
    deps, client, sdk = make(
        tmp_path,
        persona_model="persona-production-model",
        persona_base_url="http://persona-direct:8113/v1",
        router_model="router-production-model",
        router_base_url="http://router-direct:8101/v1",
    )
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    sdk.responses.create(model=MODEL, input="x", max_output_tokens=50, temperature=0.3, top_p=0.9,
                         reasoning={"effort": "low"}, service_tier="fast", metadata={"k": "v"},
                         text={"format": {"type": "json_schema", "name": "thing", "schema": schema, "strict": True}, "verbosity": "low"})
    sent = deps.upstream.bodies[0]
    assert (sent["max_completion_tokens"], sent["temperature"], sent["top_p"]) == (50, 0.3, 0.9)
    # reasoning.effort still maps onto the chat core, and the chat core then turns it
    # into the backend's thinking switch — the translation an earlier gateway used to
    # do for us, owned here since LiteLLM came out of the middle (2026-09-18).
    assert "reasoning_effort" not in sent
    assert sent["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
    assert sent["response_format"] == {"type": "json_schema", "json_schema": {"name": "thing", "schema": schema, "strict": True}}
    assert sent["model"] == "router-production-model"             # fast resolves the configured router slot


def test_a_length_finish_is_incomplete(tmp_path):
    class Short(FakeUpstream):
        async def complete(self, body):
            data, dep = await super().complete(body)
            data["choices"][0]["finish_reason"] = "length"
            return data, dep
    deps, client, sdk = make(tmp_path, Short())
    resp = sdk.responses.create(model=MODEL, input="x", max_output_tokens=2)
    assert resp.status == "incomplete" and resp.incomplete_details.reason == "max_output_tokens"


def test_a_chat_core_refusal_names_the_responses_field(tmp_path):
    deps, client, sdk = make(tmp_path)
    with pytest.raises(openai.BadRequestError) as exc:
        sdk.responses.create(model=MODEL, input="x", max_output_tokens=0)
    assert exc.value.param == "max_output_tokens"
    with pytest.raises(openai.NotFoundError):
        sdk.responses.create(model="gpt-6", input="x")


@pytest.mark.parametrize("extra,param", [
    ({"prompt": {"id": "pmpt_1"}}, "prompt"),
    ({"include": ["reasoning.encrypted_content"]}, "include"),
    ({"truncation": "auto"}, "truncation"),
    ({"top_logprobs": 3}, "top_logprobs"),
    ({"max_tool_calls": 2}, "max_tool_calls"),
    ({"reasoning": {"summary": "auto"}}, "reasoning.summary"),
    ({"tools": [{"type": "file_search", "vector_store_ids": ["vs"]}]}, "tools[0].type"),
    ({"tools": [{"type": "web_search", "search_context_size": "high"}]}, "tools[0]"),
    ({"input": [{"type": "item_reference", "id": "msg_1"}]}, "input[0].type"),
    ({"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]}, "input[0].content[0].type"),
    ({"text": {"format": {"type": "grammar"}}}, "text.format.type"),
    ({"bogus": 1}, "bogus"),
])
def test_what_the_backend_cannot_do_is_refused_by_name(tmp_path, extra, param):
    deps, client, sdk = make(tmp_path)
    r = client.post("/v1/responses", json={"model": MODEL, "input": "x", **extra})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == param
    strict(r.json(), "error")
    assert deps.upstream.bodies == []


@pytest.mark.parametrize("builtin", [
    {"type": "web_search"},
    {"type": "web_search_preview"},
    {"type": "image_generation"},
])
def test_function_tools_alongside_a_built_in_are_refused_not_silently_dropped(tmp_path, builtin):
    """Function tools own the turn (S04), so a built-in beside them would never
    run. The old behaviour echoed the input tool list and returned a `message`,
    which is the exact 'silent ignore' shape the project refuses everywhere else.
    Pinned here so a regression is caught at the wire, not in a postmortem."""
    deps, client, _ = make(tmp_path)
    r = client.post("/v1/responses", json={
        "model": MODEL, "input": "x",
        "tools": [WEATHER, builtin],
    })
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "tools"
    assert r.json()["error"]["code"] == "unsupported_value"
    strict(r.json(), "error")
    assert deps.upstream.bodies == []


def test_pure_built_in_tools_still_route_to_the_specialist(tmp_path, monkeypatch):
    """The refusal above is the new shape; pure built-ins must still work, so the
    guard did not regress the supported path. Pinned beside the new test."""
    ran = []

    async def image(job, ctx):
        ran.append(job)
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")], summary="ok")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    deps, client, _ = make(tmp_path, router=FixedRouter, router_enabled=True,
                           enabled_routes=frozenset({"image"}), image_backend=AvailableImageBackend())
    r = client.post("/v1/responses", json={
        "model": MODEL, "input": "draw a mug",
        "tools": [{"type": "image_generation"}],
    })
    assert r.status_code == 200, r.text
    assert len(ran) == 1
    assert [i["type"] for i in r.json()["output"]] == ["message", "image_generation_call"]


# --- tools --------------------------------------------------------------------------------

WEATHER = {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}, "strict": False}


@pytest.mark.parametrize("stream", [False, True])
def test_function_calls_come_back_as_items_and_the_output_continues_the_turn(tmp_path, stream):
    deps, client, sdk = make(tmp_path, Calling())
    if stream:
        with sdk.responses.stream(model=MODEL, input="weather in Paris and Oslo?", tools=[WEATHER]) as s:
            for _ in s:
                pass
            resp = s.get_final_response()
    else:
        resp = sdk.responses.create(model=MODEL, input="weather in Paris and Oslo?", tools=[WEATHER])
    calls = [i for i in resp.output if i.type == "function_call"]
    assert [(c.call_id, c.name, c.arguments) for c in calls] == [(x["id"], "get_weather", x["function"]["arguments"]) for x in CALLS]
    assert deps.upstream.bodies[0]["tools"] == [{"type": "function", "function": {"name": "get_weather", "parameters": WEATHER["parameters"], "strict": False}}]
    strict(client.get(f"/v1/responses/{resp.id}").json(), "response")

    deps.upstream.__class__ = FakeUpstream                           # the model now answers in text (bodies kept)
    follow = sdk.responses.create(model=MODEL, previous_response_id=resp.id, tools=[WEATHER], input=[
        {"type": "function_call_output", "call_id": "call_a", "output": "18C"},
        {"type": "function_call_output", "call_id": "call_b", "output": "9C"}])
    sent = deps.upstream.bodies[-1]["messages"]
    assert [m["role"] for m in sent if m["role"] != "system"] == ["user", "assistant", "tool", "tool"]
    assert [c["id"] for c in sent[-3]["tool_calls"]] == ["call_a", "call_b"] and sent[-1] == {"role": "tool", "tool_call_id": "call_b", "content": "9C"}
    assert follow.output_text == "hi there"


def test_built_in_image_generation_is_an_item_and_only_when_offered(tmp_path, monkeypatch):
    ran = []

    async def image(job, ctx):
        ran.append(job)
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")], summary="a mug")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    deps, client, sdk = make(tmp_path, router=FixedRouter, router_enabled=True,
                             enabled_routes=frozenset({"image"}), image_backend=AvailableImageBackend())
    plain = client.post("/v1/responses", json={"model": MODEL, "input": "draw a mug"}).json()
    assert ran == [] and [i["type"] for i in plain["output"]] == ["message"]

    r = client.post("/v1/responses", json={"model": MODEL, "input": "draw a mug", "tools": [{"type": "image_generation"}]})
    body = r.json()
    strict(body, "response")
    assert len(ran) == 1 and [i["type"] for i in body["output"]] == ["message", "image_generation_call"]
    img = body["output"][1]
    assert base64.b64decode(img["result"]) == PNG and img["status"] == "completed"
    assert "![image]" not in body["output"][0]["content"][0]["text"]         # never markdown on this door
    assert trace(deps, r.headers["x-request-id"])["route_decision"] == "image"


def test_built_in_web_search_reports_the_call(tmp_path, monkeypatch):
    async def search(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, summary="Paris is sunny.",
                      provenance={"kind": "search", "query": "paris weather today",
                                  "sources": [{"id": 1, "url": "https://weather.example/paris", "title": "Paris weather"}]})

    class SearchRouter:
        async def ainvoke(self, msgs):
            class R: content = '{"route": "search", "intent": "paris weather today"}'
            return R()

    monkeypatch.setitem(specialists.SPECIALISTS, "search", search)
    deps, client, sdk = make(tmp_path, router=SearchRouter, router_enabled=True, enabled_routes=frozenset({"search"}))
    body = client.post("/v1/responses", json={"model": MODEL, "input": "weather in Paris?", "tools": [{"type": "web_search"}]}).json()
    strict(body, "response")
    assert [i["type"] for i in body["output"]] == ["web_search_call", "message"]
    assert body["output"][0]["action"] == {"type": "search", "query": "paris weather today"}


# --- streaming ------------------------------------------------------------------------------

def events_of(client, body):
    with client.stream("POST", "/v1/responses", json={**body, "stream": True}) as r:
        assert r.status_code == 200
        raw = r.read().decode()
    out = []
    for frame in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        payload = json.loads(lines["data"])
        assert lines["event"] == payload["type"]
        out.append(payload)
    return out


def test_a_text_stream_is_the_spec_event_sequence(tmp_path):
    deps, client, sdk = make(tmp_path)
    events = events_of(client, {"model": MODEL, "input": "hello"})
    for e in events:
        strict(e, "response-event")
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert [e["type"] for e in events] == [
        "response.created", "response.in_progress", "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.delta", "response.output_text.done",
        "response.content_part.done", "response.output_item.done", "response.completed"]
    final = events[-1]["response"]
    assert final["status"] == "completed" and final["usage"]["total_tokens"] == 7
    assert client.get(f"/v1/responses/{final['id']}").json() == final


def test_the_sdk_streams_and_accumulates(tmp_path):
    deps, client, sdk = make(tmp_path)
    with sdk.responses.stream(model=MODEL, input="hello") as s:
        deltas = [e.delta for e in s if e.type == "response.output_text.delta"]
        final = s.get_final_response()
    assert "".join(deltas) == "hi there" == final.output_text and undeclared(final) == []


def test_an_image_stream_ends_with_the_image_item(tmp_path, monkeypatch):
    async def image(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")], summary="a mug")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    deps, client, sdk = make(tmp_path, router=FixedRouter, router_enabled=True,
                             enabled_routes=frozenset({"image"}), image_backend=AvailableImageBackend())
    events = events_of(client, {"model": MODEL, "input": "draw a mug", "tools": [{"type": "image_generation"}]})
    for e in events:
        strict(e, "response-event")
    kinds = [e["type"] for e in events]
    assert kinds[-6:] == ["response.output_item.added", "response.image_generation_call.in_progress",
                          "response.image_generation_call.generating", "response.image_generation_call.completed",
                          "response.output_item.done", "response.completed"]
    assert base64.b64decode(events[-1]["response"]["output"][-1]["result"]) == PNG


def test_a_stream_that_fails_after_headers_ends_in_response_failed(tmp_path):
    class Breaks(FakeUpstream):
        async def stream(self, body):
            yield None, {}
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}, {}
            raise RuntimeError("backend fell over")
    deps, client, sdk = make(tmp_path, Breaks())
    events = events_of(client, {"model": MODEL, "input": "hello"})
    for e in events:
        strict(e, "response-event")
    assert events[-1]["type"] == "response.failed" and events[-1]["response"]["status"] == "failed"


def test_deep_nesting_is_a_400_on_the_responses_doors_too(tmp_path):
    """The responses doors caught ValueError, which covers invalid UTF-8 but
    not RecursionError (review 2026-09-22, #5)."""
    deep = '{"model": "' + MODEL + '", "input": ' + '[' * 50000 + ']' * 50000 + '}'
    deps, client, sdk = make(tmp_path)
    r = client.post("/v1/responses", content=deep.encode(), headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_json"


class Refuses(FakeUpstream):
    """A model that declines: Chat's `refusal`, not `content`."""
    async def complete(self, body):
        self.bodies.append(body)
        return ({"choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": None, "refusal": "I can't help with that."}}],
                 "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}, {})

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for piece in ["I can't", " help with that."]:
            yield {"object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {"refusal": piece}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def test_a_streamed_refusal_is_a_refusal_part_as_non_stream_and_replay_say(tmp_path):
    """Review 2026-09-24 B3: the live stream never read `delta.refusal`, so a
    declined turn went out as an empty output_text while the non-stream answer
    and the replay of it carried a refusal part. All three now agree."""
    deps, client, sdk = make(tmp_path, Refuses())
    events = events_of(client, {"model": MODEL, "input": "something she declines"})
    for e in events:
        strict(e, "response-event")
    assert [e["type"] for e in events] == [
        "response.created", "response.in_progress", "response.output_item.added", "response.content_part.added",
        "response.refusal.delta", "response.refusal.delta", "response.refusal.done",
        "response.content_part.done", "response.output_item.done", "response.completed"]
    refusal = {"type": "refusal", "refusal": "I can't help with that."}
    assert events[3]["part"] == {"type": "refusal", "refusal": ""}
    assert events[6]["refusal"] == "I can't help with that." and events[7]["part"] == refusal
    streamed = events[-1]["response"]["output"]
    assert [i["content"] for i in streamed] == [[refusal]]

    plain = client.post("/v1/responses", json={"model": MODEL, "input": "something she declines"}).json()
    assert [i["content"] for i in plain["output"]] == [[refusal]]
    replayed = events_of_replay(client, plain["id"])
    for e in replayed:
        strict(e, "response-event")
    # The replay sends the refusal as one delta; the live stream as the model's pieces.
    def shape(evs):
        kinds = [e["type"] for e in evs]
        return [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1] or not k.endswith(".delta")]
    assert shape(events) == shape(replayed)
    assert replayed[-1]["response"]["output"][0]["content"] == [refusal]


def events_of_replay(client, rid):
    with client.stream("GET", f"/v1/responses/{rid}?stream=true") as r:
        raw = r.read().decode()
    return [json.loads([ln for ln in f.splitlines() if ln.startswith("data: ")][0][6:]) for f in raw.strip().split("\n\n") if f]
