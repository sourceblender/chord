"""GET /v1/responses/{id}?stream=true[&starting_after=N]: a stored response as its
event stream. A streamed response replays exactly what it sent (so a client that
lost its connection resumes by sequence number); one that wasn't streamed gets a
deterministic replay; a background response still running streams as it lands."""
import json

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from chord.config import Settings
from chord.server import Deps, create_app
from test_client_tools_own_the_turn import Calling
from test_responses import MODEL, WEATHER, events_of, make, strict
from test_responses_background import Slow, wait_for, wait_release


def replay(client, rid, after=None):
    q = "?stream=true" + (f"&starting_after={after}" if after is not None else "")
    with client.stream("GET", f"/v1/responses/{rid}{q}") as r:
        assert r.status_code == 200, r.read()
        raw = r.read().decode()
    return [json.loads([l for l in f.splitlines() if l.startswith("data: ")][0][6:]) for f in raw.strip().split("\n\n") if f]


def test_a_streamed_response_replays_exactly_what_it_sent(tmp_path):
    deps, client, sdk = make(tmp_path)
    original = events_of(client, {"model": MODEL, "input": "hello"})
    rid = original[-1]["response"]["id"]
    assert replay(client, rid) == original
    assert replay(client, rid, after=4) == original[5:]
    assert replay(client, rid, after=original[-1]["sequence_number"]) == []


@pytest.mark.parametrize("upstream,tools", [(None, None), (Calling(), [WEATHER])], ids=["text", "function-calls"])
def test_a_non_streamed_response_replays_deterministically_and_strict(tmp_path, upstream, tools):
    deps, client, sdk = make(tmp_path, upstream)
    body = {"model": MODEL, "input": "hello", **({"tools": tools} if tools else {})}
    resp = client.post("/v1/responses", json=body).json()
    first, second = replay(client, resp["id"]), replay(client, resp["id"])
    assert first == second
    for e in first:
        strict(e, "response-event")
    assert [e["sequence_number"] for e in first] == list(range(len(first)))
    assert first[-1]["type"] == "response.completed" and first[-1]["response"] == resp


def test_the_sdk_retrieves_a_stream(tmp_path):
    deps, client, sdk = make(tmp_path)
    resp = sdk.responses.create(model=MODEL, input="hello")
    events = list(sdk.responses.retrieve(resp.id, stream=True))
    assert events[-1].type == "response.completed" and events[-1].response.output_text == "hi there"


def test_a_running_background_response_streams_as_it_lands(tmp_path):
    up = Slow()
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=up, model=lambda n: None))) as client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=client)
        rid = sdk.responses.create(model=MODEL, input="hello", background=True).id
        wait_for(sdk, rid, {"in_progress"})
        wait_release(up)
        client.portal.call(lambda: up.release.set())
        events = replay(client, rid)
        assert events[0]["type"] == "response.created" and events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["output"][0]["content"][0]["text"] == "hi there"


def test_a_failed_stream_is_stored_and_replays_its_failure(tmp_path):
    from test_skeleton import FakeUpstream

    class Breaks(FakeUpstream):
        async def stream(self, body):
            yield None, {}
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}, {}
            raise RuntimeError("backend fell over")
    deps, client, sdk = make(tmp_path, Breaks())
    original = events_of(client, {"model": MODEL, "input": "hello"})
    rid = original[-1]["response"]["id"]
    assert sdk.responses.retrieve(rid).status == "failed"
    assert replay(client, rid) == original


@pytest.mark.parametrize("query,param", [("?starting_after=3", "starting_after"), ("?stream=true&starting_after=x", "starting_after")])
def test_replay_refusals(tmp_path, query, param):
    deps, client, sdk = make(tmp_path)
    rid = sdk.responses.create(model=MODEL, input="hello").id
    r = client.get(f"/v1/responses/{rid}{query}")
    assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_a_live_function_call_stream_is_strict(tmp_path):
    """Found by the replay test: the live stream sent `name` on
    function_call_arguments.done, which the spec's event doesn't declare."""
    deps, client, sdk = make(tmp_path, Calling())
    for e in events_of(client, {"model": MODEL, "input": "weather?", "tools": [WEATHER]}):
        strict(e, "response-event")


@pytest.mark.asyncio
async def test_a_stuck_background_row_is_failed_past_the_replay_deadline(tmp_path, monkeypatch):
    """A background task that died on a BaseException leaves its row
    non-terminal forever, and every resumed client used to poll sqlite every
    250 ms -- a retention sweep inside every get -- until process restart.
    Past the deadline the row is failed once, its terminal events replay, and
    the stream ends (review 2026-09-22)."""
    from chord import responses as responses_mod
    from chord.responses_store import ResponseStore

    monkeypatch.setattr(responses_mod, "REPLAY_WAIT_S", 0.0)
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.put({"id": "resp_stuck", "status": "in_progress", "background": True, "output": []}, [], [])

    frames = [frame async for frame in responses_mod._replay(store, "resp_stuck", -1)]

    assert any("response.failed" in frame for frame in frames)
    stored = store.get("resp_stuck")
    assert stored["status"] == "failed"
    assert stored["error"]["code"] == "server_error"


@pytest.mark.asyncio
async def test_the_replay_deadline_never_executes_a_live_task(tmp_path, monkeypatch):
    """The deadline exists for rows whose worker is GONE. A passive observer
    resuming the stream of a still-running background response used to be able
    to mark it failed for everyone after an hour (review). With a liveness
    check it re-arms and keeps waiting instead."""
    import asyncio
    import contextlib

    from chord import responses as responses_mod
    from chord.responses_store import ResponseStore

    monkeypatch.setattr(responses_mod, "REPLAY_WAIT_S", 0.0)
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.put({"id": "resp_live", "status": "in_progress", "background": True, "output": []}, [], [])

    frames: list = []

    async def consume():
        async for frame in responses_mod._replay(store, "resp_live", -1, task_alive=lambda rid: True):
            frames.append(frame)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.8)              # several polls, each past the 0-second deadline
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert store.get("resp_live")["status"] == "in_progress"
    assert not any("response.failed" in frame for frame in frames)


def test_a_cancelled_response_replays_its_opening_and_ends(tmp_path):
    """Review 2026-09-24 B21: a cancelled response has no terminal stream event
    because the pinned spec defines none -- ResponseStreamEvent ends only in
    completed, incomplete or failed; `response.cancelled` is a webhook. The
    replay sends the opening pair and closes rather than inventing one or
    relabelling a cancel as a failure (stream replay). This pins that
    choice so a change to it is a decision, not drift."""
    from qa.conformance.schema import Spec
    union = Spec().document["components"]["schemas"]["ResponseStreamEvent"]
    members = [m["$ref"].rsplit("/", 1)[1] for m in union.get("anyOf", union.get("oneOf", []))]
    assert not [m for m in members if "Cancel" in m], members      # if the spec adds one, emit it here
    up = Slow()
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=up, model=lambda n: None))) as client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=client)
        rid = sdk.responses.create(model=MODEL, input="hello", background=True).id
        wait_for(sdk, rid, {"in_progress"})
        assert sdk.responses.cancel(rid).status == "cancelled"
        events = replay(client, rid)
        for e in events:
            strict(e, "response-event")
        assert [e["type"] for e in events] == ["response.created", "response.in_progress"]
