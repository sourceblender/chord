"""background: true on Responses (spec): a queued Response at once, the turn run
as a task, progress through retrieve, and POST /responses/{id}/cancel."""
import asyncio
import time

import openai
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from openai import OpenAI

from chord.config import Settings
from chord.server import Deps, create_app
from test_responses import MODEL, strict
from test_skeleton import FakeUpstream


class Slow(FakeUpstream):
    """Holds the model call until released, so the test sees in_progress."""
    def __init__(self):
        super().__init__()
        self.release = None

    async def complete(self, body):
        self.release = asyncio.Event()
        await self.release.wait()
        return await super().complete(body)


def wait_release(up, timeout=5.0):
    """The in_progress ROW is written before the graph starts, and release
    exists only once complete() is entered -- a coupling that was NEVER
    guaranteed. The batch-3 timing change surfaced the race; it did not create
    it (a probe measured both wirings under stream_mode='debug': the conditional
    START edge adds no superstep). Wait on the call, not the row."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if up.release is not None:
            return up.release
        time.sleep(0.02)
    raise AssertionError("the model call never began")


def wait_for(sdk, rid, statuses, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = sdk.responses.retrieve(rid)
        if r.status in statuses:
            return r
        time.sleep(0.02)
    raise AssertionError(f"{rid} never reached {statuses}; last {r.status}")


@pytest.fixture
def served(tmp_path):
    up = Slow()
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=up, model=lambda n: None))) as client:
        yield up, client, OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=client)


def test_background_queues_then_completes(served):
    up, client, sdk = served
    raw = client.post("/v1/responses", json={"model": MODEL, "input": "hello", "background": True}).json()
    strict(raw, "response")
    assert raw["status"] == "queued" and raw["background"] is True and raw["output"] == []
    wait_for(sdk, raw["id"], {"in_progress"})
    wait_release(up)
    client.portal.call(lambda: up.release.set())
    done = wait_for(sdk, raw["id"], {"completed"})
    assert done.output_text == "hi there" and done.background is True
    strict(client.get(f"/v1/responses/{raw['id']}").json(), "response")


def test_empty_499_from_background_turn_records_failure(tmp_path):
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=FakeUpstream(),
                                    model=lambda n: None))) as client:
        async def disconnected(*_args, **_kwargs):
            return Response(status_code=499)

        client.app.state.run_chat = disconnected
        queued = client.post("/v1/responses", json={"model": MODEL, "input": "hello", "background": True}).json()
        end = time.monotonic() + 5
        while time.monotonic() < end:
            stored = client.get(f"/v1/responses/{queued['id']}").json()
            if stored["status"] == "failed":
                break
            time.sleep(0.02)
        assert stored["status"] == "failed"
        assert stored["error"] == {"code": "server_error", "message": "The response failed."}


def test_cancel_stops_a_running_response(served):
    up, client, sdk = served
    rid = sdk.responses.create(model=MODEL, input="hello", background=True).id
    wait_for(sdk, rid, {"in_progress"})
    cancelled = sdk.responses.cancel(rid)
    assert cancelled.status == "cancelled"
    assert sdk.responses.retrieve(rid).status == "cancelled"
    assert sdk.responses.cancel(rid).status == "cancelled"          # idempotent on a terminal response


def test_background_without_a_model_is_the_same_404_as_a_foreground_request(tmp_path):
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=FakeUpstream(),
                                    model=lambda n: None))) as client:
        missing = {"input": "hello"}
        background = client.post("/v1/responses", json={**missing, "background": True})
        foreground = client.post("/v1/responses", json=missing)
        assert background.status_code == foreground.status_code == 404
        assert background.json()["error"]["code"] == foreground.json()["error"]["code"] == "model_not_found"


def test_a_restart_fails_an_unfinished_background_response(tmp_path):
    from chord.responses_store import ResponseStore
    store = ResponseStore(tmp_path / "responses.sqlite3")
    store.put({"id": "resp_orphan", "object": "response", "status": "in_progress", "background": True,
               "output": [], "error": None}, [], [])
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=FakeUpstream(),
                                    model=lambda n: None))) as client:
        failed = client.get("/v1/responses/resp_orphan").json()
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "server_error"
        # Already terminal: cancel returns that object, it does not pretend the task is running.
        assert client.post("/v1/responses/resp_orphan/cancel").json()["status"] == "failed"


def test_cancel_of_a_queued_response_with_no_task_is_cancelled(tmp_path):
    from chord.responses_store import ResponseStore
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=FakeUpstream(),
                                    model=lambda n: None))) as client:
        store = ResponseStore(tmp_path / "responses.sqlite3")
        store.put({"id": "resp_queued", "object": "response", "status": "queued", "background": True,
                   "output": [], "error": None}, [], [])
        body = client.post("/v1/responses/resp_queued/cancel").json()
        assert body["status"] == "cancelled"


def test_a_background_failure_does_not_return_the_exception_text(tmp_path):
    class Boom(FakeUpstream):
        async def complete(self, body):
            raise RuntimeError("http://10.9.8.7/secret")

    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=Boom(),
                                    model=lambda n: None))) as client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=client)
        rid = sdk.responses.create(model=MODEL, input="hello", background=True).id
        failed = wait_for(sdk, rid, {"failed"})
        assert failed.error.message == "The response failed."
        assert "10.9.8.7" not in (failed.error.message or "")


def test_only_background_responses_can_be_cancelled(tmp_path):
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=FakeUpstream(),
                                    model=lambda n: None))) as client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=client)
        rid = sdk.responses.create(model=MODEL, input="hello").id
        with pytest.raises(openai.BadRequestError):
            sdk.responses.cancel(rid)
        with pytest.raises(openai.NotFoundError):
            sdk.responses.cancel("resp_missing")


@pytest.mark.parametrize("extra,param", [({"store": False}, "store"), ({"stream": True}, "stream"), ({"background": "yes"}, "background")])
def test_background_refusals(served, extra, param):
    up, client, sdk = served
    r = client.post("/v1/responses", json={"model": MODEL, "input": "x", "background": True, **extra})
    assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_cancel_answers_404_when_the_reread_comes_back_empty(served, monkeypatch):
    """Deployment verification (2026-09-22) once observed a cancel of a
    completed background response answer 200 with a bare `null` body: the
    route's closing re-read came back empty and JSONResponse(None) serialized
    it -- a failure shaped like a success. The re-read is now answered as what
    it is: an unknown id is a 404, never null."""
    up, client, sdk = served
    rid = client.post("/v1/responses", json={"model": MODEL, "input": "hello", "background": True}).json()["id"]
    wait_for(sdk, rid, {"in_progress"})
    wait_release(up).set()                # the model call -- not just the row -- is in flight
    wait_for(sdk, rid, {"completed"})

    from chord.responses_store import ResponseStore
    real = ResponseStore.get
    calls = {"n": 0}

    def empty_reread(self, response_id):
        calls["n"] += 1
        return real(self, response_id) if calls["n"] == 1 else None

    monkeypatch.setattr(ResponseStore, "get", empty_reread)
    r = client.post(f"/v1/responses/{rid}/cancel")
    assert r.status_code == 404, r.text
    strict(r.json(), "error")


def test_delete_of_a_running_background_response_stays_deleted(served):
    """DELETE removed the row but not the work: the still-running turn's final
    unconditional put resurrected the response as `completed`, input items and
    all, after the caller deleted it (review 2026-09-22, #1, reproduced).
    The delete now cancels the task, and every write the task makes is a
    tombstone-checked update."""
    up, client, sdk = served
    rid = client.post("/v1/responses", json={"model": MODEL, "input": "hello", "background": True}).json()["id"]
    wait_for(sdk, rid, {"in_progress"})
    wait_release(up)          # the model call is IN FLIGHT: this test's path is
                              # delete-during-the-call, and it is deterministic
                              # (the batch-3 review: without this wait the
                              # batch-1 red proof was timing-optional)

    r = client.delete(f"/v1/responses/{rid}")
    assert r.status_code == 200 and r.json()["deleted"] is True

    up.release.set()          # a straggler return must not resurrect anything
    time.sleep(0.3)           # any resurrection write gets its chance
    assert client.get(f"/v1/responses/{rid}").status_code == 404


def test_delete_before_the_model_call_burns_no_model_call(tmp_path):
    """The property the delete test above used to gesture at, asserted
    DIRECTLY (the batch-3 review): a delete that lands before the persona
    call begins means the call never begins. A blocking router parks the turn
    before speak(), so 'before the call' is a state the test controls, not a
    race it hopes to win."""
    up = Slow()

    class BlockingRouter:
        def __init__(self):
            self.release = asyncio.Event()

        async def ainvoke(self, msgs):
            await self.release.wait()
            from types import SimpleNamespace
            return SimpleNamespace(content='{"route": "chat"}')

    router = BlockingRouter()
    with TestClient(create_app(Deps(Settings(data_dir=tmp_path,
                                              router_enabled=True),
                                    upstream=up, model=lambda n: router))) as rc:
        rsdk = OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=rc)
        rid = rc.post("/v1/responses", json={"model": MODEL, "input": "hello", "background": True}).json()["id"]
        wait_for(rsdk, rid, {"in_progress"})
        assert up.release is None, "precondition: the persona call has NOT begun"

        r = rc.delete(f"/v1/responses/{rid}")
        assert r.status_code == 200 and r.json()["deleted"] is True
        router.release.set()          # the cancelled turn must not resume into it
        time.sleep(0.3)
        assert up.release is None, "a deleted response burned a model call"
        assert rc.get(f"/v1/responses/{rid}").status_code == 404


def test_chaining_from_an_unfinished_response_is_refused_by_name(served):
    """A queued/in-progress row's stored conversation is missing its own turn,
    so chaining from one silently lost that context (review 2026-09-22,
    #7). Refused by name until terminal. (Chaining from a TERMINAL response is
    the pre-existing behavior and is covered by test_responses.py; a second
    foreground call here would hang on the Slow fixture's fresh Event, which
    is the fixture's contract, not the door's.)"""
    up, client, sdk = served
    rid = client.post("/v1/responses", json={"model": MODEL, "input": "remember: the teapot is blue",
                                             "background": True}).json()["id"]
    wait_for(sdk, rid, {"in_progress"})

    refused = client.post("/v1/responses", json={"model": MODEL, "input": "and again",
                                                 "previous_response_id": rid})
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "previous_response_incomplete"
    assert refused.json()["error"]["param"] == "previous_response_id"

    wait_release(up).set()                 # let the background turn finish cleanly
    wait_for(sdk, rid, {"completed"})
