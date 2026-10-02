"""Client budgets apply after authentication and until streaming completes."""

import asyncio
import json

import pytest
import httpx
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

from chord.client_admission import ClientAdmission
from chord.config import ConfigurationError, Settings
from chord.server import Deps, create_app


A = "a" * 48
B = "b" * 48


def _client(tmp_path, **limits):
    settings = Settings(data_dir=tmp_path, service_api_key="old-key",
                        client_keys_json=json.dumps({"client-a": A, "client-b": B}),
                        persona_model="example-persona", router_model="example-router",
                        persona_base_url="http://persona.test/v1",
                        router_base_url="http://router.test/v1", **limits)
    app = create_app(Deps(settings, model=lambda _: None))

    @app.post("/_test/compute")
    async def compute():
        return {"ok": True}

    return TestClient(app)


def _get(client, key):
    return client.post("/_test/compute", headers={"Authorization": f"Bearer {key}"})


def test_per_client_budget_and_health_exemption(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("chord.client_admission._now", lambda: now[0])
    client = _client(tmp_path, max_client_per_minute=1)
    assert _get(client, A).status_code == 200
    refused = _get(client, A)
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "rate_limit_exceeded"
    assert refused.headers["Retry-After"] == "60"
    traces = [json.loads(line) for path in (tmp_path / "traces").glob("*.jsonl")
              for line in path.read_text().splitlines()]
    assert len(traces) == 1
    assert traces[0]["client_id"] == "client-a"
    assert traces[0]["admission_limits"] == ["client_per_minute"]
    assert traces[0]["retry_after_s"] == 60
    assert _get(client, B).status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {A}"}).status_code == 200
    assert client.get("/v1/models").status_code == 401
    now[0] += 60
    assert _get(client, A).status_code == 200


def test_global_budget_counts_distinct_clients(tmp_path):
    client = _client(tmp_path, max_global_per_minute=1)
    assert _get(client, A).status_code == 200
    assert _get(client, B).status_code == 429


def test_background_turn_keeps_slot_after_queued_reply_and_cancel_can_release_it(tmp_path):
    from test_responses import MODEL
    from test_responses_background import Slow, wait_release

    upstream = Slow()
    settings = Settings(data_dir=tmp_path, service_api_key="old-key",
                        max_global_inflight=1, persona_model="example-persona", router_model="example-router",
                        persona_base_url="http://persona.test/v1",
                        router_base_url="http://router.test/v1")
    with TestClient(create_app(Deps(settings, upstream=upstream, model=lambda _: None))) as client:
        headers = {"Authorization": "Bearer old-key"}
        queued = client.post("/v1/responses", headers=headers,
                             json={"model": MODEL, "input": "hello", "background": True})
        assert queued.status_code == 200
        response_id = queued.json()["id"]
        wait_release(upstream)
        assert client.post("/v1/moderations", headers=headers).status_code == 429
        cancelled = client.post(f"/v1/responses/{response_id}/cancel", headers=headers)
        assert cancelled.status_code == 200
        assert client.post("/v1/moderations", headers=headers).status_code != 429


@pytest.mark.asyncio
async def test_real_app_holds_global_slot_until_stream_finishes(tmp_path):
    settings = Settings(data_dir=tmp_path, service_api_key="old-key",
                        client_keys_json=json.dumps({"client-a": A, "client-b": B}),
                        max_global_inflight=1, persona_model="example-persona", router_model="example-router",
                        persona_base_url="http://persona.test/v1",
                        router_base_url="http://router.test/v1")
    app = create_app(Deps(settings, model=lambda _: None))
    started = asyncio.Event()
    finish = asyncio.Event()

    @app.post("/_test/stream")
    async def stream():
        async def chunks():
            yield b"first"
            started.set()
            await finish.wait()
            yield b"last"

        return StreamingResponse(chunks())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as client:
        first = asyncio.create_task(client.post("/_test/stream", headers={"Authorization": f"Bearer {A}"}))
        await asyncio.wait_for(started.wait(), 2)
        refused = await client.post("/v1/moderations", headers={"Authorization": f"Bearer {B}"})
        assert refused.status_code == 429
        finish.set()
        assert (await first).content == b"firstlast"
        assert (await client.post("/_test/stream", headers={"Authorization": f"Bearer {B}"})).status_code == 200


@pytest.mark.asyncio
async def test_inflight_slot_lasts_until_stream_ends_and_releases_on_cancel():
    started = asyncio.Event()
    finish = asyncio.Event()

    async def stream_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        started.set()
        await finish.wait()
        await send({"type": "http.response.body", "body": b"done"})

    limiter = ClientAdmission(stream_app, global_inflight=0, client_inflight=1,
                              global_per_minute=0, client_per_minute=0)

    async def call(client):
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b""}

        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
                 "state": {"client_id": client}}
        await limiter(scope, receive, send)
        return sent

    first = asyncio.create_task(call("a"))
    await started.wait()
    refused = await call("a")
    assert refused[0]["status"] == 429
    assert limiter.active_by_client["a"] == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert limiter.active_by_client["a"] == 0
    finish.set()
    assert (await call("a"))[0]["status"] == 200


@pytest.mark.asyncio
async def test_background_lease_releases_even_if_task_is_cancelled_before_first_step():
    from chord.client_admission import AdmissionLease

    async def unused():
        await asyncio.sleep(10)

    limiter = ClientAdmission(unused, global_inflight=0, client_inflight=1,
                              global_per_minute=0, client_per_minute=0)
    limiter.active = 1
    limiter.active_by_client["client-a"] = 1
    lease = AdmissionLease(limiter, "client-a")
    task = asyncio.create_task(unused())
    lease.transfer_to(task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)  # done callbacks run on the next event-loop turn
    assert limiter.active == 0
    assert limiter.active_by_client["client-a"] == 0
    lease.release()
    assert limiter.active == 0


@pytest.mark.parametrize("field", [
    "max_global_inflight", "max_client_inflight", "max_global_per_minute", "max_client_per_minute",
])
def test_invalid_limits_refuse_at_startup(field):
    settings = Settings(**{field: -1})
    with pytest.raises(ConfigurationError, match="nonnegative integer"):
        settings.validate_startup()
