"""service_tier as light model routing (2026-09-14).

Each tier reaches the model in manifest `service_tier_routes`, the response
echoes the route's tier (OpenAI: the mode actually used, `priority` for both
`fast` and `priority`), the field is never forwarded to the backend, and the
trace records the real model.
"""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from chord import manifest
from chord.config import Settings
from chord.server import Deps, create_app, create_internal_app
from chord.upstream import Upstream
from test_skeleton import FakeUpstream, make

CHAT = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
ROUTES = manifest.load()["service_tier_routes"]


def production_shaped(tmp_path, upstream=None):
    settings = Settings(
        data_dir=tmp_path,
        persona_model="persona-production-model",
        persona_base_url="http://persona-direct:8113/v1",
        router_model="router-production-model",
        router_base_url="http://router-direct:8101/v1",
    )
    deps = Deps(settings, upstream=upstream or FakeUpstream(), model=lambda name: None)
    return deps, TestClient(create_app(deps))


def test_the_table_covers_every_spec_tier():
    assert set(ROUTES) == {"auto", "default", "flex", "scale", "fast", "priority"}
    assert ROUTES["fast"]["echo"] == ROUTES["priority"]["echo"] == "priority"
    assert {route["slot"] for route in ROUTES.values()} == {"persona", "router"}
    assert all("model" not in route for route in ROUTES.values())


@pytest.mark.parametrize("tier", sorted(ROUTES))
def test_each_tier_routes_to_its_model_and_echoes_non_stream(tmp_path, tier):
    up = FakeUpstream()
    deps, client = production_shaped(tmp_path, up)
    expected_model, expected_url = deps.settings.slot_target(ROUTES[tier]["slot"])
    r = client.post("/v1/chat/completions", json={**CHAT, "service_tier": tier})
    assert r.status_code == 200
    body = r.json()
    assert body["service_tier"] == ROUTES[tier]["echo"]
    sent = up.bodies[-1]
    assert sent["model"] == expected_model
    assert expected_url in {
        "http://persona-direct:8113/v1",
        "http://router-direct:8101/v1",
    }
    assert "service_tier" not in sent  # ours to honour, never forwarded
    trace_id = r.headers["x-request-id"]
    record = TestClient(create_internal_app(deps)).get(f"/internal/traces/{trace_id}").json()
    assert record["persona_model"] == expected_model


@pytest.mark.parametrize("tier", sorted(ROUTES))
def test_each_tier_echoes_on_every_stream_chunk(tmp_path, tier):
    up = FakeUpstream()
    deps, client = production_shaped(tmp_path, up)
    with client.stream("POST", "/v1/chat/completions", json={**CHAT, "service_tier": tier, "stream": True}) as r:
        chunks = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ") and line != "data: [DONE]"]
    assert chunks and all(c.get("service_tier") == ROUTES[tier]["echo"] for c in chunks)
    assert up.bodies[-1]["model"] == deps.settings.slot_target(ROUTES[tier]["slot"])[0]


def test_no_tier_keeps_the_default_model_and_adds_no_echo(tmp_path):
    up = FakeUpstream()
    deps, client = production_shaped(tmp_path, up)
    body = client.post("/v1/chat/completions", json=CHAT).json()
    assert "service_tier" not in body
    assert up.bodies[-1]["model"] == deps.settings.persona_model


@pytest.mark.parametrize("slot", ["persona", "router"])
def test_slot_resolution_refuses_missing_direct_configuration(tmp_path, slot):
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(ValueError, match="requires a nonblank model and direct base URL"):
        settings.slot_target(slot)


def test_slot_resolution_refuses_unknown_roles(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown chat slot"):
        settings.slot_target("gateway")


@pytest.mark.asyncio
async def test_upstream_selects_only_the_explicit_direct_model_pair():
    upstream = Upstream(
        "http://persona-direct:8113/v1",
        "test",
        chat_routes={
            "persona-production-model": "http://persona-direct:8113/v1",
            "router-production-model": "http://router-direct:8101/v1",
        },
    )
    try:
        assert upstream._chat_client({"model": "persona-production-model"}).base_url == httpx.URL(
            "http://persona-direct:8113/v1/"
        )
        assert upstream._chat_client({"model": "router-production-model"}).base_url == httpx.URL(
            "http://router-direct:8101/v1/"
        )
        with pytest.raises(RuntimeError, match="no direct chat route"):
            upstream._chat_client({"model": "example/chat"})
    finally:
        await upstream.aclose()


def test_one_model_cannot_ambiguously_name_two_direct_routes(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        persona_model="same-model",
        persona_base_url="http://persona-direct:8113/v1",
        router_model="same-model",
        router_base_url="http://router-direct:8101/v1",
    )
    with pytest.raises(ValueError, match="cannot identify two different direct routes"):
        Deps(settings)


@pytest.mark.parametrize("bad", ["turbo", "", 3, ["priority"]])
def test_an_unknown_tier_is_an_openai_400(tmp_path, bad):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**CHAT, "service_tier": bad})
    err = r.json()["error"]
    assert r.status_code == 400 and err["code"] == "invalid_service_tier" and err["param"] == "service_tier"
    assert up.bodies == []


def test_prompt_cache_retention_is_now_a_declared_noop(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**CHAT, "prompt_cache_retention": "24h"})
    assert r.status_code == 200
    assert "prompt_cache_retention" in manifest.load()["declared_noop_params"]


from chord.server import NOOP_VALIDATORS


def test_every_declared_noop_has_a_validator():
    assert set(manifest.load()["declared_noop_params"]) == set(NOOP_VALIDATORS)


GOOD = {"metadata": {"k": "v"}, "user": "u1", "safety_identifier": "x" * 64,
        "prompt_cache_key": "k", "prompt_cache_options": {"ttl": "30m", "mode": "explicit"},
        "prompt_cache_retention": "24h"}
BAD = [("prompt_cache_retention", "forever"), ("prompt_cache_retention", 7), ("prompt_cache_retention", {}),
       ("prompt_cache_retention", ["24h"]), ("metadata", {"k": 1}), ("metadata", "x"), ("store", "yes"), ("store", 1),
       ("user", 5), ("safety_identifier", "x" * 65), ("prompt_cache_key", 3),
       ("prompt_cache_options", {"ttl": "24h"}), ("prompt_cache_options", {"mode": "sometimes"}), ("prompt_cache_options", "30m")]


@pytest.mark.parametrize("param,value", sorted(GOOD.items()))
def test_valid_noop_values_are_accepted(tmp_path, param, value):
    _, client = make(tmp_path)
    assert client.post("/v1/chat/completions", json={**CHAT, param: value}).status_code == 200


@pytest.mark.parametrize("param,value", BAD)
def test_invalid_noop_values_are_an_openai_400_not_forwarded(tmp_path, param, value):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**CHAT, param: value})
    err = r.json()["error"]
    assert r.status_code == 400 and err["code"] == "invalid_value" and err["param"] == param
    assert up.bodies == []


def test_the_echo_is_the_mode_actually_used_s11():
    """Pinned spec: fast and priority are one mode (Fast mode), reported as
    priority; the echo is the processing mode actually used. A tier we don't
    distinguish runs as default and says so (S11, red team pass 1)."""
    assert ROUTES["fast"] == ROUTES["priority"] and ROUTES["fast"]["echo"] == "priority"
    default_slot = ROUTES["default"]["slot"]
    for tier, route in ROUTES.items():
        if route["slot"] == default_slot:
            assert route["echo"] == "default", tier
        else:
            assert route["echo"] != "default", tier


# S13g (S-cf-039): Predicted Outputs promise a faster completion and
# accepted/rejected prediction token counts. Neither is produced, so a value is
# refused. Null is the SDK's unset optional and is not a request for the feature.
def test_prediction_is_refused_until_the_token_counts_exist(tmp_path):
    from test_skeleton import FakeUpstream, make
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
    absent = client.post("/v1/chat/completions", json={**body, "prediction": None})
    assert absent.status_code == 200, absent.text
    assert "prediction" not in up.bodies[-1]
    for value in ({"type": "content", "content": "the quick brown fox"},
                  {"type": "content", "content": [{"type": "text", "text": "x"}]},
                  "fox", {"type": "content"}):
        r = client.post("/v1/chat/completions", json={**body, "prediction": value})
        assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter", (value, r.text)
        assert r.json()["error"]["param"] == "prediction"
    assert len(up.bodies) == 1
