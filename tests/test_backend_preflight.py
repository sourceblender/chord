"""The deploy probe must exercise each configured backend, without echoing secrets."""

from __future__ import annotations

import json

import httpx
import pytest

from chord import backend_preflight
from chord.backend_preflight import check
from chord.config import Settings


def settings() -> Settings:
    return Settings(
        service_api_key="service-secret",
        persona_model="persona-model", persona_base_url="http://persona.test/v1",
        router_model="router-model", router_base_url="http://router.test/v1",
        stt_model="stt-model", stt_base_url="http://stt.test/v1",
        tts_model="tts-model", tts_base_url="http://tts.test/v1",
        embeddings_model="embedding-model", embeddings_base_url="http://embeddings.test/dense",
        embeddings_basic_auth="user:embedding-secret",
    )


def test_every_configured_backend_receives_a_real_small_request() -> None:
    seen: dict[str, httpx.Request] = {}

    def answer(request: httpx.Request) -> httpx.Response:
        name = request.url.host.split(".")[0]
        seen[name] = request
        if name in {"persona", "router"}:
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        if name == "stt":
            return httpx.Response(200, json={"text": ""})
        if name == "tts":
            return httpx.Response(200, content=b"RIFFprobe")
        return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        assert check(settings(), client) == []

    assert set(seen) == {"persona", "router", "stt", "tts", "embeddings"}
    assert json.loads(seen["persona"].content)["model"] == "persona-model"
    assert json.loads(seen["router"].content)["model"] == "router-model"
    assert b"probe.wav" in seen["stt"].content
    assert json.loads(seen["tts"].content)["voice"] == "alloy"
    assert seen["embeddings"].headers["Authorization"].startswith("Basic ")
    assert "embedding-secret" not in seen["embeddings"].headers["Authorization"]


@pytest.mark.parametrize("base_url", ["http://embeddings.test", "http://embeddings.test/v1"])
def test_embeddings_preflight_uses_the_same_v1_route_as_the_service(base_url: str) -> None:
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        assert check(Settings(persona_base_url="", router_base_url="", stt_base_url="", tts_base_url="",
                              embeddings_base_url=base_url, embeddings_model="embedding-model"), client) == ["persona", "router"]
    assert seen == ["/v1/embeddings"]


@pytest.mark.parametrize("failed", ["persona", "router", "stt", "tts", "embeddings"])
def test_each_backend_failure_is_named_without_its_response_body(failed: str) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        name = request.url.host.split(".")[0]
        if name == failed:
            return httpx.Response(503, text="private upstream body with embedding-secret")
        if name in {"persona", "router"}:
            return httpx.Response(200, json={"choices": [{}]})
        if name == "stt":
            return httpx.Response(200, json={"text": ""})
        if name == "tts":
            return httpx.Response(200, content=b"RIFFprobe")
        return httpx.Response(200, json={"data": [{}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        failures = check(settings(), client)
    assert failures == [failed]
    assert "embedding-secret" not in repr(failures)


def test_redirect_is_not_a_healthy_backend() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://other.test/"})

    with httpx.Client(transport=httpx.MockTransport(answer), follow_redirects=False) as client:
        assert check(settings(), client) == ["persona", "router", "stt", "tts", "embeddings"]


def test_main_uses_yaml_endpoints_over_stale_environment(tmp_path, monkeypatch) -> None:
    config = tmp_path / "chord.yaml"
    config.write_text("""version: 1
endpoints:
  chat: {type: openai-chat, url: http://yaml-backend.test/v1, model: yaml-model}
""")
    monkeypatch.setenv("CHORD_CONFIG", str(config))
    monkeypatch.setenv("CHORD_API_KEY", "test-key")
    monkeypatch.setenv("PERSONA_BASE_URL", "http://stale-env.test/v1")
    seen = []
    monkeypatch.setattr(backend_preflight, "check", lambda selected, client: seen.append(selected) or [])
    assert backend_preflight.main() == 0
    assert seen[0].persona_base_url == "http://yaml-backend.test/v1"
    assert seen[0].persona_model == "yaml-model"
