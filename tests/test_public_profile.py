"""The public profile end to end: a generic YAML config with the package's own
manifest and registry, not the optional overlays conftest points the rest of the
suite at.

A fresh install names one text model and nothing else. Whatever the router
picks, no specialist may run, and every door without a backend refuses by name.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chord import manifest, registry
from chord.config import Settings
from chord.server import Deps, create_app
from chord.internal_api import create_app as create_internal_app
from test_skeleton import FakeUpstream

SRC = Path(__file__).resolve().parents[1] / "src" / "chord"
CONFIG = "endpoints:\n  chat: {type: openai-chat, url: http://127.0.0.1:9/v1, model: generic-chat}\n"


class RouteTo:
    """A router that always picks one route, so the specialist gate is what decides."""

    def __init__(self, route: str) -> None:
        self.route = route

    def bind(self, **_):
        return self

    async def ainvoke(self, _messages):
        class Reply:
            content = json.dumps({"route": self.route, "intent": "do it"})
        return Reply()


@pytest.fixture
def public(monkeypatch, tmp_path) -> Settings:
    monkeypatch.setattr(manifest, "PATH", SRC / "manifest.yaml")
    monkeypatch.setattr(registry, "PATH", SRC / "registry.yaml")
    manifest.load.cache_clear()
    config = tmp_path / "chord.yaml"
    config.write_text(CONFIG)
    yield replace(Settings.from_yaml(config), data_dir=tmp_path / "data", router_enabled=True)
    manifest.load.cache_clear()


def last_trace(settings: Settings) -> dict:
    lines = [line for p in sorted(Path(settings.trace_dir).glob("*.jsonl")) for line in p.read_text().splitlines()]
    return json.loads(lines[-1])


def test_the_public_profile_is_the_one_under_test(public) -> None:
    assert manifest.load()["release"] == "text-only"
    assert all(cap.model == "none" and not cap.routable for cap in registry.load().values())
    assert public.enabled_routes == frozenset()


@pytest.mark.parametrize("route", ["image", "search", "audio", "video"])
def test_a_routed_specialist_is_unavailable_and_only_the_text_model_is_called(public, route) -> None:
    upstream = FakeUpstream()
    deps = Deps(public, upstream=upstream, model=lambda _: RouteTo(route))
    r = TestClient(create_app(deps)).post("/v1/chat/completions", json={
        "model": "chord-1-poly", "messages": [{"role": "user", "content": "please do the thing"}]})
    assert r.status_code == 200, r.text
    trace = last_trace(public)
    # The router did choose this route; the gate refused it.
    assert trace["route_decision"] == route
    assert trace["route_unavailable"] == route
    assert "specialist" not in trace
    assert [body["model"] for body in upstream.bodies] == ["generic-chat"]


def test_web_search_options_does_not_search_on_the_public_profile(public) -> None:
    upstream = FakeUpstream()
    deps = Deps(public, upstream=upstream, model=lambda _: RouteTo("chat"))
    r = TestClient(create_app(deps)).post("/v1/chat/completions", json={
        "model": "chord-1-poly", "web_search_options": {},
        "messages": [{"role": "user", "content": "any news?"}]})
    assert r.status_code == 200, r.text
    trace = last_trace(public)
    assert trace["route_unavailable"] == "search" and "specialist" not in trace
    assert [body["model"] for body in upstream.bodies] == ["generic-chat"]


@pytest.mark.parametrize(("method", "path", "kwargs", "status", "code"), [
    ("post", "/v1/audio/speech", {"json": {"model": "chord-1-poly", "input": "hi", "voice": "alloy"}},
     503, "speech_not_configured"),
    ("post", "/v1/audio/transcriptions",
     {"files": {"file": ("a.wav", b"RIFF0000WAVEfmt ", "audio/wav")}, "data": {"model": "chord-1-poly"}},
     503, "transcription_not_configured"),
    ("post", "/v1/videos", {"json": {"model": "sora-2", "prompt": "x"}}, 503, "backend_unavailable"),
    ("post", "/v1/images/generations", {"json": {"model": "chord-1-poly", "prompt": "a cat"}},
     503, "capability_unavailable"),
    ("post", "/v1/chat/completions", {"json": {
        "model": "chord-1-poly", "modalities": ["text", "audio"], "audio": {"voice": "alloy", "format": "wav"},
        "messages": [{"role": "user", "content": "hi"}]}}, 400, "unsupported_modality"),
])
def test_doors_without_a_backend_refuse_by_name(public, method, path, kwargs, status, code) -> None:
    # The real Upstream: nothing is faked, and nothing is configured behind these doors.
    client = TestClient(create_app(Deps(public)))
    r = getattr(client, method)(path, **kwargs)
    assert (r.status_code, r.json()["error"]["code"]) == (status, code), r.text


def test_models_lists_only_the_public_alias(public) -> None:
    deps = Deps(public)
    r = TestClient(create_app(deps)).get("/v1/models")
    assert r.status_code == 200
    assert [(m["id"], m["owned_by"]) for m in r.json()["data"]] == [("chord-1-poly", "chord")]
    health = TestClient(create_internal_app(deps)).get("/internal/health").json()
    assert health["capabilities"] == {
        "input": {"image": False},
        "output": {"image": False, "image_edit": False, "image_variation": False},
        "image_chat_routable": False,
        "video": False,
    }
