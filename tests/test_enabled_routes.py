"""The install owner enables a specialist route; a certified record never decides it."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chord.config import ConfigurationError, Settings
from chord.server import Deps, create_app
from chord.specialists import search as S
from test_progress import last_trace
from test_search import ASK, QueryModel, SearchRouter
from test_skeleton import FakeUpstream

YAML = """
endpoints:
  local:
    type: openai-chat
    url: http://localhost:11434/v1
    model: example-model
    auth: null
routing:
  chat: {endpoint: local}
  router: {endpoint: local}
"""


def write(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "chord.yaml"
    path.write_text(YAML + extra)
    return path


def test_yaml_enables_its_own_routes_and_still_ignores_stale_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENTAL_ROUTES", "image,search,audio")
    settings = Settings.from_yaml(write(tmp_path, "enabled_routes: [search]\n"))
    settings.validate_startup()
    assert settings.enabled_routes == frozenset({"search"})


def test_yaml_without_enabled_routes_enables_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENABLED_ROUTES", "search")
    assert Settings.from_yaml(write(tmp_path)).enabled_routes == frozenset()


@pytest.mark.parametrize("value", ["search", "[1]", "['']", "{search: true}"])
def test_yaml_enabled_routes_must_be_a_list_of_names(tmp_path, value) -> None:
    with pytest.raises(ConfigurationError, match="enabled_routes must be a list"):
        Settings.from_yaml(write(tmp_path, f"enabled_routes: {value}\n"))


def test_env_enabled_routes_and_the_earlier_name(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENTAL_ROUTES", "audio")
    assert Settings().enabled_routes == frozenset({"audio"})
    monkeypatch.setenv("ENABLED_ROUTES", "search, audio")
    assert Settings().enabled_routes == frozenset({"search", "audio"})


def test_an_explicitly_empty_enabled_routes_disables_the_earlier_name(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENTAL_ROUTES", "search,audio")
    monkeypatch.setenv("ENABLED_ROUTES", "")
    assert Settings().enabled_routes == frozenset()


@pytest.mark.parametrize("source", ["yaml", "env"])
def test_an_unknown_route_fails_startup_in_either_mode(tmp_path, monkeypatch, source) -> None:
    if source == "yaml":
        settings = Settings.from_yaml(write(tmp_path, "enabled_routes: [search, telepathy]\n"))
    else:
        monkeypatch.setenv("ENABLED_ROUTES", "search,telepathy")
        settings = Settings(data_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="not in the registry: telepathy"):
        settings.validate_startup()


def _client(tmp_path, monkeypatch, enabled: frozenset, certify: bool):
    async def fake_search(query, key, transport=None, **options):
        fake_search.calls = getattr(fake_search, "calls", 0) + 1
        return "brave", [S.Hit("A title", "https://example.com/a", "A snippet.")], []
    monkeypatch.setattr(S, "search", fake_search)
    settings = Settings(data_dir=tmp_path, router_enabled=True, enabled_routes=enabled)
    deps = Deps(settings, upstream=FakeUpstream(),
                model=lambda n: SearchRouter() if n == settings.router_model else QueryModel())
    if certify:
        cap = deps.capabilities["search"]
        cap.certified = {"passed_at": "2026-10-04", "model": cap.model, "prompt_version": cap.prompt_version}
        assert cap.routable
    return TestClient(create_app(deps)), settings, fake_search


def ask(client) -> None:
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": ASK}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200


def test_a_certified_record_does_not_turn_a_route_on(tmp_path, monkeypatch) -> None:
    client, settings, fake = _client(tmp_path, monkeypatch, frozenset(), certify=True)
    ask(client)
    assert getattr(fake, "calls", 0) == 0
    assert last_trace(settings).get("specialist") != "search"


def test_an_enabled_route_runs_without_any_certified_record(tmp_path, monkeypatch) -> None:
    client, settings, fake = _client(tmp_path, monkeypatch, frozenset({"search"}), certify=False)
    ask(client)
    trace = last_trace(settings)
    assert fake.calls == 1
    assert trace["specialist"] == "search" and trace["result_status"] == "completed"
    assert trace["experimental_route"] == "search"  # labelled as untested, never refused


def test_a_certified_enabled_route_is_not_labelled_untested(tmp_path, monkeypatch) -> None:
    client, settings, fake = _client(tmp_path, monkeypatch, frozenset({"search"}), certify=True)
    ask(client)
    trace = last_trace(settings)
    assert fake.calls == 1 and "experimental_route" not in trace
