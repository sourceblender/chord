"""Version 2 config names each job: chat (main), helper, fast, and a separate dispatcher.

Version 1 and env-only installs keep their earlier bindings exactly (docs/configure.md).
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from chord import manifest
from chord.backend_preflight import check
from chord.config import ConfigurationError, Settings
from chord.contract import Job
from chord.server import Deps, create_app
from chord.specialists import SpecialistContext
from chord.specialists import search as S
from chord.trace import Trace
from test_skeleton import FakeUpstream

PACKAGED_MANIFEST = Path(__file__).resolve().parents[1] / "src" / "chord" / "manifest.yaml"
MAIN = "  main: {type: openai-chat, url: http://main.test/v1, model: main-model, auth: null}\n"
SMALL = "  small: {type: openai-chat, url: http://small.test/v1, model: small-model, auth: null}\n"
QUICK = "  quick: {type: openai-chat, url: http://quick.test/v1, model: quick-model, auth: null, thinking: passthrough}\n"


def load(tmp_path: Path, text: str) -> Settings:
    path = tmp_path / "chord.yaml"
    path.write_text(text)
    return Settings.from_yaml(path)


def v2(tmp_path, routing="  chat: {endpoint: main}\n", endpoints=MAIN, extra=""):
    return load(tmp_path, f"version: 2\nendpoints:\n{endpoints}routing:\n{routing}{extra}")


# --- defaults: one main model is enough, helper and fast default independently ---

def test_a_main_only_file_resolves_every_writer_to_main_and_no_dispatch(tmp_path):
    s = v2(tmp_path)
    s.validate_startup()
    assert (s.router_model, s.router_base_url) == ("main-model", "http://main.test/v1")
    assert s.tier_model("fast") == s.tier_model("persona") == "main-model"
    assert s.router_enabled is False and not s.helper_explicit and not s.fast_explicit


def test_a_helper_does_not_move_fast(tmp_path):
    s = v2(tmp_path, "  chat: {endpoint: main}\n  helper: {endpoint: small}\n", MAIN + SMALL)
    assert s.router_model == "small-model" and s.tier_model("fast") == "main-model"


def test_a_fast_writer_does_not_move_the_helper(tmp_path):
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: small}\n", MAIN + SMALL)
    assert s.router_model == "main-model" and s.tier_model("fast") == "small-model"


def test_dispatch_by_helper_without_a_helper_is_the_main_model(tmp_path):
    s = v2(tmp_path, extra="dispatch: {by: helper}\n")
    assert s.router_enabled and s.router_backend == "llm" and s.router_model == "main-model"


# --- one form per file ---

@pytest.mark.parametrize("routing, extra, message", [
    ("  chat: {endpoint: main}\n  router: {endpoint: main}\n", "", "routing.router is the version 1 name"),
])
def test_version_two_refuses_the_version_one_name(tmp_path, routing, extra, message):
    with pytest.raises(ConfigurationError, match=message):
        v2(tmp_path, routing, extra=extra)


@pytest.mark.parametrize("addition, message", [
    ("  helper: {endpoint: main}\n", "routing.helper needs version: 2"),
    ("  fast: {endpoint: main}\n", "routing.fast needs version: 2"),
])
def test_version_one_refuses_the_version_two_names(tmp_path, addition, message):
    with pytest.raises(ConfigurationError, match=message):
        load(tmp_path, f"version: 1\nendpoints:\n{MAIN}routing:\n  chat: {{endpoint: main}}\n{addition}")


def test_version_one_refuses_a_dispatch_block(tmp_path):
    with pytest.raises(ConfigurationError, match="dispatch needs version: 2"):
        load(tmp_path, f"endpoints:\n{MAIN}routing:\n  chat: {{endpoint: main}}\ndispatch: {{by: none}}\n")


@pytest.mark.parametrize("block, message", [
    ("{by: telepathy}", "dispatch.by must be classifier, helper or none"),
    ("{by: classifier}", "needs classifier_url"),
    ("{by: helper, classifier_url: 'http://c.test/route'}", "need by: classifier"),
    ("{by: none, classifier_timeout_s: 1}", "need by: classifier"),
    ("{by: classifier, classifier_url: 'http://c.test/route', classifier_timeout_s: 0}", "greater than zero"),
    ("{by: classifier, classifier_url: 'http://c.test/route', classifier_timeout_s: .nan}", "greater than zero"),
    ("{by: classifier, classifier_url: 'http://c.test/route', classifier_timeout_s: true}", "greater than zero"),
    ("{classifier_url: 'http://c.test/route'}", "dispatch needs by"),
    ("{by: helper, colour: blue}", "dispatch needs by"),
    ("helper", "dispatch needs by"),
])
def test_an_invalid_dispatch_block_is_refused_naming_the_key(tmp_path, block, message):
    with pytest.raises(ConfigurationError, match=message):
        v2(tmp_path, extra=f"dispatch: {block}\n")


@pytest.mark.parametrize("routing, message", [
    ("  chat: {endpoint: main}\n  helper: {endpoint: nowhere}\n", "routing.helper names unknown endpoint"),
    ("  chat: {endpoint: main}\n  fast: {endpoint: nowhere}\n", "routing.fast names unknown endpoint"),
    ("  chat: {endpoint: main}\n  fast: {endpoint: comfy}\n", "routing.fast cannot use endpoint type"),
])
def test_a_writer_must_name_a_chat_endpoint(tmp_path, routing, message):
    endpoints = MAIN + "  comfy: {type: comfyui, url: 'http://comfy.test:8188'}\n"
    with pytest.raises(ConfigurationError, match=message):
        v2(tmp_path, routing, endpoints)


@pytest.mark.parametrize("url, ok", [
    ("not-a-url", False), ("http://quick.test:notaport/v1", False), ("http://quick.test:70000/v1", False),
    ("ftp://quick.test/v1", False), ("http://quick.test:8000/v1", True),
])
def test_the_fast_endpoint_url_is_validated_like_the_others(tmp_path, url, ok):
    quick = f"  quick: {{type: openai-chat, url: '{url}', model: quick-model, auth: null}}\n"
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: quick}\n", MAIN + quick)
    if ok:
        s.validate_startup()
    else:
        with pytest.raises(ConfigurationError, match="routing.fast endpoint must be an absolute http"):
            s.validate_startup()


def test_one_model_name_on_two_addresses_is_refused_across_all_writers(tmp_path):
    clash = "  other: {type: openai-chat, url: http://elsewhere.test/v1, model: main-model, auth: null}\n"
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: other}\n", MAIN + clash)
    with pytest.raises(ConfigurationError, match="cannot identify two different direct routes"):
        s.validate_startup()


# --- dispatch comes from one place ---

def test_version_two_dispatch_ignores_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_ENABLED", "true")
    monkeypatch.setenv("ROUTER_BACKEND", "llm")
    monkeypatch.setenv("ROUTER_CLASSIFIER_URL", "http://stale.test/route")
    s = v2(tmp_path, extra="dispatch: {by: classifier, classifier_url: 'http://c.test/route', classifier_timeout_s: 2}\n")
    assert (s.router_enabled, s.router_backend, s.router_classifier_url, s.router_classifier_timeout_s) == (
        True, "classifier", "http://c.test/route", 2.0)
    assert v2(tmp_path).router_enabled is False  # no block under ROUTER_ENABLED=true: none


def test_a_version_one_house_file_keeps_every_binding(tmp_path, monkeypatch):
    """Today's live shape: v1, a separate router endpoint, dispatch from env."""
    monkeypatch.setenv("ROUTER_ENABLED", "true")
    monkeypatch.setenv("ROUTER_BACKEND", "classifier")
    monkeypatch.setenv("ROUTER_CLASSIFIER_URL", "http://c.test/route")
    s = load(tmp_path, """version: 1
endpoints:
  chat: {type: openai-chat, url: http://main.test/v1, model: main-model, auth: null, thinking: qwen_chat_template}
  router: {type: openai-chat, url: http://small.test/v1, model: small-model, auth: null, thinking: passthrough}
routing:
  chat: {endpoint: chat}
  router: {endpoint: router}
""")
    s.validate_startup()
    assert s.config_version == 1 and s.router_model == "small-model"
    assert s.tier_model("fast") == s.tier_model("router") == "small-model"  # fast follows router, as before
    assert (s.router_enabled, s.router_backend, s.router_classifier_url) == (True, "classifier", "http://c.test/route")
    # The fast reply keeps the persona's thinking mode in v1, as before.
    assert s.tier_thinking_mode("router") == s.tier_thinking_mode("fast") == "qwen_chat_template"


def test_an_env_only_install_keeps_fast_on_the_router(monkeypatch):
    s = Settings(persona_model="main-model", persona_base_url="http://main.test/v1",
                 router_model="small-model", router_base_url="http://small.test/v1",
                 persona_thinking_mode="qwen_chat_template", router_thinking_mode="passthrough")
    assert s.config_version == 0 and s.tier_model("fast") == "small-model"
    assert s.tier_thinking_mode("router") == "qwen_chat_template"


# --- the fast endpoint owns its thinking in v2 ---

def test_a_version_two_fast_reply_uses_the_fast_endpoints_thinking(tmp_path):
    main = "  main: {type: openai-chat, url: http://main.test/v1, model: main-model, auth: null, thinking: qwen_chat_template}\n"
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: quick}\n", main + QUICK)
    assert s.tier_thinking_mode("fast") == "passthrough"
    assert s.tier_thinking_mode("persona") == s.tier_thinking_mode(None) == "qwen_chat_template"


SAME_MODEL = ("  main: {type: openai-chat, url: http://main.test/v1, model: main-model, auth: null, thinking: qwen_chat_template}\n"
              "  plain: {type: openai-chat, url: http://main.test/v1, model: main-model, auth: null, thinking: passthrough}\n")


def test_role_not_model_name_decides_fast_thinking(tmp_path):
    """Main and fast: one model at one address, different thinking (Tama, review of 19f4828d)."""
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: plain}\n", SAME_MODEL)
    s.validate_startup()
    assert s.tier_model("fast") == s.tier_model("persona") == "main-model"
    assert s.tier_thinking_mode("fast") == "passthrough"
    assert s.tier_thinking_mode(None) == "qwen_chat_template"


# --- manifest tier slots ---

def test_the_packaged_manifest_names_the_fast_slot():
    from yaml import safe_load
    routes = safe_load(PACKAGED_MANIFEST.read_text())["service_tier_routes"]
    assert {r["slot"] for r in routes.values()} == {"persona", "fast"}
    assert routes["fast"]["slot"] == routes["priority"]["slot"] == "fast"


def test_router_is_the_earlier_spelling_of_the_fast_slot(tmp_path):
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: small}\n", MAIN + SMALL)
    assert s.tier_model("router") == s.tier_model("fast") == "small-model"
    with pytest.raises(ValueError, match="service tier slot 'banana'"):
        s.tier_model("banana")


def test_an_unknown_manifest_slot_stops_startup(tmp_path, monkeypatch):
    bad = tmp_path / "manifest.yaml"
    bad.write_text(PACKAGED_MANIFEST.read_text().replace("slot: fast", "slot: banana"))
    monkeypatch.setattr(manifest, "PATH", bad)
    manifest.load.cache_clear()
    try:
        with pytest.raises(ConfigurationError, match="unknown slots: banana"):
            v2(tmp_path).validate_startup()
    finally:
        monkeypatch.undo()
        manifest.load.cache_clear()


# --- every chat entry point answers a fast-tier request with the fast model ---

@pytest.mark.parametrize("door", ["chat", "chat_stream", "responses", "responses_stream"])
def test_every_entry_point_sends_a_fast_request_to_the_fast_model(tmp_path, monkeypatch, door, request):
    monkeypatch.setattr(manifest, "PATH", PACKAGED_MANIFEST)
    manifest.load.cache_clear()
    request.addfinalizer(manifest.load.cache_clear)
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: quick}\n", MAIN + QUICK)
    from dataclasses import replace
    s = replace(s, data_dir=tmp_path)
    up = FakeUpstream()
    client = TestClient(create_app(Deps(s, upstream=up, model=lambda name: None)))
    if door.startswith("chat"):
        body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}], "service_tier": "fast",
                "stream": door == "chat_stream"}
        path = "/v1/chat/completions"
    else:
        body = {"model": "chord-1-poly", "input": "hi", "service_tier": "fast", "stream": door == "responses_stream"}
        path = "/v1/responses"
    with client.stream("POST", path, json=body) as r:
        assert r.status_code == 200
        "".join(r.iter_text())
    assert up.bodies and up.bodies[-1]["model"] == "quick-model"
    # passthrough on the fast endpoint: no Qwen thinking switch was added for it
    assert "chat_template_kwargs" not in up.bodies[-1]


@pytest.mark.parametrize("tier, switched", [("fast", False), (None, True)])
def test_same_model_main_and_fast_answer_with_their_own_thinking(tmp_path, monkeypatch, request, tier, switched):
    monkeypatch.setattr(manifest, "PATH", PACKAGED_MANIFEST)
    manifest.load.cache_clear()
    request.addfinalizer(manifest.load.cache_clear)
    from dataclasses import replace
    s = replace(v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: plain}\n", SAME_MODEL), data_dir=tmp_path)
    up = FakeUpstream()
    deps = Deps(s, upstream=up, model=lambda name: None)
    client = TestClient(create_app(deps))
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json={**body, **({"service_tier": tier} if tier else {})})
    assert r.status_code == 200
    assert up.bodies[-1]["model"] == "main-model"
    assert ("chat_template_kwargs" in up.bodies[-1]) is switched
    from chord.server import create_internal_app
    record = TestClient(create_internal_app(deps)).get(f"/internal/traces/{r.headers['x-request-id']}").json()
    assert record["dispatch_source"] == "yaml"
    if tier:
        assert record["fast_model"] == "main-model" and record["reply_thinking"] == "passthrough"
    else:
        assert "fast_model" not in record


# --- search: the helper writes the query in v2; v1/env keep the registry's writer ---

class Writer:
    def __init__(self, name, log):
        self.name, self.log, self.bound = name, log, None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    async def ainvoke(self, messages):
        self.log.append((self.name, self.bound))
        return SimpleNamespace(content="blue mugs")


def run_search(settings, monkeypatch, registry_model="house-chat"):
    log: list = []
    monkeypatch.setattr(S, "load", lambda: {"search": SimpleNamespace(model=registry_model)})

    async def fake_search(query, key, transport=None, **options):
        return "brave", [S.Hit("A", "https://example.com/a", "a")], []
    monkeypatch.setattr(S, "search", fake_search)
    ctx = SpecialistContext(settings=settings, artifacts=SimpleNamespace(),
                            trace=Trace(persona_id="generic", model_id_requested="chord-1-poly"),
                            model=lambda name: Writer(name, log))
    job = Job(job_id="j", persona_id="generic", intent="find blue mugs",
              conversation=[{"role": "user", "text": "find blue mugs"}])
    asyncio.run(S.run(job, ctx))
    return log


def test_version_two_search_query_is_the_helpers_whatever_the_registry_says(tmp_path, monkeypatch):
    s = v2(tmp_path, "  chat: {endpoint: main}\n  helper: {endpoint: small}\n", MAIN + SMALL)
    from dataclasses import replace
    s = replace(s, router_thinking_mode="qwen_chat_template")
    assert run_search(s, monkeypatch) == [
        ("small-model", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})]


def test_a_main_only_search_query_binds_thinking_off(tmp_path, monkeypatch):
    """helper = chat: the client factory's name-keyed switch skips this case."""
    main = "  main: {type: openai-chat, url: http://main.test/v1, model: main-model, auth: null, thinking: qwen_chat_template}\n"
    s = v2(tmp_path, endpoints=main)
    assert run_search(s, monkeypatch) == [
        ("main-model", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})]


@pytest.mark.parametrize("version", [0, 1, 2])
def test_a_writer_that_cannot_be_built_falls_back_to_the_users_words(monkeypatch, version):
    """Building the writer sits inside the bounded fallback, as before (Tama, review of 19f4828d)."""
    seen = []
    monkeypatch.setattr(S, "load", lambda: {"search": SimpleNamespace(model="gone")})

    async def fake_search(query, key, transport=None, **options):
        seen.append(query)
        return "brave", [S.Hit("A", "https://example.com/a", "a")], []
    monkeypatch.setattr(S, "search", fake_search)

    def no_backend(name):
        raise ValueError(f"model {name!r} has no configured backend")
    s = Settings(persona_model="m", persona_base_url="http://m.test/v1", router_model="m",
                 router_base_url="http://m.test/v1", config_version=version)
    ctx = SpecialistContext(settings=s, artifacts=SimpleNamespace(),
                            trace=Trace(persona_id="generic", model_id_requested="chord-1-poly"), model=no_backend)
    job = Job(job_id="j", persona_id="generic", intent="x", conversation=[{"role": "user", "text": "find blue mugs"}])
    result = asyncio.run(S.run(job, ctx))
    assert seen == ["find blue mugs"] and result.status.value == "completed"


@pytest.mark.parametrize("version", [0, 1])
def test_version_one_and_env_keep_the_registry_search_writer(monkeypatch, version):
    s = Settings(persona_model="main-model", persona_base_url="http://main.test/v1",
                 router_model="small-model", router_base_url="http://small.test/v1",
                 router_thinking_mode="qwen_chat_template", config_version=version)
    assert run_search(s, monkeypatch, registry_model="registry-literal") == [("registry-literal", None)]


# --- preflight: each distinct writer target once; the classifier gets a lane request ---

def answer_all(seen):
    def answer(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, json.loads(request.content) if request.content else None,
                     request.headers.get("Authorization")))
        if request.url.path.endswith("/route"):
            return httpx.Response(200, json={"route": "chat"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
    return answer


def test_preflight_probes_each_distinct_writer_once(tmp_path):
    seen: list = []
    s = v2(tmp_path)  # main-only: three writers, one target
    with httpx.Client(transport=httpx.MockTransport(answer_all(seen))) as client:
        assert check(s, client) == []
    assert [h for h, _, _ in seen] == ["main.test"]


def test_preflight_probes_each_model_served_from_one_address(tmp_path):
    """Distinct targets are URL + model (+ credential). Two credentials on one URL
    are refused by credential_for before preflight, so only the model axis is
    exercised here; the credential axis needs a separate address (next test)."""
    same_host = "  small: {type: openai-chat, url: http://main.test/v1, model: small-model, auth: null}\n"
    s = v2(tmp_path, "  chat: {endpoint: main}\n  helper: {endpoint: small}\n", MAIN + same_host)
    seen: list = []
    with httpx.Client(transport=httpx.MockTransport(answer_all(seen))) as client:
        assert check(s, client) == []
    assert sorted(body["model"] for _, body, _ in seen) == ["main-model", "small-model"]


def test_preflight_sends_each_target_its_own_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("QUICK_KEY", "k-quick")
    quick = "  quick: {type: openai-chat, url: http://quick.test/v1, model: quick-model, auth: '${QUICK_KEY}'}\n"
    s = v2(tmp_path, "  chat: {endpoint: main}\n  fast: {endpoint: quick}\n", MAIN + quick)
    seen: list = []
    with httpx.Client(transport=httpx.MockTransport(answer_all(seen))) as client:
        assert check(s, client) == []
    assert {(h, a) for h, _, a in seen} == {("main.test", None), ("quick.test", "Bearer k-quick")}


def test_preflight_sends_the_classifier_a_lane_request_never_a_chat_one(tmp_path):
    s = v2(tmp_path, extra="dispatch: {by: classifier, classifier_url: 'http://classify.test/v1/route'}\n")
    seen: list = []
    with httpx.Client(transport=httpx.MockTransport(answer_all(seen))) as client:
        assert check(s, client) == []
    classifier = [body for host, body, _ in seen if host == "classify.test"]
    assert classifier == [{"text": "user: Reply OK"}]


def test_a_classifier_that_answers_no_lane_fails_preflight(tmp_path):
    s = v2(tmp_path, extra="dispatch: {by: classifier, classifier_url: 'http://classify.test/v1/route'}\n")

    def answer(request):
        if request.url.host == "classify.test":
            return httpx.Response(200, json={"route": "poetry"})
        return httpx.Response(200, json={"choices": [{}]})
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        assert check(s, client) == ["classifier"]


def test_a_shared_failed_target_still_names_every_writer_on_it(tmp_path):
    s = v2(tmp_path)

    def answer(request):
        return httpx.Response(503)
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        assert check(s, client) == ["persona", "router", "fast"]


# --- the guide's examples load as documented ---

def test_the_guides_examples_load(tmp_path):
    import re
    guide = (Path(__file__).resolve().parents[1] / "docs" / "configure.md").read_text()
    blocks = re.findall(r"```yaml\n(.*?)```", guide, re.S)
    minimal, full = (load(tmp_path, blocks[0]), load(tmp_path, blocks[1]))
    assert minimal.router_model == minimal.tier_model("fast") == minimal.persona_model and not minimal.router_enabled
    assert full.router_model == "my-small-model" and full.tier_model("fast") == "my-main-model"
    assert full.router_backend == "classifier"


@pytest.mark.parametrize("version, baked", [(2, None), (1, {"chat_template_kwargs": {"enable_thinking": False}})])
def test_a_cached_client_carries_a_thinking_switch_only_in_version_one(tmp_path, version, baked):
    """v2 binds thinking per call by role, so one model's cached client can serve
    helper (off) and fast (passthrough) without carrying either; v1 keeps the
    earlier name-keyed switch."""
    from dataclasses import replace
    s = Settings(data_dir=tmp_path, persona_model="main-model", persona_base_url="http://main.test/v1",
                 router_model="small-model", router_base_url="http://small.test/v1",
                 router_thinking_mode="qwen_chat_template", config_version=version)
    deps = Deps(replace(s), upstream=FakeUpstream())
    assert deps.model("small-model").extra_body == baked
