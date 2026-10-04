from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from chord.config import ConfigurationError, Settings
from chord.graph import router_client, thinking_switch
from chord.router import route
from chord.server import Deps, create_app
from test_skeleton import FakeUpstream


def test_env_only_defaults_are_portable_without_backend_options(monkeypatch) -> None:
    for key in ("PERSONA_THINKING_MODE", "ROUTER_THINKING_MODE", "ROUTER_JSON_MODE"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings()
    assert settings.persona_thinking_mode == "passthrough"
    assert settings.router_thinking_mode == "passthrough"
    assert settings.router_json_mode is False
    body = {"reasoning_effort": "low"}
    assert thinking_switch(body.copy()) == body


def test_env_only_backend_options_can_be_selected_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("PERSONA_THINKING_MODE", "qwen_chat_template")
    monkeypatch.setenv("ROUTER_THINKING_MODE", "qwen_chat_template")
    monkeypatch.setenv("ROUTER_JSON_MODE", "true")
    settings = Settings()
    assert settings.persona_thinking_mode == "qwen_chat_template"
    assert settings.router_thinking_mode == "qwen_chat_template"
    assert settings.router_json_mode is True
    assert thinking_switch({"reasoning_effort": "low"}, settings.persona_thinking_mode) == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}
    }


def test_public_default_manifest_is_text_only(monkeypatch) -> None:
    from chord import manifest, registry

    root = Path(__file__).resolve().parents[1]
    public = root / "src/chord/manifest.yaml"
    public_registry = root / "src/chord/registry.yaml"
    assert all(cap.model == "none" for cap in registry.load(public_registry).values())
    monkeypatch.setattr(manifest, "PATH", public)
    manifest.load.cache_clear()
    try:
        assert manifest.load()["release"] == "text-only"
        public_manifest = manifest.load()
        assert not public_manifest["input"]["image"]
        assert not public_manifest["input"]["audio"]
        assert not public_manifest["output"]["audio"]
        assert public_manifest["limits"]["max_input_tokens"] is None
        assert manifest.models_entry("chord-1-poly")["owned_by"] == "chord"
    finally:
        manifest.load.cache_clear()


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "chord.yaml"
    path.write_text(content)
    return path


def test_operator_video_workflows_load_from_yaml_and_refuse_escape(tmp_path: Path) -> None:
    directory = tmp_path / "workflows"
    directory.mkdir()
    for kind in ("t2v", "r2v"):
        graph = {"input": {"class_type": "VideoSampler", "inputs": {
            "text": "sample", "width": 1, "height": 1, "seconds": 1, "seed": 0,
            **({"image": "sample.png"} if kind == "r2v" else {}),
        }}, "save": {"class_type": "SaveVideo", "inputs": {"video": ["input", 0]}}}
        (directory / f"{kind}.json").write_text(json.dumps(graph))
    config = write(tmp_path, """
version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: 'http://127.0.0.1:11434/v1', model: example}
  comfy: {type: comfyui, url: 'http://127.0.0.1:8188'}
routing:
  video:
    endpoint: comfy
    t2v: {workflow: t2v.json, output_node_id: save, prompt_node_id: input,
          width_node_id: input, height_node_id: input, duration_node_id: input, seed_node_id: input,
          duration_unit: frames, fps: 24}
    r2v: {workflow: r2v.json, output_node_id: save, prompt_node_id: input,
          width_node_id: input, height_node_id: input, duration_node_id: input, seed_node_id: input,
          reference_node_id: input}
""")
    settings = Settings.from_yaml(config)
    settings.validate_startup()
    assert settings.comfy_base_url == "http://127.0.0.1:8188"
    assert set(settings.video_workflows or {}) == {"t2v", "r2v"}
    from chord.__main__ import effective_config_view
    video_view = effective_config_view(settings, yaml_mode=True)["routes"]["video"]
    assert video_view["configured"] is True
    assert video_view["kinds"] == ["r2v", "t2v"]
    assert settings.video_workflows["t2v"].load().for_request(
        prompt="waves", width=720, height=1280, seconds=8, seed=1
    )["input"]["inputs"]["seconds"] == 192
    assert settings.video_workflows["r2v"].load().for_request(
        prompt="waves", width=720, height=1280, seconds=8, seed=1,
        reference_name="uploaded.png")["input"]["inputs"]["image"] == "uploaded.png"
    t2v_only = config.read_text().replace("    r2v: {workflow: r2v.json, output_node_id: save, prompt_node_id: input,\n"
                                           "          width_node_id: input, height_node_id: input, duration_node_id: input, seed_node_id: input,\n"
                                           "          reference_node_id: input}\n", "")
    config.write_text(t2v_only)
    assert set(Settings.from_yaml(config).video_workflows or {}) == {"t2v"}
    (directory / "t2v.json").unlink()
    (directory / "t2v.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ConfigurationError, match="escapes image_workflows_dir"):
        settings.validate_startup()


def test_video_workflow_rejects_non_string_duration_unit_as_configuration(tmp_path: Path) -> None:
    config = write(tmp_path, """
version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: 'http://127.0.0.1:11434/v1', model: example}
  comfy: {type: comfyui, url: 'http://127.0.0.1:8188'}
routing:
  video:
    endpoint: comfy
    t2v: {workflow: t2v.json, output_node_id: save, prompt_node_id: input,
          width_node_id: input, height_node_id: input, duration_node_id: input,
          seed_node_id: input, duration_unit: []}
""")
    with pytest.raises(ConfigurationError, match="duration_unit must be seconds or frames"):
        Settings.from_yaml(config)


def test_one_generic_backend_serves_chat_and_router(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_MODEL", "example-model")
    monkeypatch.setenv("PERSONA_BASE_URL", "https://stale-gateway.test/v1")
    monkeypatch.setenv("PERSONA_API_KEY", "stale-key")
    monkeypatch.setenv("EXPERIMENTAL_ROUTES", "image,search,audio")
    settings = Settings.from_yaml(write(tmp_path, """
endpoints:
  local:
    type: openai-chat
    url: http://localhost:11434/v1
    model: ${LOCAL_MODEL}
    auth: null
routing:
  chat: {endpoint: local}
  router: {endpoint: local}
"""))
    settings.validate_startup()
    assert settings.slot_target("persona") == ("example-model", "http://localhost:11434/v1")
    assert settings.slot_target("router") == settings.slot_target("persona")
    assert settings.persona_thinking_mode == "passthrough"
    assert settings.router_thinking_mode == "passthrough"
    assert settings.router_json_mode is False
    assert settings.stt_model == settings.tts_model == settings.embeddings_model == ""
    with pytest.raises(ValueError, match="no configured backend"):
        settings.base_url_for("unconfigured-model")
    assert settings.enabled_routes == frozenset()
    body = {"reasoning_effort": "low"}
    assert thinking_switch(body, settings.persona_thinking_mode) == {"reasoning_effort": "low"}

    class Bindable:
        def bind(self, **kwargs):
            raise AssertionError(f"unexpected backend-specific bind: {kwargs}")

    model = Bindable()
    assert router_client(model, settings.router_thinking_mode) is model


def test_version_one_and_versionless_compatibility(tmp_path: Path) -> None:
    config = "endpoints: {chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}}\n"
    assert Settings.from_yaml(write(tmp_path, "version: 1\n" + config)).persona_model == "example"
    assert Settings.from_yaml(write(tmp_path, config)).persona_model == "example"
    for version in ("2", "true", "'1'"):
        with pytest.raises(ConfigurationError, match="version must be 1"):
            Settings.from_yaml(write(tmp_path, f"version: {version}\n" + config))


def test_check_config_exits_without_starting_a_server(tmp_path: Path) -> None:
    config = write(tmp_path, """version: 1
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
""")
    result = subprocess.run([sys.executable, "-m", "chord", "--check-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config)},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Chord configuration OK"


def test_comfy_image_workflow_is_operator_data_and_validated_offline(tmp_path: Path) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "still.json").write_text(json.dumps({
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }))
    config = write(tmp_path, """version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: https://comfy.example.test:8188/private/sample-secret-that-must-not-print}
routing:
  image:
    endpoint: renderer
    workflow: still.json
    prompt_node_id: '6'
    output_node_id: '9'
""")
    settings = Settings.from_yaml(config)
    settings.validate_startup()
    assert settings.image_comfy_base_url.endswith("/sample-secret-that-must-not-print")
    assert settings.image_workflow is not None
    assert settings.image_workflow.load().for_request("a tree")["6"]["inputs"]["text"] == "a tree"
    result = subprocess.run([sys.executable, "-m", "chord", "--show-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config)},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    shown = json.loads(result.stdout)
    assert shown["routes"]["image"] == {
        "configured": True, "origin": "https://comfy.example.test:8188",
        "workflow_configured": True, "serving": True, "chat_routable": False,
    }
    assert "sample-secret-that-must-not-print" not in result.stdout + result.stderr
    assert str(workflows) not in result.stdout + result.stderr


@pytest.mark.parametrize("workflow", ["../outside.json", "/tmp/absolute.json"])
def test_image_workflow_cannot_escape_mounted_directory(tmp_path: Path, workflow: str) -> None:
    config = write(tmp_path, f"""version: 1
image_workflows_dir: workflows
endpoints:
  chat: {{type: openai-chat, url: http://localhost:11434/v1, model: example}}
  renderer: {{type: comfyui, url: http://localhost:8188}}
routing:
  image: {{endpoint: renderer, workflow: '{workflow}', prompt_node_id: '6', output_node_id: '9'}}
""")
    if workflow.startswith("/"):
        with pytest.raises(ConfigurationError, match="relative path"):
            Settings.from_yaml(config)
    else:
        with pytest.raises(ConfigurationError, match="escapes image_workflows_dir"):
            Settings.from_yaml(config).validate_startup()


def test_image_workflow_symlink_cannot_escape_mounted_directory(tmp_path: Path) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    (workflows / "linked.json").symlink_to(outside)
    config = write(tmp_path, """version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: http://localhost:8188}
routing:
  image: {endpoint: renderer, workflow: linked.json, prompt_node_id: '6', output_node_id: '9'}
""")
    with pytest.raises(ConfigurationError, match="escapes image_workflows_dir"):
        Settings.from_yaml(config).validate_startup()


def test_image_workflow_bad_binding_fails_before_backend_contact(tmp_path: Path) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "still.json").write_text(json.dumps({
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }))
    config = write(tmp_path, """version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: http://not-running.invalid:8188}
routing:
  image: {endpoint: renderer, workflow: still.json, prompt_node_id: '5', output_node_id: '9'}
""")
    result = subprocess.run([sys.executable, "-m", "chord", "--check-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config)},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 2
    assert "image workflow has no input" in result.stderr
    assert "Traceback" not in result.stderr


def test_image_workflow_fifo_refuses_without_reading(tmp_path: Path) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    os.mkfifo(workflows / "still.json")
    config = write(tmp_path, """version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: http://localhost:8188}
routing:
  image: {endpoint: renderer, workflow: still.json, prompt_node_id: '6', output_node_id: '9'}
""")
    result = subprocess.run([sys.executable, "-m", "chord", "--check-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config)},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 2
    assert "must be a regular file" in result.stderr


def test_image_endpoint_rejects_implicit_basic_auth(tmp_path: Path) -> None:
    config = write(tmp_path, """version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: https://user:secret@comfy.example.test:8188}
routing:
  image: {endpoint: renderer, workflow: still.json, prompt_node_id: '6', output_node_id: '9'}
""")
    with pytest.raises(ConfigurationError, match="must not contain userinfo") as exc:
        Settings.from_yaml(config)
    assert "secret" not in str(exc.value)


def test_show_config_reports_resolved_routes_without_credential_urls(tmp_path: Path) -> None:
    secret = "user:sample-secret-that-must-not-print"
    config = write(tmp_path, """version: 1
endpoints:
  local:
    type: openai-chat
    url: https://user:pass@api.example.test:8443/v1/${TEST_TOKEN}?key=${TEST_TOKEN}#fragment
    model: example-model
    auth: ${TEST_TOKEN}
  vectors:
    type: tei-embeddings
    url: https://vectors.example.test/embeddings
    model: example-embed
    auth: {basic: "${TEST_TOKEN}"}
routing:
  chat: {endpoint: local}
  router: {endpoint: local}
  embeddings: {endpoint: vectors}
""")
    result = subprocess.run([sys.executable, "-m", "chord", "--show-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config), "TEST_TOKEN": secret},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["mode"] == "yaml-v1"
    assert output["routes"]["chat"] == {
        "configured": True, "model": "example-model", "origin": "https://api.example.test:8443",
        "auth": "bearer", "thinking": "passthrough"}
    assert output["routes"]["router"]["model"] == "example-model"
    assert output["routes"]["embeddings"]["auth"] == "basic"
    assert output["routes"]["embeddings"]["origin"] == "https://vectors.example.test"
    assert output["routes"]["stt"]["configured"] is False
    assert output["routes"]["image"]["serving"] is False
    assert secret not in result.stdout + result.stderr
    assert "user:pass" not in result.stdout + result.stderr
    assert "/v1/" not in result.stdout + result.stderr


@pytest.mark.parametrize("url", ["http://host:notaport/v1", "host-without-scheme:8080/v1",
                                     "http://[broken/v1"])
def test_invalid_endpoint_url_fails_offline_validation(tmp_path: Path, url: str) -> None:
    config = write(tmp_path, f"""version: 1
endpoints:
  chat: {{type: openai-chat, url: '{url}', model: example}}
""")
    settings = Settings.from_yaml(config)
    with pytest.raises(ConfigurationError, match="PERSONA_BASE_URL must be an absolute http"):
        settings.validate_startup()

    result = subprocess.run([sys.executable, "-m", "chord", "--show-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config)},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 2
    assert "PERSONA_BASE_URL must be an absolute http" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_yaml_does_not_echo_a_literal_secret(tmp_path: Path) -> None:
    secret = "super-secret-value-for-test"
    config = write(tmp_path, f"version: 1\nendpoints:\n  chat: {{auth: {secret}, url: [\n")
    with pytest.raises(ConfigurationError) as exc:
        Settings.from_yaml(config)
    assert "invalid YAML" in str(exc.value)
    assert secret not in str(exc.value)

    result = subprocess.run([sys.executable, "-m", "chord", "--check-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config)},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 2
    assert "invalid YAML" in result.stderr
    assert "Traceback" not in result.stderr
    assert secret not in result.stdout + result.stderr


def test_generic_text_chat_does_not_send_qwen_fields(tmp_path: Path) -> None:
    settings = Settings.from_yaml(write(tmp_path, """
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: generic-chat}
"""))
    upstream = FakeUpstream()
    deps = Deps(replace(settings, data_dir=tmp_path / "data"),
                upstream=upstream, model=lambda _: None)
    response = TestClient(create_app(deps)).post("/v1/chat/completions", json={
        "model": "chord-1-poly", "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200, response.text
    assert upstream.bodies[0]["model"] == "generic-chat"
    assert upstream.bodies[0]["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in upstream.bodies[0]


@pytest.mark.asyncio
async def test_generic_router_does_not_require_json_response_format() -> None:
    class GenericRouter:
        def bind(self, **kwargs):
            raise AssertionError(f"unsupported JSON mode: {kwargs}")

        async def ainvoke(self, _messages):
            class Reply:
                content = '{"route":"chat"}'
            return Reply()

    decision, _ = await route([{"role": "user", "text": "hi"}], {}, GenericRouter(),
                              json_mode=False)
    assert decision.route == "chat"


def test_yaml_secret_interpolation_and_optional_endpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXAMPLE_TOKEN", "sample-secret")
    settings = Settings.from_yaml(write(tmp_path, """
endpoints:
  chat: {type: openai-chat, url: https://example.test/v1, model: example, auth: "${EXAMPLE_TOKEN}"}
  speech: {type: openai-audio, url: https://speech.test/v1, model: voice}
routing:
  tts: {endpoint: speech}
"""))
    assert settings.persona_api_key == settings.router_api_key == "sample-secret"
    assert settings.tts_model == "voice"
    assert settings.stt_model == ""


def test_chat_route_is_the_default_router_and_embeddings_bearer_is_forwarded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_TOKEN", "test-token")
    settings = Settings.from_yaml(write(tmp_path, """
endpoints:
  local: {type: openai-chat, url: http://localhost:11434/v1, model: local-model}
  vectors: {type: openai-embeddings, url: https://vectors.example.test/v1, model: embed-model, auth: "${EMBEDDINGS_TOKEN}"}
routing:
  chat: {endpoint: local}
  embeddings: {endpoint: vectors}
"""))
    assert settings.slot_target("router") == settings.slot_target("persona")
    assert settings.embeddings_model == "embed-model"
    assert settings.embeddings_auth_header() == "Bearer test-token"


@pytest.mark.parametrize("content,fragment", [
    ("endpoints: {chat: {type: openai-chat, url: http://localhost/v1, model: x, fallback: {url: bad}}}", "unsupported fields"),
    ('endpoints: {chat: {type: openai-chat, url: http://localhost/v1, model: "${MISSING_MODEL}"}}', "needs environment variable"),
    ('endpoints: {chat: {type: openai-chat, url: http://localhost/v1, model: "${MODEL:-fallback}"}}', "unsupported environment interpolation"),
    ("endpoints: {chat: {type: openai-audio, url: http://localhost/v1, model: x}}", "cannot use endpoint type"),
    ('endpoints: {chat: {type: openai-chat, url: http://localhost/v1, model: x, json_mode: "yes"}}', "json_mode must be true or false"),
])
def test_invalid_yaml_fails_closed(tmp_path: Path, monkeypatch, content: str, fragment: str) -> None:
    monkeypatch.delenv("MISSING_MODEL", raising=False)
    with pytest.raises(ConfigurationError, match=fragment):
        Settings.from_yaml(write(tmp_path, content))
