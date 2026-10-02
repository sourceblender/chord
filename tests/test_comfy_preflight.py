"""The deploy-time reachability check, including its own positive control.

A preflight that can only pass is decoration. Each refusal below is exercised,
and `test_a_real_comfyui_passes` is the positive control: it proves the checker
can still say yes, so a green preflight means "reachable", not "the check is
broken in the permissive direction".
"""

from __future__ import annotations

import http.server
import json

import pytest
import threading
from contextlib import contextmanager
from types import SimpleNamespace

from chord import comfy_preflight


@contextmanager
def serving(status: int, body: bytes, content_type: str = "application/json"):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_real_comfyui_passes():
    # POSITIVE CONTROL. Without this, every other test here passes just as
    # happily against a checker that always fails.
    body = json.dumps({"system": {"comfyui_version": "0.36.0"}}).encode()
    with serving(200, body) as url:
        code, message = comfy_preflight.check(url)
    assert code == 0
    assert "0.36.0" in message


def test_probe_messages_do_not_print_secret_url_path():
    body = json.dumps({"system": {"comfyui_version": "0.36.0"}}).encode()
    with serving(200, body) as url:
        code, message = comfy_preflight.check(url + "/private-token")
    assert code == 0
    assert "private-token" not in message
    code, message = comfy_preflight.check("http://127.0.0.1:1/private-token", timeout=1.0)
    assert code == 1
    assert "private-token" not in message


def test_invalid_environment_number_is_a_clean_preflight_refusal(monkeypatch, capsys):
    monkeypatch.delenv("CHORD_CONFIG", raising=False)
    monkeypatch.setenv("PUBLIC_PORT", "invalid-number")
    assert comfy_preflight.main() == 1
    assert "invalid configuration" in capsys.readouterr().err


def test_an_unconfigured_deployment_is_not_a_failure():
    # No ComfyUI is a legitimate deployment: the route is simply not advertised.
    code, message = comfy_preflight.check("")
    assert code == 0
    assert "serves no video" in message


def test_video_preflight_refuses_without_operator_tools(monkeypatch, capsys):
    monkeypatch.setattr(comfy_preflight, "load_settings", lambda: SimpleNamespace(
        video_workflows={"t2v": object()}, comfy_base_url="http://127.0.0.1:1",
        image_workflow=None,
    ))
    monkeypatch.setattr(comfy_preflight, "video_tools_available", lambda: False)
    assert comfy_preflight.main() == 1
    assert "ffmpeg and ffprobe" in capsys.readouterr().err


def test_an_unreachable_host_fails_the_deploy():
    # Nothing is listening on this port; this is the case the whole module exists
    # for, and the one that currently ships green.
    code, message = comfy_preflight.check("http://127.0.0.1:1", timeout=1.0)
    assert code == 1
    assert "unreachable" in message


def test_a_non_200_fails():
    with serving(503, b"busy") as url:
        code, message = comfy_preflight.check(url)
    assert code == 1
    assert "503" in message


def test_a_200_that_is_not_json_fails():
    # A parked page or a proxy error page returns 200 with HTML.
    with serving(200, b"<html>hello</html>", "text/html") as url:
        code, message = comfy_preflight.check(url)
    assert code == 1
    assert "not ComfyUI JSON" in message


def test_json_without_a_version_fails():
    # Something else is listening on the port. 200 and valid JSON are not
    # evidence that it is ComfyUI.
    with serving(200, json.dumps({"system": {}}).encode()) as url:
        code, message = comfy_preflight.check(url)
    assert code == 1
    assert "may not be ComfyUI" in message


def test_base_url_without_an_operator_video_workflow_does_not_advertise_video(monkeypatch):
    monkeypatch.delenv("CHORD_CONFIG", raising=False)
    monkeypatch.setenv("CHORD_API_KEY", "test-key")
    monkeypatch.setenv("PERSONA_MODEL", "example")
    monkeypatch.setenv("ROUTER_MODEL", "example")
    monkeypatch.setenv("PERSONA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ROUTER_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("COMFY_BASE_URL", "http://127.0.0.1:1")
    assert comfy_preflight.main() == 0
    monkeypatch.setenv("COMFY_BASE_URL", "")
    assert comfy_preflight.main() == 0


def test_main_uses_yaml_video_endpoint_over_stale_environment(tmp_path, monkeypatch):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    for kind in ("t2v", "r2v"):
        (workflows / f"{kind}.json").write_text(json.dumps({
            "input": {"class_type": "VideoSampler", "inputs": {
                "text": "sample", "width": 1, "height": 1, "seconds": 1, "seed": 0,
                **({"image": "sample.png"} if kind == "r2v" else {}),
            }},
            "save": {"class_type": "SaveVideo", "inputs": {"video": ["input", 0]}},
        }))
    config = tmp_path / "chord.yaml"
    config.write_text("""version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: http://127.0.0.1:1}
routing:
  video:
    endpoint: renderer
    t2v: {workflow: t2v.json, output_node_id: save, prompt_node_id: input,
          width_node_id: input, height_node_id: input, duration_node_id: input, seed_node_id: input}
    r2v: {workflow: r2v.json, output_node_id: save, prompt_node_id: input,
          width_node_id: input, height_node_id: input, duration_node_id: input, seed_node_id: input,
          reference_node_id: input}
""")
    monkeypatch.setenv("CHORD_CONFIG", str(config))
    monkeypatch.setenv("CHORD_API_KEY", "test-key")
    monkeypatch.setenv("COMFY_BASE_URL", "")
    assert comfy_preflight.main() == 1


def test_main_probes_yaml_still_endpoint_without_video(tmp_path, monkeypatch):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "still.json").write_text(json.dumps({
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }))
    config = tmp_path / "chord.yaml"
    config.write_text("""version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: http://127.0.0.1:1}
routing:
  image: {endpoint: renderer, workflow: still.json, prompt_node_id: '6', output_node_id: '9'}
""")
    monkeypatch.setenv("CHORD_CONFIG", str(config))
    monkeypatch.setenv("CHORD_API_KEY", "test-key")
    assert comfy_preflight.main() == 1


@pytest.mark.parametrize("route", ["image_edit", "image_variation"])
def test_main_probes_yaml_edit_and_variation_endpoints(tmp_path, monkeypatch, route):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "input.json").write_text(json.dumps({
        "10": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "20": {"class_type": "TextEncode", "inputs": {"text": "placeholder"}},
        "40": {"class_type": "SaveImage", "inputs": {"images": ["30", 0]}},
    }))
    prompt = ", prompt_node_id: '20'" if route == "image_edit" else ""
    config = tmp_path / "chord.yaml"
    config.write_text(f"""version: 1
image_workflows_dir: workflows
endpoints:
  chat: {{type: openai-chat, url: http://localhost:11434/v1, model: example}}
  renderer: {{type: comfyui, url: http://127.0.0.1:1}}
routing:
  {route}: {{endpoint: renderer, workflow: input.json, image_node_id: '10'{prompt}, output_node_id: '40'}}
""")
    monkeypatch.setenv("CHORD_CONFIG", str(config))
    monkeypatch.setenv("CHORD_API_KEY", "test-key")
    monkeypatch.setenv("COMFY_BASE_URL", "")
    seen: list[str] = []
    real = comfy_preflight.check
    monkeypatch.setattr(comfy_preflight, "check", lambda url, *a, **k: (seen.append(url), real(url, *a, **k))[1])
    assert comfy_preflight.main() == 1
    assert "http://127.0.0.1:1" in seen
