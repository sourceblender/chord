"""Configured image workflow serves routed chat."""

import base64
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from chord import manifest, registry
from chord.config import ImageWorkflowConfig, Settings
from chord.server import Deps, create_app, load_specialists
from chord.internal_api import create_app as create_internal_app
from chord.__main__ import effective_config_view

from test_skeleton import FakeUpstream


def test_loading_image_provider_imports_only_registered_specialists():
    process = subprocess.run(
        [sys.executable, "-c", """
import sys
from chord.server import load_specialists
load_specialists()
assert 'chord.specialists.image' in sys.modules
"""],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr


class ImageRouter:
    async def ainvoke(self, _messages):
        class Reply:
            content = json.dumps({"route": "image", "intent": "a blue mug", "constraints": ["no text"]})
        return Reply()


class PromptWriter:
    """The chat model the user is talking to, writing the render prompt."""
    seen: list = []

    async def ainvoke(self, messages):
        PromptWriter.seen = messages

        class Reply:
            content = "A blue ceramic mug on a wooden table, soft morning light"
        return Reply()


def _png() -> bytes:
    output = BytesIO()
    metadata = PngInfo()
    metadata.add_text("workflow", "private-checkpoint-and-graph")
    Image.new("RGB", (1024, 1024), "blue").save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


def test_experimental_image_route_without_a_provider_is_unavailable(tmp_path):
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        enabled_routes=frozenset({"image"}))
    upstream = FakeUpstream()
    deps = Deps(settings, upstream=upstream, model=lambda _: ImageRouter())
    response = TestClient(create_app(deps)).post("/v1/chat/completions", json={
        "model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a blue mug"}],
    })
    assert response.status_code == 200, response.text
    traces = list(Path(settings.trace_dir).glob("*.jsonl"))
    trace = json.loads(traces[0].read_text().splitlines()[-1])
    assert trace["route_decision"] == "image"
    assert trace["route_unavailable"] == "image"
    assert "specialist" not in trace
    assert "You can make pictures" not in upstream.bodies[-1]["messages"][0]["content"]


@pytest.mark.parametrize("bad_png", [False, True])
def test_configured_chat_image_uses_generic_workflow(
    tmp_path, monkeypatch, bad_png
):
    package = Path(__file__).resolve().parents[1] / "src" / "chord"
    monkeypatch.setattr(manifest, "PATH", package / "manifest.yaml")
    monkeypatch.setattr(registry, "PATH", package / "registry.yaml")
    manifest.load.cache_clear()
    load_specialists()
    workflow = tmp_path / "still.json"
    workflow.write_text(json.dumps({
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }))
    settings = Settings(
        data_dir=tmp_path, router_enabled=True, enabled_routes=frozenset(),
        persona_model="chat-model", router_model="router-model",
        image_comfy_base_url="http://localhost:8188",
        image_workflow=ImageWorkflowConfig(workflow, tmp_path, "6", "9"),
    )
    calls = []
    upstream = FakeUpstream()

    class Backend:
        async def render(self, prompt):
            calls.append(prompt)
            return (b"invalid" if bad_png else _png()), "comfy-prompt-1"

        async def aclose(self):
            pass

    try:
        deps = Deps(settings, upstream=upstream,
                    model=lambda n: ImageRouter() if n == settings.router_model else PromptWriter(),
                    image_backend=Backend())
        health = TestClient(create_internal_app(deps)).get("/internal/health").json()
        assert health["capabilities"] == {
            "input": {"image": False},
            "output": {"image": True, "image_edit": False, "image_variation": False},
            "image_chat_routable": True,
            "video": False,
        }
        assert effective_config_view(settings, yaml_mode=True)["routes"]["image"]["chat_routable"] is True
        empty_registry = tmp_path / "empty-registry.yaml"
        empty_registry.write_text("capabilities: []\n")
        monkeypatch.setattr(registry, "PATH", empty_registry)
        assert effective_config_view(settings, yaml_mode=True)["routes"]["image"]["chat_routable"] is False
        monkeypatch.setattr(registry, "PATH", package / "registry.yaml")
        response = TestClient(create_app(deps)).post("/v1/chat/completions", json={
            "model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a blue mug"}],
        })
        assert response.status_code == 200, response.text
        # The chat model wrote the prompt; the router's brief is not the prompt.
        assert calls == ["A blue ceramic mug on a wooden table, soft morning light"]
        assert "draw a blue mug" in PromptWriter.seen[-1].content
        assert "make pictures" in upstream.bodies[-1]["messages"][0]["content"]
        content = response.json()["choices"][0]["message"]["content"]
        assert ("![image](data:image/png;base64," in content) is not bad_png
        if not bad_png:
            encoded = content.split("data:image/png;base64,", 1)[1].split(")", 1)[0]
            assert b"private-checkpoint-and-graph" not in base64.b64decode(encoded)
        traces = list(Path(settings.trace_dir).glob("*.jsonl"))
        trace = json.loads(traces[0].read_text().splitlines()[-1])
        assert trace["route_decision"] == "image"
        assert trace.get("route_unavailable") is None
        assert trace["image_backend"] == "comfyui"
        assert trace["image_prompt_source"] == "chat_model"
        assert trace["image_prompt_model"] == settings.persona_model
        assert (trace["result_status"] == "completed") is not bad_png
    finally:
        manifest.load.cache_clear()


@pytest.mark.parametrize("door", ["chat_stream", "responses", "responses_stream"])
def test_chat_model_prompt_reaches_comfy_and_the_image_is_delivered_on_every_door(tmp_path, monkeypatch, door):
    package = Path(__file__).resolve().parents[1] / "src" / "chord"
    monkeypatch.setattr(manifest, "PATH", package / "manifest.yaml")
    monkeypatch.setattr(registry, "PATH", package / "registry.yaml")
    manifest.load.cache_clear()
    load_specialists()
    workflow = tmp_path / "still.json"
    workflow.write_text(json.dumps({
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }))
    settings = Settings(
        data_dir=tmp_path, router_enabled=True, enabled_routes=frozenset(),
        persona_model="chat-model", router_model="router-model",
        image_comfy_base_url="http://localhost:8188",
        image_workflow=ImageWorkflowConfig(workflow, tmp_path, "6", "9"),
    )
    calls = []

    class Backend:
        async def render(self, prompt):
            calls.append(prompt)
            return _png(), "comfy-prompt-1"

        async def aclose(self):
            pass

    try:
        deps = Deps(settings, upstream=FakeUpstream(),
                    model=lambda n: ImageRouter() if n == settings.router_model else PromptWriter(),
                    image_backend=Backend())
        client = TestClient(create_app(deps))
        if door == "chat_stream":
            with client.stream("POST", "/v1/chat/completions", json={
                "model": "chord-1-poly", "stream": True,
                "messages": [{"role": "user", "content": "draw a blue mug"}],
            }) as r:
                assert r.status_code == 200
                body = "".join(r.iter_text())
            assert "data:image/png;base64," in body
        else:
            payload = {"model": "chord-1-poly", "input": "draw a blue mug", "tools": [{"type": "image_generation"}]}
            if door == "responses":
                r = client.post("/v1/responses", json=payload)
                assert r.status_code == 200, r.text
                assert "image_generation_call" in [i["type"] for i in r.json()["output"]]
            else:
                with client.stream("POST", "/v1/responses", json={**payload, "stream": True}) as r:
                    assert r.status_code == 200
                    body = "".join(r.iter_text())
                assert "image_generation_call" in body
        assert calls == ["A blue ceramic mug on a wooden table, soft morning light"]
        trace = json.loads(sorted(Path(settings.trace_dir).glob("*.jsonl"))[0].read_text().splitlines()[-1])
        assert trace["image_prompt_source"] == "chat_model" and trace["result_status"] == "completed"
    finally:
        manifest.load.cache_clear()
