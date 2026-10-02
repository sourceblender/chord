"""Edits and variations run an operator-supplied ComfyUI workflow, like still images.

No edit or variation graph ships with Chord. routing.image_edit and
routing.image_variation name a workflow under image_workflows_dir and the node
inputs Chord fills; without them both doors answer 503.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from chord import comfy
from chord.comfy_workflow import ComfyImageInputBackend, ComfyImageInputWorkflow, WorkflowError
from chord.config import ConfigurationError, Settings, VideoWorkflowConfig
from chord.server import Deps, create_app
from test_skeleton import PNG, FakeUpstream

EDIT_GRAPH = {
    "10": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
    "20": {"class_type": "TextEncode", "inputs": {"text": "placeholder"}},
    "30": {"class_type": "KSampler", "inputs": {"seed": 0}},
    "40": {"class_type": "SaveImage", "inputs": {"images": ["30", 0]}},
}


def _workflows(tmp_path: Path) -> Path:
    workflows = tmp_path / "workflows"
    workflows.mkdir(exist_ok=True)
    (workflows / "edit.json").write_text(json.dumps(EDIT_GRAPH))
    (workflows / "vary.json").write_text(json.dumps(EDIT_GRAPH))
    return workflows


def _config(tmp_path: Path, routing: str) -> Path:
    _workflows(tmp_path)
    path = tmp_path / "chord.yaml"
    path.write_text("""version: 1
image_workflows_dir: workflows
endpoints:
  chat: {type: openai-chat, url: http://localhost:11434/v1, model: example}
  renderer: {type: comfyui, url: http://comfy.example.test:8188}
routing:
""" + routing)
    return path


EDIT_ROUTE = """  image_edit:
    endpoint: renderer
    workflow: edit.json
    image_node_id: '10'
    prompt_node_id: '20'
    seed_node_id: '30'
    output_node_id: '40'
"""
VARY_ROUTE = """  image_variation:
    endpoint: renderer
    workflow: vary.json
    image_node_id: '10'
    seed_node_id: '30'
    output_node_id: '40'
"""


class FakeComfy:
    """Records what Chord sends; each completed prompt yields one PNG."""

    def __init__(self, hold: asyncio.Event | None = None) -> None:
        self.uploads: list[tuple[bytes, str]] = []
        self.graphs: list[dict] = []
        self.interrupted: list[str] = []
        self.hold = hold

    async def upload_image(self, data: bytes, filename: str) -> str:
        self.uploads.append((data, filename))
        return "stored-input.png"

    async def submit(self, graph: dict) -> str:
        self.graphs.append(graph)
        return f"prompt-{len(self.graphs)}"

    async def poll(self, prompt_id: str) -> tuple[str, float]:
        if self.hold is not None:
            await self.hold.wait()
        return comfy.COMPLETED, 1.0

    async def fetch_pngs(self, prompt_id: str, output_node_id: str | None = None) -> list[bytes]:
        assert output_node_id == "40"
        return [PNG + prompt_id.encode()]

    async def interrupt(self, prompt_id: str) -> bool:
        self.interrupted.append(prompt_id)
        return True

    async def aclose(self) -> None:
        pass


def _edit_workflow(**overrides) -> ComfyImageInputWorkflow:
    kwargs = dict(image_node_id="10", prompt_node_id="20", seed_node_id="30", output_node_id="40")
    kwargs.update(overrides)
    return ComfyImageInputWorkflow(EDIT_GRAPH, **kwargs)


def test_an_edit_fills_only_the_bound_inputs_of_the_operator_graph():
    backend = ComfyImageInputBackend(_edit_workflow(), FakeComfy())
    png = asyncio.run(backend.edit_image(b"source", "cat.png", "add a hat"))
    sent = backend.client.graphs[0]
    assert backend.client.uploads == [(b"source", "cat.png")]
    assert sent["10"]["inputs"]["image"] == "stored-input.png"
    assert sent["20"]["inputs"]["text"] == "add a hat"
    assert isinstance(sent["30"]["inputs"]["seed"], int)
    assert sent["40"] == EDIT_GRAPH["40"]
    assert EDIT_GRAPH["20"]["inputs"]["text"] == "placeholder"       # the loaded graph is never mutated
    assert png == PNG + b"prompt-1"


def test_variations_upload_once_and_submit_once_per_image_with_fresh_seeds():
    backend = ComfyImageInputBackend(_edit_workflow(prompt_node_id=None), FakeComfy())
    pngs = asyncio.run(backend.vary_images(b"source", "square.png", 3))
    assert len(backend.client.uploads) == 1
    assert len(backend.client.graphs) == 3
    seeds = {graph["30"]["inputs"]["seed"] for graph in backend.client.graphs}
    assert len(seeds) == 3
    assert all(graph["20"]["inputs"]["text"] == "placeholder" for graph in backend.client.graphs)
    assert pngs == [PNG + b"prompt-1", PNG + b"prompt-2", PNG + b"prompt-3"]


def test_without_a_seed_binding_a_variation_workflow_returns_one_image():
    backend = ComfyImageInputBackend(_edit_workflow(prompt_node_id=None, seed_node_id=None), FakeComfy())
    assert backend.max_variations == 1
    with pytest.raises(WorkflowError, match="1 to 1 variations"):
        asyncio.run(backend.vary_images(b"source", "square.png", 2))
    assert backend.client.graphs == []
    assert len(asyncio.run(backend.vary_images(b"source", "square.png", 1))) == 1


def test_a_workflow_load_refuses_bindings_the_graph_does_not_have(tmp_path: Path):
    path = _workflows(tmp_path) / "edit.json"
    with pytest.raises(WorkflowError, match="no input '10'.pixels"):
        ComfyImageInputWorkflow.load(path, image_node_id="10", image_input_name="pixels", output_node_id="40")
    with pytest.raises(WorkflowError, match="no output node '99'"):
        ComfyImageInputWorkflow.load(path, image_node_id="10", output_node_id="99")
    with pytest.raises(WorkflowError, match="no input '77'.text"):
        ComfyImageInputWorkflow.load(path, image_node_id="10", output_node_id="40", prompt_node_id="77")


def test_a_cancelled_edit_interrupts_its_prompt_and_releases_the_gpu_lock():
    async def run():
        lock = asyncio.Lock()
        fake = FakeComfy(hold=asyncio.Event())
        backend = ComfyImageInputBackend(_edit_workflow(), fake, render_lock=lock)
        task = asyncio.create_task(backend.edit_image(b"source", "cat.png", "night sky"))
        await asyncio.sleep(0.05)
        assert lock.locked()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return fake.interrupted, lock.locked()

    interrupted, locked = asyncio.run(run())
    assert interrupted == ["prompt-1"]
    assert locked is False


def test_yaml_configures_edit_and_variation_and_validates_them_offline(tmp_path: Path):
    config = _config(tmp_path, EDIT_ROUTE + VARY_ROUTE)
    settings = Settings.from_yaml(config)
    settings.validate_startup()
    assert settings.image_edit_comfy_base_url == "http://comfy.example.test:8188"
    assert settings.image_edit_workflow is not None and settings.image_variation_workflow is not None
    assert settings.image_variation_workflow.prompt_node_id is None
    edit = settings.image_edit_workflow.load()
    assert edit.for_request("in.png", prompt="p", seed=5)["30"]["inputs"]["seed"] == 5
    result = subprocess.run([sys.executable, "-m", "chord", "--show-config"],
                            env={**os.environ, "CHORD_CONFIG": str(config)},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    routes = json.loads(result.stdout)["routes"]
    assert routes["image_edit"] == {"configured": True, "origin": "http://comfy.example.test:8188"}
    assert routes["image_variation"] == {"configured": True, "origin": "http://comfy.example.test:8188",
                                         "max_n": 10}
    assert routes["image"]["configured"] is False


@pytest.mark.parametrize(("routing", "message"), [
    ("""  image_variation: {endpoint: renderer, workflow: vary.json, image_node_id: '10',
                    output_node_id: '40', prompt_node_id: '20'}
""", "routing.image_variation needs endpoint, workflow, image_node_id and output_node_id"),
    ("""  image_edit: {endpoint: renderer, workflow: edit.json, image_node_id: '10', output_node_id: '40'}
""", "routing.image_edit needs endpoint, workflow, image_node_id, prompt_node_id and output_node_id"),
    ("""  image_edit: {endpoint: chat, workflow: edit.json, image_node_id: '10', prompt_node_id: '20',
               output_node_id: '40'}
""", "routing.image_edit needs a configured comfyui endpoint"),
    ("""  image_edit: {endpoint: renderer, workflow: /abs/edit.json, image_node_id: '10', prompt_node_id: '20',
               output_node_id: '40'}
""", "routing.image_edit.workflow must be a relative path"),
    ("""  image_edit: {endpoint: renderer, workflow: edit.json, image_node_id: '', prompt_node_id: '20',
               output_node_id: '40'}
""", "routing.image_edit node and input bindings must be nonblank strings"),
])
def test_a_malformed_edit_or_variation_route_is_refused_with_its_name(tmp_path: Path, routing: str, message: str):
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_yaml(_config(tmp_path, routing))


def test_an_edit_workflow_cannot_escape_the_mounted_directory(tmp_path: Path):
    (tmp_path / "outside.json").write_text(json.dumps(EDIT_GRAPH))
    config = _config(tmp_path, EDIT_ROUTE.replace("workflow: edit.json", "workflow: ../outside.json"))
    with pytest.raises(ConfigurationError, match="image edit: image workflow escapes image_workflows_dir"):
        Settings.from_yaml(config).validate_startup()


def test_routes_on_one_comfyui_share_one_gpu_lock_and_others_do_not(tmp_path: Path):
    config = _config(tmp_path, EDIT_ROUTE + VARY_ROUTE.replace("renderer", "second"))
    config.write_text(config.read_text().replace(
        "  renderer: {type: comfyui, url: http://comfy.example.test:8188}\n",
        "  renderer: {type: comfyui, url: http://comfy.example.test:8188}\n"
        "  second: {type: comfyui, url: http://other.example.test:8188/}\n"))
    settings = Settings.from_yaml(config)
    video_graph = {"1": {"class_type": "VideoSampler", "inputs": {
        "text": "", "width": 512, "height": 512, "seconds": 4, "seed": 0}},
        "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}}}
    video_path = tmp_path / "workflows" / "video.json"
    video_path.write_text(json.dumps(video_graph))
    video_workflow = VideoWorkflowConfig(
        video_path, video_path.parent, "2",
        {"prompt": ("1", "text"), "width": ("1", "width"),
         "height": ("1", "height"), "duration": ("1", "seconds"),
         "seed": ("1", "seed")},
    )
    shared = settings.__class__(**{**settings.__dict__,
        "comfy_base_url": "http://comfy.example.test:8188/",
        "video_workflows": {"t2v": video_workflow}})
    deps = Deps(shared, upstream=FakeUpstream(), model=lambda n: None)
    assert deps.edit_backend is not None and deps.variation_backend is not None
    assert deps.video_backend is not None
    assert deps.edit_backend.render_lock is deps.video_backend._one_render
    assert deps.variation_backend.render_lock is not deps.edit_backend.render_lock


def _square_png() -> bytes:
    out = BytesIO()
    Image.new("RGB", (64, 64), "teal").save(out, format="PNG")
    return out.getvalue()


def test_without_workflows_edits_and_variations_answer_503(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, service_api_key="k", comfy_base_url="http://comfy.example.test:8188")
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None)))
    headers = {"Authorization": "Bearer k"}
    edit = client.post("/v1/images/edits", data={"prompt": "add a hat"},
                       files={"image": ("cat.png", _square_png(), "image/png")}, headers=headers)
    vary = client.post("/v1/images/variations", data={"n": "1"},
                       files={"image": ("square.png", _square_png(), "image/png")}, headers=headers)
    assert (edit.status_code, edit.json()["error"]["code"]) == (503, "backend_unavailable")
    assert (vary.status_code, vary.json()["error"]["code"]) == (503, "backend_unavailable")


def test_n_above_a_workflows_limit_is_a_400_on_n_before_any_render(tmp_path: Path):
    backend = ComfyImageInputBackend(_edit_workflow(prompt_node_id=None, seed_node_id=None), FakeComfy())
    settings = Settings(data_dir=tmp_path, service_api_key="k")
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None,
                                        variation_backend=backend)))
    response = client.post("/v1/images/variations", data={"n": "2"},
                           files={"image": ("square.png", _square_png(), "image/png")},
                           headers={"Authorization": "Bearer k"})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "n"
    assert backend.client.graphs == []
