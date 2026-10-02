from __future__ import annotations

import json

import pytest

from chord.comfy_video_workflow import VideoWorkflow, VideoWorkflowError, parse_bindings


def _workflow(tmp_path, *, reference: bool = False, unit: str = "seconds", fps: int | None = None):
    graph = {
        "inputs": {"class_type": "VideoSampler", "inputs": {
            "text": "sample", "width": 1, "height": 1, "duration": 1, "seed": 0,
            **({"reference": "sample.png"} if reference else {}),
        }},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["inputs", 0]}},
    }
    path = tmp_path / "operator-video.json"
    path.write_text(json.dumps(graph))
    bindings = {name: ("inputs", field) for name, field in {
        "prompt": "text", "width": "width", "height": "height",
        "duration": "duration", "seed": "seed",
        **({"reference": "reference"} if reference else {}),
    }.items()}
    return VideoWorkflow.load(path, output_node_id="save", bindings=bindings,
                              duration_unit=unit, fps=fps)


def test_operator_graph_receives_only_bound_inputs_and_is_not_mutated(tmp_path):
    workflow = _workflow(tmp_path)
    graph = workflow.for_request(prompt="lighthouse", width=1280, height=720, seconds=8, seed=17)
    assert graph["inputs"]["inputs"] == {
        "text": "lighthouse", "width": 1280, "height": 720, "duration": 8, "seed": 17,
    }
    assert workflow.graph["inputs"]["inputs"]["text"] == "sample"
    assert workflow.for_request(prompt="trees", width=720, height=1280, seconds=4, seed=18)["inputs"]["inputs"]["text"] == "trees"


def test_reference_and_frame_duration_are_explicit(tmp_path):
    workflow = _workflow(tmp_path, reference=True, unit="frames", fps=24)
    graph = workflow.for_request(prompt="motion", width=720, height=1280, seconds=8,
                                 seed=1, reference_name="uploaded.png")
    assert graph["inputs"]["inputs"]["duration"] == 192
    assert graph["inputs"]["inputs"]["reference"] == "uploaded.png"
    with pytest.raises(VideoWorkflowError, match="reference"):
        workflow.for_request(prompt="motion", width=720, height=1280, seconds=8, seed=1)


def test_invalid_graph_or_binding_refuses_at_load(tmp_path):
    workflow = _workflow(tmp_path)
    path = tmp_path / "operator-video.json"
    with pytest.raises(VideoWorkflowError, match="no output node"):
        VideoWorkflow.load(path, output_node_id="wrong", bindings=workflow.bindings)
    with pytest.raises(VideoWorkflowError, match="no seed input"):
        VideoWorkflow.load(path, output_node_id="save",
                           bindings={**workflow.bindings, "seed": ("wrong", "seed")})
    with pytest.raises(VideoWorkflowError, match="positive integer fps"):
        VideoWorkflow.load(path, output_node_id="save", bindings=workflow.bindings,
                           duration_unit="frames", fps=0)


def test_yaml_binding_parser_requires_only_named_inputs():
    spec = {f"{name}_node_id": "input" for name in
            ("prompt", "width", "height", "duration", "seed", "reference")}
    spec.update(output_node_id="save", duration_unit="frames", fps=24)
    bindings, output, unit, fps = parse_bindings(spec, reference=True)
    assert bindings["reference"] == ("input", "image")
    assert (output, unit, fps) == ("save", "frames", 24)
    with pytest.raises(VideoWorkflowError, match="reference_node_id"):
        parse_bindings({key: value for key, value in spec.items() if key != "reference_node_id"},
                       reference=True)
