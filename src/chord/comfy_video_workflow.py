"""Operator-owned ComfyUI video graphs and named request bindings."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VideoWorkflowError(ValueError):
    """A configured video workflow cannot honor a request."""


def parse_bindings(spec: dict[str, Any], *, reference: bool) -> tuple[dict[str, tuple[str, str]], str, str, int | None]:
    """Turn named YAML node/input bindings into the workflow's immutable contract."""
    required = ("prompt", "width", "height", "duration", "seed")
    if reference:
        required += ("reference",)
    bindings: dict[str, tuple[str, str]] = {}
    defaults = {"prompt": "text", "width": "width", "height": "height",
                "duration": "seconds", "seed": "seed", "reference": "image"}
    for name in required:
        node_id = spec.get(f"{name}_node_id")
        input_name = spec.get(f"{name}_input_name", defaults[name])
        if not isinstance(node_id, str) or not node_id.strip():
            raise VideoWorkflowError(f"video workflow needs {name}_node_id")
        if not isinstance(input_name, str) or not input_name.strip():
            raise VideoWorkflowError(f"video workflow needs a nonblank {name}_input_name")
        bindings[name] = (node_id, input_name)
    output = spec.get("output_node_id")
    if not isinstance(output, str) or not output.strip():
        raise VideoWorkflowError("video workflow needs output_node_id")
    unit = spec.get("duration_unit", "seconds")
    fps = spec.get("fps")
    if not isinstance(unit, str) or unit not in ("seconds", "frames"):
        raise VideoWorkflowError("video duration_unit must be seconds or frames")
    if unit == "frames" and (type(fps) is not int or fps <= 0):
        raise VideoWorkflowError("video frames duration needs a positive integer fps")
    if unit == "seconds" and fps is not None:
        raise VideoWorkflowError("video fps is only valid for frames duration")
    return bindings, output, unit, fps


@dataclass(frozen=True)
class VideoWorkflow:
    graph: dict[str, dict[str, Any]]
    output_node_id: str
    bindings: dict[str, tuple[str, str]]
    duration_unit: str = "seconds"
    fps: int | None = None

    @classmethod
    def load(
        cls, path: Path, *, output_node_id: str,
        bindings: dict[str, tuple[str, str]],
        duration_unit: str = "seconds", fps: int | None = None,
    ) -> VideoWorkflow:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise VideoWorkflowError(f"cannot load video workflow {path}: {type(exc).__name__}") from None
        if not isinstance(raw, dict) or not raw or any(
            not isinstance(node_id, str) or not isinstance(node, dict)
            or not isinstance(node.get("class_type"), str)
            or not isinstance(node.get("inputs"), dict)
            for node_id, node in raw.items()
        ):
            raise VideoWorkflowError("video workflow must be a nonempty API-format node map")
        if output_node_id not in raw:
            raise VideoWorkflowError(f"video workflow has no output node {output_node_id!r}")
        required = {"prompt", "width", "height", "duration", "seed"}
        if not required <= bindings.keys() or set(bindings) - required - {"reference"}:
            raise VideoWorkflowError("video workflow needs prompt, width, height, duration and seed bindings")
        for name, (node_id, input_name) in bindings.items():
            if node_id not in raw or input_name not in raw[node_id]["inputs"]:
                raise VideoWorkflowError(f"video workflow has no {name} input {node_id!r}.{input_name}")
        if not isinstance(duration_unit, str) or duration_unit not in ("seconds", "frames"):
            raise VideoWorkflowError("video duration_unit must be seconds or frames")
        if duration_unit == "frames" and (type(fps) is not int or fps <= 0):
            raise VideoWorkflowError("video frames duration needs a positive integer fps")
        if duration_unit == "seconds" and fps is not None:
            raise VideoWorkflowError("video fps is only valid for frames duration")
        return cls(raw, output_node_id, bindings, duration_unit, fps)

    def for_request(
        self, *, prompt: str, width: int, height: int, seconds: int,
        seed: int, reference_name: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not prompt.strip() or width <= 0 or height <= 0 or seconds <= 0:
            raise VideoWorkflowError("video request needs prompt, dimensions and duration")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise VideoWorkflowError("video seed must be a nonnegative 64-bit integer")
        if ("reference" in self.bindings) != (reference_name is not None):
            raise VideoWorkflowError("video reference does not match the configured workflow")
        duration = seconds
        if self.duration_unit == "frames":
            if self.fps is None:
                raise VideoWorkflowError("video frames duration needs fps")
            duration = seconds * self.fps
        values: dict[str, Any] = {
            "prompt": prompt, "width": width, "height": height,
            "duration": duration,
            "seed": seed,
        }
        if reference_name is not None:
            values["reference"] = reference_name
        graph = copy.deepcopy(self.graph)
        for name, value in values.items():
            node_id, input_name = self.bindings[name]
            graph[node_id]["inputs"][input_name] = value
        return graph
