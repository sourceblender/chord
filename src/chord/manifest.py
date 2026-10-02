"""Load the capability manifest and derive every advertisement from it."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

import os

# The deployed service reads the chosen manifest through CHORD_MANIFEST.
PATH = Path(os.environ.get("CHORD_MANIFEST", Path(__file__).parent / "manifest.yaml"))

# The model's creation date, and it is honorary on purpose: the first commit of
# this repository, 2026-09-11 12:00:39 -0400. The spec requires `created` to be a
# real Unix timestamp; we shipped 0 until a review caught it, which reads as
# 1970-01-01 to every client that renders it. It is a constant rather than the
# build time because a model's creation date does not change when we redeploy it.
MODEL_CREATED = 1789142439


@lru_cache(maxsize=1)
def load() -> dict:
    return yaml.safe_load(PATH.read_text())


def allowed_parts() -> set[str]:
    m = load()["input"]
    parts = {"text"} if m["text"] else set()
    if m["image"]:
        parts.add("image_url")
    if m["audio"]:
        parts.add("input_audio")
    return parts


def models_entry(model_id: str) -> dict:
    """The OpenAI Model object and nothing else (S15), served by both
    `GET /v1/models` and `GET /v1/models/{model}`. What the model can do is in the
    manifest, not on the public wire.

    `shutdown_date` is optional in the schema but present in the spec's examples
    for both routes, and a client that reads it cannot tell a field we omit from
    one we have no value for. We serve it explicitly as null: nothing announced."""
    return {"id": model_id, "object": "model", "created": MODEL_CREATED,
            "owned_by": load().get("owner", "chord"), "shutdown_date": None}


def capability_sentence(
    reachable: frozenset | set | None = None,
    audio_output: bool = True,
    *,
    configured_image: bool = False,
) -> str:
    """The base layer's only statement of what the assistant can do, generated from the
    manifest so it grows as capabilities are certified and never runs ahead of
    them.

    `reachable`: the specialist capabilities THIS request can actually reach
    (None: the manifest alone). Told "You can make pictures" on a request that
    couldn't, the reply said "Here's a yellow mug for you!" with nothing attached
    (live, Responses without image_generation, 2026-09-16). `audio_output`: the
    request asked for audio, the only way a voice message reaches a caller."""
    m = load()
    can = []
    if m["input"]["image"]:
        can.append("see the pictures you're sent")
    if (m["output"]["image"] or configured_image) and (reachable is None or "image" in reachable):
        can.append("make pictures")
    if m["input"]["audio"]:
        can.append("hear voice messages")
    if m["output"]["audio"] and audio_output:
        can.append("send voice messages")
    if not can:
        return ""
    listed = ", ".join(can[:-1]) + (" and " if len(can) > 1 else "") + can[-1]
    return f"You can {listed}. That's part of what you're good at."
