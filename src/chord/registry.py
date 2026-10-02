"""Capability registry: which capabilities exist and which have earned routing.

A capability is routable only when its `certified` block names the exact model
and prompt version currently configured. Change either one and it stops being
routable until it is certified again. That's deliberate.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"
PATH = Path(os.environ.get("CHORD_REGISTRY", Path(__file__).parent / "registry.yaml"))


def prompt_version(path: Path) -> str:
    """Git blob sha of the prompt file, the same value `git hash-object` prints."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@dataclass
class Capability:
    id: str
    description: str
    model: str
    prompt: str  # filename under prompts/
    tools: list[str] = field(default_factory=list)
    certified: dict | None = None

    @property
    def prompt_path(self) -> Path:
        return PROMPTS_DIR / self.prompt

    @property
    def prompt_version(self) -> str:
        return prompt_version(self.prompt_path)

    @property
    def routable(self) -> bool:
        c = self.certified or {}
        return (
            bool(c.get("passed_at"))
            and c.get("model") == self.model
            and c.get("prompt_version") == self.prompt_version
        )


def load(path: Path | None = None) -> dict[str, Capability]:
    path = path or PATH
    raw = yaml.safe_load(path.read_text()) or {}
    return {c["id"]: Capability(**c) for c in raw.get("capabilities", [])}
