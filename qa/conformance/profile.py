"""Authoritative loader for Chord's declared OpenAI API surface."""

from __future__ import annotations

import json
from pathlib import Path


PROFILE_PATH = Path(__file__).with_name("endpoint-profile.json")
OPERATION_STATUSES = frozenset({
    "supported",
    "experimental",
    "unsupported-by-design",
    "unimplemented",
})
SERVED_STATUSES = frozenset({"supported", "experimental"})


def load_profile(path: Path = PROFILE_PATH) -> dict:
    """Load and structurally validate the API-surface registry."""
    profile = json.loads(path.read_text())
    required = {"spec_commit", "operations", "extensions", "unlisted_openai_operations"}
    if set(profile) != required:
        raise ValueError(f"endpoint profile fields {sorted(profile)} != {sorted(required)}")
    if profile["unlisted_openai_operations"] not in OPERATION_STATUSES:
        raise ValueError("endpoint profile has an unknown unlisted-operation status")
    if not isinstance(profile["operations"], dict) or not isinstance(profile["extensions"], dict):
        raise ValueError("endpoint profile operations and extensions must be objects")
    for key, claim in profile["operations"].items():
        if not isinstance(key, str) or " " not in key or not isinstance(claim, dict):
            raise ValueError(f"invalid endpoint profile operation entry {key!r}")
        if claim.get("status") not in OPERATION_STATUSES:
            raise ValueError(f"{key}: unknown status {claim.get('status')!r}")
    for key, claim in profile["extensions"].items():
        if not isinstance(key, str) or " " not in key or not isinstance(claim, dict):
            raise ValueError(f"invalid endpoint profile extension entry {key!r}")
        if claim.get("status") != "supported-extension" or not claim.get("reason"):
            raise ValueError(f"{key}: extensions require supported-extension status and a reason")
    return profile


def served_operations(profile: dict) -> list[str]:
    """Sorted OpenAI operations the service claims to answer."""
    return sorted(
        key
        for key, claim in profile["operations"].items()
        if claim.get("status") in SERVED_STATUSES
    )
