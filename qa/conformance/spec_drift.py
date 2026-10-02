#!/usr/bin/env python3
"""Spec drift report: what changed in OpenAI's spec since our pin, for the
operations we actually serve. Report only; it never touches the pin.

For each operation in endpoint-profile.json with status supported/experimental,
it walks every schema the operation reaches (request body, parameters,
responses, transitively through $ref) in the pinned spec and in the candidate
spec, and reports:

- operations added or removed upstream (whole surface, for the roadmap);
- for our operations: schemas added to or removed from their reach, schemas
  whose JSON changed, and required fields newly added (the ones that break us).

    python qa/conformance/spec_drift.py --latest [--json]
    python qa/conformance/spec_drift.py --candidate path/to/openapi.json [--json]

Exit 0 when nothing we serve drifted, 1 when something did, 2 on bad input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from .profile import PROFILE_PATH, load_profile, served_operations
except ImportError:  # direct script execution
    from profile import PROFILE_PATH, load_profile, served_operations

HERE = Path(__file__).resolve().parent
PINNED = HERE / "spec" / "openapi.json"
PROFILE = PROFILE_PATH
METHODS = ("get", "post", "put", "patch", "delete")
LATEST_URL = "https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.json"


def fetch_latest() -> dict:
    """Fetch the current generated JSON from OpenAI's official spec repo."""
    request = urllib.request.Request(LATEST_URL, headers={"User-Agent": "chord-spec-drift/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def operations(spec: dict) -> dict[str, dict]:
    out = {}
    for path, item in spec.get("paths", {}).items():
        for method in METHODS:
            if method in item:
                out[f"{method.upper()} {path}"] = item[method]
    return out


def _refs(node, found: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            found.add(ref.rsplit("/", 1)[1])
        for value in node.values():
            _refs(value, found)
    elif isinstance(node, list):
        for value in node:
            _refs(value, found)


def reach(spec: dict, op: dict) -> set[str]:
    """Every component schema an operation can touch, transitively."""
    schemas = spec.get("components", {}).get("schemas", {})
    seen: set[str] = set()
    frontier: set[str] = set()
    _refs(op, frontier)
    while frontier:
        name = frontier.pop()
        if name in seen or name not in schemas:
            continue
        seen.add(name)
        nxt: set[str] = set()
        _refs(schemas[name], nxt)
        frontier |= nxt - seen
    return seen


def _digest(node) -> str:
    return hashlib.sha256(json.dumps(node, sort_keys=True).encode()).hexdigest()


def _required(schema: dict) -> set[str]:
    req = set(schema.get("required", []))
    for part in schema.get("allOf", []):
        if isinstance(part, dict):
            req |= set(part.get("required", []))
    return req


def drift(pinned: dict, candidate: dict, served: list[str]) -> dict:
    old_ops, new_ops = operations(pinned), operations(candidate)
    old_s = pinned.get("components", {}).get("schemas", {})
    new_s = candidate.get("components", {}).get("schemas", {})
    report = {
        "operations_added_upstream": sorted(set(new_ops) - set(old_ops)),
        "operations_removed_upstream": sorted(set(old_ops) - set(new_ops)),
        "served": {},
    }
    for key in served:
        if key not in old_ops:
            report["served"][key] = {"error": "not in the pinned spec"}
            continue
        if key not in new_ops:
            report["served"][key] = {"removed_upstream": True}
            continue
        old_reach, new_reach = reach(pinned, old_ops[key]), reach(candidate, new_ops[key])
        changed = sorted(n for n in old_reach & new_reach if _digest(old_s[n]) != _digest(new_s[n]))
        entry = {
            "schemas_added": sorted(new_reach - old_reach),
            "schemas_removed": sorted(old_reach - new_reach),
            "schemas_changed": changed,
            "new_required": {n: sorted(_required(new_s[n]) - _required(old_s[n]))
                             for n in changed if _required(new_s[n]) - _required(old_s[n])},
            "operation_changed": _digest(old_ops[key]) != _digest(new_ops[key]),
        }
        if any(entry[k] for k in ("schemas_added", "schemas_removed", "schemas_changed", "new_required", "operation_changed")):
            report["served"][key] = entry
    return report


def markdown(report: dict) -> str:
    lines = ["# Spec drift for served operations", ""]
    if not report["served"]:
        lines.append("No served operation drifted.")
    for key, entry in sorted(report["served"].items()):
        lines.append(f"## {key}")
        for field in ("removed_upstream", "error", "operation_changed", "schemas_added", "schemas_removed", "schemas_changed", "new_required"):
            if entry.get(field):
                lines.append(f"- **{field}**: {entry[field]}")
        lines.append("")
    lines += ["## Whole surface", f"- operations added upstream: {len(report['operations_added_upstream'])}"]
    lines += [f"  - {op}" for op in report["operations_added_upstream"]]
    lines += [f"- operations removed upstream: {len(report['operations_removed_upstream'])}"]
    lines += [f"  - {op}" for op in report["operations_removed_upstream"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument("--latest", action="store_true",
                        help="fetch the latest JSON from OpenAI's official openai-openapi repository")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        pinned = json.loads(PINNED.read_text())
        candidate = fetch_latest() if a.latest else json.loads(a.candidate.read_text())
        profile = load_profile(PROFILE)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"spec_drift: {exc}", file=sys.stderr)
        return 2
    report = drift(pinned, candidate, served_operations(profile))
    print(json.dumps(report, indent=2) if a.json else markdown(report))
    return 1 if report["served"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
