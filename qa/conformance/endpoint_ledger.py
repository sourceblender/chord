#!/usr/bin/env python3
"""Derive the exhaustive endpoint ledger from the pinned spec and profile."""

from __future__ import annotations

import argparse
import json

try:
    from .profile import OPERATION_STATUSES, PROFILE_PATH, load_profile
    from .schema import SPEC_COMMIT, Spec
except ImportError:  # direct script execution
    from profile import OPERATION_STATUSES, PROFILE_PATH, load_profile
    from schema import SPEC_COMMIT, Spec


METHODS = {"get", "post", "put", "patch", "delete"}
# `unsupported-by-design`: a route we serve that refuses the operation on purpose
# (DELETE /models/{model}: no owned fine-tunes). It is in the profile for two-way
# route completeness but never in the declared numerator or coverage (#182).
STATUSES = ("supported", "experimental", "unsupported-by-design", "unimplemented")
assert set(STATUSES) == OPERATION_STATUSES


def _refs(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            found.add(ref.rsplit("/", 1)[1])
        for value in node.values():
            found |= _refs(value)
    elif isinstance(node, list):
        for value in node:
            found |= _refs(value)
    return found


def _response_reach(operation: dict, schemas: dict) -> set[str]:
    """Schemas reachable from the operation's responses, transitively."""
    seen: set[str] = set()
    frontier = _refs(operation.get("responses", {}))
    while frontier:
        name = frontier.pop()
        if name in seen or name not in schemas:
            continue
        seen.add(name)
        frontier |= _refs(schemas[name]) - seen
    return seen


def _validate_claim_schemas(key: str, claim: dict, operation: dict, schemas: dict) -> None:
    reachable = _response_reach(operation, schemas)
    for field in ("response_schema", "stream_schema"):
        name = claim.get(field)
        if name is None:
            continue
        if name not in schemas:
            raise ValueError(f"{key}: {field} {name!r} is absent from the pinned spec")
        if name not in reachable:
            raise ValueError(f"{key}: {field} {name!r} is not reachable from the operation responses")


def ledger() -> dict:
    spec = Spec().document
    profile = load_profile(PROFILE_PATH)
    if profile["spec_commit"] != SPEC_COMMIT:
        raise ValueError("endpoint profile and OpenAI spec pins differ")
    claims = profile["operations"]
    schemas = spec.get("components", {}).get("schemas", {})
    rows = []
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            if method not in METHODS:
                continue
            key = f"{method.upper()} {path}"
            claim = claims.get(key)
            if claim:
                _validate_claim_schemas(key, claim, operation, schemas)
            rows.append({
                "operation": key,
                "operation_id": operation.get("operationId"),
                "status": claim["status"] if claim else profile["unlisted_openai_operations"],
                **({"response_schema": claim.get("response_schema"), "stream_schema": claim.get("stream_schema")} if claim else {}),
                **({"reason": claim["reason"]} if claim and "reason" in claim else {}),
            })
    missing = sorted(set(claims) - {row["operation"] for row in rows})
    if missing:
        raise ValueError(f"profile claims operations absent from pinned spec: {missing}")
    unknown = sorted({row["status"] for row in rows} - set(STATUSES))
    if unknown:
        raise ValueError(f"profile uses unknown status(es) {unknown}; known: {list(STATUSES)}")
    for key, claim in claims.items():
        if claim["status"] == "unsupported-by-design" and not claim.get("reason"):
            raise ValueError(f"{key}: unsupported-by-design needs a reason")
    counts = {name: sum(row["status"] == name for row in rows) for name in STATUSES}
    # Supported or experimental, not "declared": an unsupported-by-design row is
    # declared in the profile too, and must not read as coverage (#182).
    served = counts["supported"] + counts["experimental"]
    return {
        "spec_commit": SPEC_COMMIT,
        "spec_operations": len(rows),
        "profile_operations": len(claims),
        "supported_or_experimental_operations": served,
        "broad_surface_coverage_percent": round(100 * served / len(rows), 2),
        "counts": counts,
        "extensions": profile["extensions"],
        "operations": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    data = ledger()
    if args.summary:
        data = {key: data[key] for key in ("spec_commit", "spec_operations", "profile_operations",
                                           "supported_or_experimental_operations",
                                           "broad_surface_coverage_percent", "counts", "extensions")}
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
