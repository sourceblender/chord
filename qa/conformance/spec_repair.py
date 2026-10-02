#!/usr/bin/env python3
"""Spec repair: auditable, opt-in view of the pinned OpenAI spec that
satisfies JSON Schema metaschema validation, while leaving the verbatim
upstream pin untouched.

Why this exists
---------------
The pinned spec at ``qa/conformance/spec/openapi.json`` is a byte-identical
copy of ``openai/openai-openapi@4bb21ba`` (spec 2.3.0). It contains one schema,
``ContainerResource.required``, whose 9 entries include four duplicates of
``id``, ``name``, ``created_at``, and ``status``. JSON Schema's metaschema
declares ``required`` as ``uniqueItems: true``, so ``check_schema()`` rejects
the schema outright -- not at instance validation time, but at the moment
any tool tries to load the schema document itself. Per-schema validators
that resolve by ``$ref`` (the project's normal path) are unaffected; the
moment a tool tries to load the whole document, e.g. to enumerate or
to feed a whole-document validator, it falls over.

Two rejected design choices
------------------------------------------------------
1. Sanitise on load (de-duplicate in every consumer). Means consumers
   validate against a document they did not pin. Drift invisible.
2. Reach-only resolution (resolve per-operation, never the whole doc).
   The project already does this for the validator. Correct, but it
   cannot answer "does the whole spec parse", which is the question
   a future whole-document tool would need to ask.

What we ship instead
--------------------
A separate, auditable repaired copy alongside the verbatim pin. The
verbatim file is never written by this tool; the repair script's
output is deterministic and committed. A ``--check`` mode re-applies
the repair and asserts byte-identical output, so a regression in
either the verbatim (the duplicates return) or the repaired (someone
hand-edits it) breaks the suite before it ships.

Every repair the script applies is recorded here:

  * components.schemas.ContainerResource.required: de-duplicate while
    preserving first-occurrence order. From
    ``['id','object','name','created_at','status','id','name','created_at','status']``
    to
    ``['id','object','name','created_at','status']``.

If a newer pin is vendored and the upstream has new repair-worthy
schema bugs, list them here and update ``expected_repairs()`` below;
the test will fail until both are in sync.

Usage
-----
    python qa/conformance/spec_repair.py --check
    python qa/conformance/spec_repair.py --emit path/to/repaired.json

``--check`` exits 0 when the stored repaired copy is byte-identical to
what the script would emit; 1 when it differs (drift, hand-edit, or a
new repair-worthy bug appeared in the verbatim pin). 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERBATIM = HERE / "spec" / "openapi.json"
REPAIRED = HERE / "spec" / "openapi.repaired.json"


def expected_repairs() -> list[dict]:
    """The repairs this script applies, recorded for audit and tested.

    Each entry: ``{"path": JSON-pointer-like path inside the document,
    "before": the verbatim value, "after": the post-repair value,
    "reason": human-readable justification}``.
    """
    return [
        {
            "path": "/components/schemas/ContainerResource/required",
            "before": ["id", "object", "name", "created_at", "status",
                       "id", "name", "created_at", "status"],
            "after": ["id", "object", "name", "created_at", "status"],
            "reason": (
                "verbatim contains 4 duplicates of id/name/created_at/status; "
                "JSON Schema metaschema declares required as uniqueItems: true; "
                "preserving first-occurrence order preserves the field ordering "
                "the upstream intends"
            ),
        },
    ]


def repair(document: dict) -> dict:
    """Apply every expected repair to a fresh copy of the verbatim document."""
    out = json.loads(json.dumps(document))  # deep copy via JSON round-trip
    for r in expected_repairs():
        # Walk the JSON-pointer path.
        node = out
        parts = r["path"].lstrip("/").split("/")
        for p in parts:
            if p == "":
                continue
            # Numeric segments index into arrays; string segments into dicts.
            if isinstance(node, list):
                node = node[int(p)]
            else:
                node = node[p]
        if node != r["before"]:
            raise ValueError(
                f"repair pre-condition failed at {r['path']}: "
                f"expected {r['before']!r}, found {node!r}. "
                f"A newer pin has likely changed this schema; update "
                f"expected_repairs() in {__file__}."
            )
        # Apply.
        if isinstance(node, list):
            # For arrays we mutate the slot in place so surrounding structure
            # stays identical.
            parent = out
            for p in parts[:-1]:
                if p == "":
                    continue
                if isinstance(parent, list):
                    parent = parent[int(p)]
                else:
                    parent = parent[p]
            if isinstance(parent, list):
                parent[int(parts[-1])] = list(r["after"])
            else:
                parent[parts[-1]] = list(r["after"])
        else:
            parent = out
            for p in parts[:-1]:
                if p == "":
                    continue
                if isinstance(parent, list):
                    parent = parent[int(p)]
                else:
                    parent = parent[p]
            parent[parts[-1]] = r["after"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="verify the stored repaired copy is in sync with the verbatim + repairs")
    mode.add_argument("--emit", type=Path,
                      help="write the repaired document to PATH")
    a = ap.parse_args(argv)

    try:
        verbatim = json.loads(VERBATIM.read_text())
    except (OSError, ValueError) as exc:
        print(f"spec_repair: cannot read verbatim spec at {VERBATIM}: {exc}", file=sys.stderr)
        return 2

    try:
        repaired = repair(verbatim)
    except ValueError as exc:
        print(f"spec_repair: {exc}", file=sys.stderr)
        return 1

    if a.emit is not None:
        a.emit.parent.mkdir(parents=True, exist_ok=True)
        a.emit.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n")
        return 0

    # --check mode.
    if not REPAIRED.exists():
        print(f"spec_repair: stored repaired copy missing at {REPAIRED}; "
              f"run `python {Path(__file__).name} --emit {REPAIRED}` to write it.", file=sys.stderr)
        return 1
    stored = REPAIRED.read_text()
    emitted = json.dumps(repaired, indent=2, sort_keys=True) + "\n"
    if stored != emitted:
        # Show a brief diff hint: which repair caused the mismatch?
        try:
            stored_doc = json.loads(stored)
        except ValueError:
            print("spec_repair: stored repaired copy is not valid JSON", file=sys.stderr)
            return 1
        for r in expected_repairs():
            node = stored_doc
            parts = r["path"].lstrip("/").split("/")
            for p in parts:
                if p == "":
                    continue
                if isinstance(node, list):
                    node = node[int(p)]
                else:
                    node = node[p]
            if node != r["after"]:
                print(f"spec_repair: drift at {r['path']}: "
                      f"stored {node!r}, expected {r['after']!r}", file=sys.stderr)
                return 1
        print("spec_repair: stored repaired copy differs from emitted output "
              "but no expected repair matches; the verbatim may have new "
              "drift. Re-run with --emit and inspect the diff.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
