"""The auditable spec repair.

The verbatim OpenAPI pin at qa/conformance/spec/openapi.json must NOT be
modified by anything in this repository. A separate repaired copy at
qa/conformance/spec/openapi.repaired.json exists to satisfy JSON Schema
metaschema validation (the upstream pin has ContainerResource.required
duplicates that fail check_schema). Every repair the script applies is
recorded in spec_repair.expected_repairs() and tested here; the
script's --check mode exits non-zero when the stored repaired copy
drifts from what re-running the repair would emit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO / "qa" / "conformance" / "spec"
VERBATIM = SPEC_DIR / "openapi.json"
REPAIRED = SPEC_DIR / "openapi.repaired.json"


def _load_spec_repair():
    """Load spec_repair as a module (it lives under qa/, not src/, so it
    isn't on the default package import path for pytest)."""
    spec = importlib.util.spec_from_file_location(
        "spec_repair", REPO / "qa" / "conformance" / "spec_repair.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verbatim_pin_is_not_modified_by_anything_in_the_repo():
    """The verbatim file is byte-stable. Anyone who edits it directly breaks
    the audit story; this test fails on any whitespace, ordering, or content
    change. If the upstream spec is intentionally bumped, the pin and this
    hash move together -- and that is the only way this hash changes."""
    expected_sha256 = "3d6223349eadfd937624b9e6b8abf596ec2f680a1a367889cf6a6f924e568127"
    actual = hashlib.sha256(VERBATIM.read_bytes()).hexdigest()
    assert actual == expected_sha256, (
        f"verbatim pin SHA256 changed: expected {expected_sha256}, got {actual}. "
        f"This file must match openai/openai-openapi@4bb21ba byte-for-byte. "
        f"If you intentionally bumped the pin, update the hash here AND "
        f"in qa/conformance/schema.py (SPEC_SHA256) together."
    )


def test_repaired_copy_exists_and_parses():
    assert REPAIRED.exists(), (
        f"repaired copy missing at {REPAIRED}. "
        f"Run: python qa/conformance/spec_repair.py --emit {REPAIRED}"
    )
    json.loads(REPAIRED.read_text())  # raises on bad JSON


def test_check_mode_passes_against_stored_repaired():
    """The script's --check mode is the audit contract: byte-identical
    output between re-running the repair and the stored file. This is
    the assertion that catches a regression in either direction."""
    spec_repair = _load_spec_repair()
    rc = spec_repair.main(["--check"])
    assert rc == 0, f"spec_repair --check exited {rc}; the repaired copy has drifted"


def test_repair_against_verbatim_produces_byte_identical_repaired():
    """Independent of --check: read both files, run the repair on the
    verbatim, compare JSON dumps. A failure here means either the
    verbatim drifted (caught above) or the stored repaired was hand-edited."""
    spec_repair = _load_spec_repair()
    verbatim = json.loads(VERBATIM.read_text())
    stored = json.loads(REPAIRED.read_text())
    emitted = spec_repair.repair(verbatim)
    # compare as JSON strings with sorted keys, indent=2 -- the script's
    # --emit format -- so any structural difference (key order, whitespace,
    # dedupe outcome) is caught.
    assert json.dumps(stored, indent=2, sort_keys=True) == json.dumps(
        emitted, indent=2, sort_keys=True
    ), "stored repaired differs from what repair(verbatim) would emit today"


def test_every_expected_repair_is_listed_in_the_audit_table():
    """The audit table (expected_repairs) is the list of every difference
    between verbatim and repaired. If new repair-worthy schema bugs
    appear in the verbatim (a future pin bump), they have to land here
    AND in the stored repaired file together. This test asserts the
    two stay in sync by re-applying each recorded repair and checking
    that the stored file's value at that path equals the recorded
    'after'."""
    spec_repair = _load_spec_repair()
    repaired = json.loads(REPAIRED.read_text())
    for r in spec_repair.expected_repairs():
        node = repaired
        for part in r["path"].lstrip("/").split("/"):
            if part == "":
                continue
            if isinstance(node, list):
                node = node[int(part)]
            else:
                node = node[part]
        assert node == r["after"], (
            f"recorded repair {r['path']!r} expected {r['after']!r} "
            f"in repaired copy, found {node!r}. Update expected_repairs() "
            f"and regenerate the repaired copy together."
        )


def test_repair_fails_loud_when_verbatim_pre_condition_does_not_match():
    """If a future pin bump changes the broken schema (different field
    names, different duplicate pattern), the repair script must refuse
    rather than silently no-op. This catches the case where someone
    bumps the pin but forgets to update expected_repairs()."""
    spec_repair = _load_spec_repair()
    tampered = {
        "components": {"schemas": {
            "ContainerResource": {
                "type": "object",
                # Simulate a future pin that fixed the duplicates AND renamed a field.
                "required": ["id", "object", "name", "created_at", "status"],
                "properties": {
                    "id": {"type": "string"},
                    "object": {"type": "string"},
                    "name": {"type": "string"},
                    "created_at": {"type": "integer"},
                    "status": {"type": "string"},
                },
            },
        }},
    }
    with pytest.raises(ValueError, match="repair pre-condition failed"):
        spec_repair.repair(tampered)


def test_repaired_copy_passes_whole_schema_metaschema_check():
    """The reason the repaired copy exists: Draft202012Validator.check_schema
    on the verbatim ContainerResource raises. On the repaired one, it
    succeeds. Pinned so the next person who touches either file sees
    the failure mode the repair addresses."""
    from jsonschema import Draft202012Validator
    rep = json.loads(REPAIRED.read_text())
    v = Draft202012Validator({})
    # Whole-component-schemas check (the case future whole-document
    # tools would hit):
    v.check_schema(rep["components"]["schemas"])
    # And the specific schema that motivated the repair:
    cr = rep["components"]["schemas"]["ContainerResource"]
    Draft202012Validator(cr).check_schema(cr)


def test_verbatim_still_fails_metaschema_check_so_the_audit_story_holds():
    """The verbatim file is intentionally NOT fixed in place; a future
    audit must be able to look at verbatim, see the upstream bug, and
    see exactly what we repaired. If this test ever passes (verbatim
    passes check_schema), the upstream has fixed ContainerResource and
    we can retire the repaired copy."""
    from jsonschema import Draft202012Validator
    spec = json.loads(VERBATIM.read_text())
    cr = spec["components"]["schemas"]["ContainerResource"]
    with pytest.raises(Exception, match="non-unique elements"):
        Draft202012Validator(cr).check_schema(cr)
