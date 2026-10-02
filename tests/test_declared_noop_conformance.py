"""Falsifiers for #288, reviewed independently of the implementation.

The shape fix in #291 left two directions untested. These cover those cases,
plus the guard that would have caught the original
defect — because correcting the lambda closes today's bug and leaves the mechanism
that admitted it.

What admitted it: `test_every_declared_noop_has_a_validator` asserts key-set equality
between `declared_noop_params` and `NOOP_VALIDATORS`. That is a PRESENCE check. It
says a validator exists, never that it matches the pinned spec, so a lambda accepting
an invented vocabulary satisfies it perfectly — and one did, for a safety-shaped
field, past a green suite.

2026-09-18.
"""
import json
import pathlib

import pytest

from chord import manifest
from chord.server import NOOP_VALIDATORS

from test_skeleton import make

SPEC = pathlib.Path(__file__).resolve().parents[1] / "qa" / "conformance" / "spec" / "openapi.json"
BASE = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("stream", [False, True])
def test_declared_noops_never_reach_the_model_backend(tmp_path, stream):
    deps, client = make(tmp_path)
    values = {
        "user": "end-user-opaque-id",
        "safety_identifier": "safety-opaque-id",
        "prompt_cache_key": "private-cache-key",
        "prompt_cache_options": {"ttl": "30m"},
        "prompt_cache_retention": "24h",
    }
    assert set(values) == set(manifest.load()["declared_noop_params"])
    response = client.post("/v1/chat/completions", json={**BASE, **values, "stream": stream})
    assert response.status_code == 200, response.text
    assert not set(values) & set(deps.upstream.bodies[-1])


def _schemas():
    return json.loads(SPEC.read_text())["components"]["schemas"]


def _chat_request_properties():
    """Resolve CreateChatCompletionRequest's properties, following $ref through allOf.

    Merging only the top-level `allOf` parts finds 2 of the 8 declared no-ops and
    reads exactly like "this cannot be checked against the spec". Six of them live
    behind a $ref inside that allOf. Following refs, all 8 resolve — which is what
    makes this whole file possible, so it gets asserted rather than assumed.
    """
    sc = _schemas()

    def props(node, seen=()):
        out = {}
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            if name in seen:
                return out
            return props(sc[name], seen + (name,))
        for part in node.get("allOf", []):
            out.update(props(part, seen))
        out.update(node.get("properties", {}))
        return out

    return props(sc["CreateChatCompletionRequest"])


def _standalone_schema(param):
    """One chat param's schema, with ONLY the definitions it actually reaches.

    Injecting all of `components.schemas` as `$defs` fails before it validates
    anything: the pinned spec carries `ContainerResource.required` with nine entries
    of which four are duplicated, which is invalid JSON Schema (`required` is
    `uniqueItems`), so jsonschema rejects the metaschema. That duplication is byte-
    identical in the frozen 2026-09-15 copy and in `qa/conformance/spec`, so it comes
    from upstream and is not ours to edit — but it does mean nothing may validate
    against the whole pinned document at once. Reaching only what the param needs
    keeps this test away from unrelated upstream damage (2026-09-18).
    """
    sc = _schemas()
    root = _chat_request_properties()[param]
    need, seen = [root], set()

    def refs(node):
        if isinstance(node, dict):
            r = node.get("$ref")
            if isinstance(r, str) and r.startswith("#/components/schemas/"):
                yield r.split("/")[-1]
            for v in node.values():
                yield from refs(v)
        elif isinstance(node, list):
            for v in node:
                yield from refs(v)

    while need:
        for name in refs(need.pop()):
            if name not in seen:
                seen.add(name)
                need.append(sc[name])
    schema = dict(root)
    schema["$defs"] = {n: sc[n] for n in seen}
    return json.loads(json.dumps(schema).replace("#/components/schemas/", "#/$defs/"))


def test_every_declared_noop_is_a_real_chat_request_field_in_the_pinned_spec():
    """The weakest possible conformance claim, and the one nobody was making.

    A param we accept and quietly drop must at least be a field the spec declares on
    this request. Without this, `declared_noop_params` can name anything at all.
    """
    props = _chat_request_properties()
    missing = [p for p in manifest.load()["declared_noop_params"] if p not in props]
    assert not missing, f"declared no-ops absent from the pinned CreateChatCompletionRequest: {missing}"


def test_the_resolver_reaches_every_declared_noop():
    """Guards the helper above, not the service. If $ref-following regresses, every
    test in this file passes vacuously on an empty property set — the failure mode
    where a conformance suite agrees with everything."""
    props = _chat_request_properties()
    assert len(props) > 30, f"resolver returned {len(props)} properties; it is not resolving refs"
    for name in ("moderation", "prediction", "metadata", "prompt_cache_retention"):
        assert name in props, f"{name} unreachable; the resolver is broken, not the manifest"


def test_no_param_is_both_refused_and_a_declared_noop():
    """`moderation` was in `declared_noop_params` while the argument about whether to
    refuse it was still live. A param in both lists has two answers, and which one a
    caller gets depends on statement order in _validate."""
    m = manifest.load()
    both = set(m["refused_params"]) & set(m["declared_noop_params"])
    assert not both, f"params claimed as both refused and declared no-ops: {sorted(both)}"


def test_a_refused_param_has_no_validator_and_a_noop_param_has_one():
    """The two lists own disjoint halves of NOOP_VALIDATORS. A refused param keeping
    its validator is the exact residue the #288 fix had to remove: dead code shaped
    like a capability claim."""
    m = manifest.load()
    assert set(m["declared_noop_params"]) == set(NOOP_VALIDATORS), (
        "every declared no-op needs a validator and nothing else may have one")
    stale = set(m["refused_params"]) & set(NOOP_VALIDATORS)
    assert not stale, f"refused params still carrying a validator: {sorted(stale)}"


# --- direction 1: every shape the spec declares is refused, not silently accepted ---

SPEC_VALID_MODERATION = [
    pytest.param({"model": "omni-moderation-latest"}, id="spec-example-model"),
    pytest.param({"model": "chord-1-poly"}, id="our-own-model"),
    pytest.param({"model": "omni-moderation-latest", "policy": None}, id="explicit-null-policy"),
    pytest.param({"model": "m", "policy": {"input": {"mode": "score"}}}, id="policy-input-score"),
    pytest.param({"model": "m", "policy": {"input": {"mode": "block"}}}, id="policy-input-BLOCK"),
    pytest.param({"model": "m", "policy": {"output": {"mode": "block"}}}, id="policy-output-BLOCK"),
    pytest.param({"model": "m", "policy": {"input": {"mode": "block"},
                                           "output": {"mode": "block"}}}, id="policy-both-BLOCK"),
]


@pytest.mark.parametrize("value", SPEC_VALID_MODERATION)
def test_every_spec_valid_moderation_is_refused_with_the_frozen_literal(tmp_path, value):
    """The decision on 2026-09-18 was to refuse it honestly.

    `policy.output.mode` is here because the test covers `input` only, and the
    output half is the one that would moderate what WE generate — the side a caller
    has no other way to check.

    The assertion is the frozen pass-1 literal for S-cf-033, verbatim: 400,
    `unsupported_parameter`, param `moderation`. Not "a 400" — the earlier defect
    also produced a 400 (`invalid_value`) and matching on status alone would have
    called that fixed.
    """
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={**BASE, "moderation": value})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["code"] == "unsupported_parameter", err
    assert err["param"] == "moderation", err


# --- direction 2: the invented vocabulary is gone, both values ---

@pytest.mark.parametrize("value", ["auto", "none"])
def test_the_invented_string_vocabulary_no_longer_returns_200(tmp_path, value):
    """The original defect, and the half that never surfaced as an error anyone saw.

    `{"auto", "none"}` appears nowhere in the Chat spec — not as ModerationParam, and
    not as the Images `moderation` enum, which is `low|auto`. It was a third set. A
    client sending "auto" got a 200 and believed it had configured moderation.

    `"none"` is here because the test pins `"auto"` only, and a fix that special-
    cased one string would pass that and fail this.
    """
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={**BASE, "moderation": value})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "unsupported_parameter", r.text


@pytest.mark.parametrize("value", [True, 1, [], ["block"], {"policy": {"input": {"mode": "block"}}}])
def test_malformed_moderation_is_refused_too(tmp_path, value):
    """Refusal must not depend on the value parsing as anything in particular. The
    last case is a policy with no `model`, which the spec forbids (`required: [model]`)
    and which still asks us to block."""
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={**BASE, "moderation": value})
    assert r.status_code == 400, r.text


# --- direction 3: null is the absence of a request, and must still work ---

def test_an_explicit_null_moderation_is_not_a_request_for_moderation(tmp_path):
    """A review found this by firing variants rather than by reading, and nothing in the
    certification path would have caught it: the frozen S-cf-033 bar only covers the
    object form. The spec declares `anyOf[ModerationParam, null]`, and SDKs serialise
    unset optionals as null, so refusing it 400s a client for asking us to do nothing.
    """
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={**BASE, "moderation": None})
    assert r.status_code == 200, r.text


def test_the_spec_still_declares_moderation_nullable():
    """The reason null is skipped. If the pinned spec is ever refreshed and drops the
    null arm, the skip above becomes an unjustified hole and this says so."""
    props = _chat_request_properties()
    arms = props["moderation"].get("anyOf") or []
    assert any(a.get("type") == "null" for a in arms), (
        "the pinned spec no longer declares moderation nullable; the null skip in "
        "_validate has lost its justification")
    assert any(a.get("$ref", "").endswith("ModerationParam") for a in arms), props["moderation"]


# --- the guard that would actually have caught #288 -------------------------------
#
# Everything above either reproduces the defect (the behavioural cases, red on the
# unfixed tree) or prevents a DIFFERENT one. None of the structural tests above would
# have caught #288: `moderation` is a real spec field and was always in exactly one
# list. What was wrong was the VALUE VOCABULARY, and nothing compared a validator to
# the schema it claims to enforce.
#
# So this does. Each declared no-op names one value the pinned spec accepts and one it
# rejects; both claims are checked against the spec with jsonschema before the
# validator is asked about them, so the table cannot drift into fiction. Then the
# validator must agree with the spec in both directions.

SPEC_CASES = {
    "user":                   ("u1", 5),
    "safety_identifier":      ("x" * 64, 5),
    "prompt_cache_key":       ("k", 3),
    "prompt_cache_options":   ({"ttl": "30m"}, {"ttl": "forever"}),
    "prompt_cache_retention": ("24h", "forever"),
    "prediction":             ({"type": "content", "content": "Mars"}, {"type": "nope"}),
    # Not a declared no-op since 2026-09-18 (the decision was to refuse), kept because it is the
    # regression: the deleted lambda accepted "auto" and rejected this object, and this
    # row is what turns that back into a red test if the field is ever re-admitted.
    "moderation":             ({"model": "omni-moderation-latest"}, "auto"),
}


def test_the_spec_case_table_covers_every_declared_noop():
    """Completeness, so a new no-op cannot be added with an unchecked validator — which
    is precisely how the last one got in."""
    missing = [p for p in manifest.load()["declared_noop_params"] if p not in SPEC_CASES]
    assert not missing, f"declared no-ops with no spec case: {missing}"


@pytest.mark.parametrize("param", sorted(SPEC_CASES))
def test_the_spec_case_table_is_true_about_the_spec(param):
    """The table is only worth something if the spec agrees it is right. Without this,
    a wrong 'valid' example would quietly re-derive the wrong validator."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = _standalone_schema(param)
    good, bad = SPEC_CASES[param]
    jsonschema.validate(good, schema)          # raises if the table lies about `good`
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


@pytest.mark.parametrize("param", sorted(SPEC_CASES))
def test_each_validator_agrees_with_the_pinned_spec(param):
    """The recurrence guard. `test_every_declared_noop_has_a_validator` asserts a
    validator EXISTS; this asserts it enforces the right thing.

    Red on the unfixed tree for `moderation`: the deleted lambda rejected
    {"model": "omni-moderation-latest"} — the spec's own example — and accepted the
    string "auto", which the Chat spec declares nowhere.
    """
    validator = NOOP_VALIDATORS.get(param)
    if validator is None:
        pytest.skip(f"{param} is not a declared no-op in this tree")
    good, bad = SPEC_CASES[param]
    assert validator(good), f"{param} validator rejects a value the pinned spec accepts: {good!r}"
    assert not validator(bad), f"{param} validator accepts a value the pinned spec rejects: {bad!r}"
