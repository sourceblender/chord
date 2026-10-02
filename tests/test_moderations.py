"""`POST /v1/moderations` is served, spec-shaped, and decides nothing (#121).

2026-09-18: build the route, take the values, return nothing off-spec, do
nothing inside, and document it so nobody believes it works.

The risk with a no-op is not that it fails — it is that it succeeds, quietly, and a
caller reads `flagged: false` as "we checked". So the tests below pin the shape AND
pin the emptiness, and one of them exists purely to make the emptiness loud.
"""
import json
import pathlib

import pytest

from chord import moderations
from test_skeleton import make, make_keyed

SPEC = json.loads((pathlib.Path(__file__).resolve().parents[1] /
                   "qa" / "conformance" / "spec" / "openapi.json").read_text())
RESULT = SPEC["components"]["schemas"]["CreateModerationResponse"]["properties"]["results"]["items"]


def test_the_categories_match_the_pinned_spec():
    """Written out in the module so the service never reads the spec to answer a
    request; this is what stops that copy drifting from the spec it copied."""
    assert sorted(moderations.CATEGORIES) == sorted(RESULT["properties"]["categories"]["properties"])


def test_the_response_is_the_spec_object(tmp_path):
    _, client = make(tmp_path)
    r = client.post("/v1/moderations", json={"model": "chord-1-poly", "input": "anything at all"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == set(SPEC["components"]["schemas"]["CreateModerationResponse"]["required"])
    assert body["id"].startswith("modr-")
    assert len(body["results"]) == 1
    assert set(body["results"][0]) == set(RESULT["required"])


def test_it_decides_nothing_whatever_it_is_given(tmp_path):
    """The point of the whole file. Hand it the things a real moderator exists for
    and it still says nothing was flagged — because it looked at nothing.

    If this ever fails it means someone armed the route, and the right response is
    to check that `MODERATION_DECIDES_NOTHING` came down with it and that #121's
    design actually landed, not to edit this test."""
    assert moderations.MODERATION_DECIDES_NOTHING, "the route classifies now — #121 must have landed"
    _, client = make(tmp_path)
    for text in ("hello", "I want to hurt someone", "a" * 2000, ""):
        result = client.post("/v1/moderations", json={"input": text}).json()["results"][0]
        assert result["flagged"] is False
        assert not any(result["categories"].values())
        assert set(result["category_scores"].values()) == {0.0}


def test_it_never_answers_as_a_real_moderation_model(tmp_path):
    """#121: return our real model name, never impersonate omni-moderation. A caller
    pinning a real moderation model must get a 404, not a comfortable lie."""
    _, client = make(tmp_path)
    assert client.post("/v1/moderations", json={"input": "x"}).json()["model"] == "chord-1-poly"
    for pinned in ("omni-moderation-latest", "omni-moderation-2024-09-26", "text-moderation-stable"):
        r = client.post("/v1/moderations", json={"model": pinned, "input": "x"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize("value,n", [
    ("one string", 1),
    (["a", "b", "c"], 3),
    ([{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "data:,"}}], 2),
])
def test_every_input_form_returns_one_result_each(tmp_path, value, n):
    """The spec's three shapes: a string, an array of strings, an array of parts."""
    _, client = make(tmp_path)
    assert len(client.post("/v1/moderations", json={"input": value}).json()["results"]) == n


@pytest.mark.parametrize("body,code,param", [
    ({}, "missing_required_parameter", "input"),
    ({"input": 5}, "invalid_value", "input"),
    ({"input": []}, "invalid_value", "input"),
    ({"input": ["a"] * 101}, "invalid_value", "input"),
    ({"input": "x", "temperature": 1}, "unknown_parameter", "temperature"),
])
def test_refusals_are_shaped_and_name_the_field(tmp_path, body, code, param):
    _, client = make(tmp_path)
    r = client.post("/v1/moderations", json=body)
    assert r.status_code == 400, r.text
    assert set(r.json()) == {"error"}                       # no extension keys (S15)
    assert r.json()["error"]["code"] == code and r.json()["error"]["param"] == param


def test_it_needs_the_service_key_like_everything_else(tmp_path):
    client = make_keyed(tmp_path)
    assert client.post("/v1/moderations", json={"input": "x"}).status_code == 401
    assert client.post("/v1/moderations", json={"input": "x"},
                       headers={"Authorization": "Bearer s3cret"}).status_code == 200
