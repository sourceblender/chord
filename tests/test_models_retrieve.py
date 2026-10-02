"""S02 (red team pass 1, 2026-09-15): GET /v1/models/{model} was an Invalid URL
for the service's own advertised id, raw or %2F-encoded, so the official SDK's
models.retrieve() failed."""
import openai
import pytest

from test_skeleton import PNG, make, make_keyed
from test_responses import strict


def test_every_listed_id_can_be_retrieved_raw_and_encoded(tmp_path):
    _, client = make(tmp_path)
    listed = client.get("/v1/models").json()["data"]
    assert listed
    for entry in listed:
        for path in (entry["id"], entry["id"].replace("/", "%2F")):
            r = client.get(f"/v1/models/{path}")
            assert r.status_code == 200, (path, r.text)
            assert r.json() == entry          # same object the list returned


def test_unknown_model_is_an_openai_shaped_404(tmp_path):
    _, client = make(tmp_path)
    r = client.get("/v1/models/chord-1-open")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found" and err["param"] == "model" and err["type"] == "invalid_request_error"


def test_retrieve_requires_auth_like_the_list(tmp_path):
    client = make_keyed(tmp_path)
    assert client.get("/v1/models/chord-1-poly").status_code == 401
    assert client.get("/v1/models/chord-1-poly", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_official_sdk_models_retrieve(tmp_path):
    """The red team's failing call, through openai-python itself."""
    _, client = make(tmp_path)
    sdk = openai.OpenAI(api_key="test", base_url="http://testserver/v1", http_client=client)
    m = sdk.models.retrieve("chord-1-poly")
    assert m.id == "chord-1-poly" and m.object == "model"


# S13 child, S-models-004 (red team pass 1, 2026-09-15): DELETE on a model id
# was "Invalid URL". deleteModel is for fine-tunes; we have none.
@pytest.mark.parametrize("path", [
    "/v1/models/ft-nonexistent-000", "/v1/models/ft%3Agpt-4o%3Aorg%3Ax", "/v1/models/nobody/else",
    "/v1/models/chord-1-poly", "/v1/models/chord-1-open",
])
def test_delete_answers_the_spec_object_with_deleted_false(tmp_path, path):
    """The 2026-09-17 decision: deny with `false` rather than error, so a harness
    walking the Models API does not break on a call we will never honour. Any id,
    served or not, gets the same DeleteModelResponse."""
    _, client = make(tmp_path)
    r = client.delete(path)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"id", "object", "deleted"}          # the spec's fields, nothing added (S15)
    assert body["deleted"] is False and body["object"] == "model"


def test_delete_needs_the_service_key_like_get(tmp_path):
    client = make_keyed(tmp_path)
    assert client.delete("/v1/models/ft-x").status_code == 401
    assert client.delete("/v1/models/ft-x", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_official_sdk_models_delete_returns_a_not_deleted_object(tmp_path):
    """Through openai-python, the call a harness actually makes: it parses, and it
    reports not-deleted rather than raising (replaces the two error-path SDK tests
    that pinned the 404 and the 400, #182)."""
    _, client = make(tmp_path)
    sdk = openai.OpenAI(api_key="test", base_url="http://testserver/v1", http_client=client, max_retries=0)
    for model in ("ft-nonexistent-000", "chord-1-poly"):
        result = sdk.models.delete(model)
        assert result.deleted is False and result.id == model and result.object == "model"


def test_delete_never_deletes_anything(tmp_path):
    """The S13f lane ruling for S-models-004 rests on this: a DELETE here cannot mutate
    anything, so re-running the frozen case against prod is read-only BY EFFECT even
    though the lane policy reads its verb as destructive.

    This pin is on the EFFECT. Until 2026-09-17 it read the handler's status codes
    instead ("every return is an `error(...)`"), which made a presentation change look
    like a ruling change — the handler now answers 200, and deletes exactly as much as
    it did before: nothing. The day `deleted` can come back true, this fails, the ruling
    is void, and the case goes back to the destructive lane with a resource receipt."""
    _, client = make(tmp_path)
    before = client.get("/v1/models").json()["data"]
    for model in ("chord-1-poly", "ft:nope", "anything/at/all", "x"):
        assert client.delete(f"/v1/models/{model}").json()["deleted"] is False
    # Delete every model we serve, twice, then look: the catalogue is untouched.
    for entry in before:
        client.delete(f"/v1/models/{entry['id']}")
        client.delete(f"/v1/models/{entry['id']}")
    assert client.get("/v1/models").json()["data"] == before
    assert client.get("/v1/models/chord-1-poly").status_code == 200


def test_every_models_object_is_strict_valid_against_the_pinned_spec(tmp_path):
    """The strict lane could not see these routes at all until 2026-09-17: `Model` and
    `DeleteModelResponse` had no kind in conformance.SCHEMAS, so a regression on either
    was invisible to it. A review caught the delete half while signing S13f and checked the
    schema by hand; the retrieve half was the same hole one route over.

    All three Models objects now go through the same validator everything else does."""
    _, client = make(tmp_path)
    listing = client.get("/v1/models").json()
    strict(listing, "models")
    for entry in listing["data"]:
        strict(client.get(f"/v1/models/{entry['id']}").json(), "model")
    strict(client.delete("/v1/models/ft-nonexistent-000").json(), "model-deleted")
    strict(client.delete("/v1/models/chord-1-poly").json(), "model-deleted")


def test_no_route_carries_its_own_copy_of_the_model_id(tmp_path, monkeypatch):
    """Every route that takes a model must resolve it through `graph.persona_for`, so
    the id this edition serves lives in exactly one place.

    Not a style preference. Renaming to `chord-1-poly` changed the constant and the
    Responses **compaction** route kept answering for the old id, because it carried the
    prefix as an inlined string literal. The first version of this test hand-listed four
    doors, none of which was the door that drifted, and it passed with the defect planted
    back in. So the doors are DERIVED from the app here, and every POST route must be
    either a door or explicitly excused — a new route cannot quietly skip the check.

    The method: re-point `MODEL_ID`, then knock on every door with the retired id. A door
    that still opens is holding its own copy of the name."""
    from chord import graph as graph_mod
    _, client = make(tmp_path)
    retired = graph_mod.MODEL_ID
    wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00"

    # Routes that take a model, with a body good enough to reach the model check.
    doors = {
        "/v1/chat/completions":        dict(json={"model": retired, "messages": [{"role": "user", "content": "hi"}]}),
        "/v1/completions":             dict(json={"model": retired, "prompt": "hi", "max_tokens": 4}),
        "/v1/images/generations":      dict(json={"model": retired, "prompt": "a cube"}),
        "/v1/responses":               dict(json={"model": retired, "input": "hi"}),
        "/v1/responses/compact":       dict(json={"model": retired}),
        "/v1/responses/input_tokens":  dict(json={"model": retired, "input": "hi"}),
        "/v1/audio/speech":            dict(json={"model": retired, "input": "hi", "voice": "alloy"}),
        "/v1/audio/transcriptions":    dict(data={"model": retired}, files={"file": ("c.mp3", wav, "audio/mpeg")}),
        "/v1/audio/translations":      dict(data={"model": retired}, files={"file": ("c.mp3", wav, "audio/mpeg")}),
        "/v1/moderations":             dict(json={"model": retired, "input": "x"}),
        "/v1/embeddings":              dict(json={"model": retired, "input": "x"}),
        "/v1/images/edits":            dict(data={"model": retired, "prompt": "a cube"},
                                            files={"image": ("a.png", PNG, "image/png")}),
        "/v1/images/variations":       dict(data={"model": retired},
                                            files={"image": ("a.png", PNG, "image/png")}),
    }
    # POST routes that legitimately take no model, each with the reason it is excused.
    no_model = {
        "/v1/responses/{response_id}/cancel": "acts on a stored response by id",
        "/v1/conversations": "creates a conversation; the model is on the turn, not the container",
        "/v1/conversations/{conversation_id}": "updates metadata",
        "/v1/conversations/{conversation_id}/items": "adds items to an existing conversation",
        "/v1/chat/completions/{completion_id}": "updates a stored completion's metadata",
        "/v1/files": "uploads bytes",
        "/v1/videos": "video models are sora-2 and sora-2-pro, not the chat model id",
    }

    posts = {r.path for r in client.app.routes
             if "POST" in getattr(r, "methods", set()) and r.path.startswith("/v1/")}
    unclassified = posts - set(doors) - set(no_model)
    assert not unclassified, (
        f"new POST route(s) {sorted(unclassified)} are neither a model door nor excused — "
        "classify them here so the id cannot drift into a route nobody knocks on")

    monkeypatch.setattr(graph_mod, "MODEL_ID", "chord-1-open")
    for path, kwargs in doors.items():
        r = client.post(path, **kwargs)
        assert r.status_code == 404, f"{path} still serves the retired id {retired!r}: {r.status_code} {r.text[:200]}"
        assert r.json()["error"]["code"] == "model_not_found", (path, r.text)

    # And the listing advertises whatever the constant says, never a remembered string.
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["chord-1-open"]
