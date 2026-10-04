import json
import pytest
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "qa"))
from conformance.schema import (
    SPEC_SHA256,
    Spec,
    _buffered_decoded_response,
    _summary,
    parse_sse,
    validate_http_response,
    validate_payload,
    validate_stream,
)
from conformance.endpoint_ledger import ledger


CHAT = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "chord-1-poly",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hello", "refusal": None},
        "finish_reason": "stop",
        "logprobs": None,
    }],
}

CHUNK = {
    "id": "chatcmpl-test",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "chord-1-poly",
    "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
}


def test_pin_is_loaded_and_verified():
    spec = Spec()
    assert spec.document["info"]["version"] == "2.3.0"
    assert len(SPEC_SHA256) == 64


def test_changed_pin_is_refused_before_validation(tmp_path):
    from conformance.schema import SPEC_PATH

    changed = tmp_path / "openapi.json"
    changed.write_bytes(SPEC_PATH.read_bytes() + b"\n")
    try:
        Spec(changed)
    except ValueError as exc:
        assert "spec hash mismatch" in str(exc)
    else:
        raise AssertionError("changed spec must not be accepted")


def test_published_chat_shape_passes():
    assert validate_payload(CHAT, kind="chat")["verdict"] == "pass"


def test_red_proof_missing_required_field_is_caught():
    planted = deepcopy(CHAT)
    del planted["object"]
    row = validate_payload(planted, kind="chat")
    assert row["verdict"] == "fail"
    assert any(error["json_path"] == "$" and "object" in error["expectation"]
               for error in row["evidence"]["errors"] if error["validator"] == "required")


def test_red_proof_wrong_nested_type_is_caught_without_payload_echo():
    planted = deepcopy(CHAT)
    planted["choices"][0]["index"] = "zero"
    row = validate_payload(planted, kind="chat")
    assert row["verdict"] == "fail"
    assert row["evidence"]["errors"][0]["json_path"] == "$.choices[0].index"
    assert "hello" not in json.dumps(row)


def test_red_proof_schema_error_never_echoes_offending_content():
    secret = "PLANTED-SECRET-MODEL-PROSE"
    planted = deepcopy(CHAT)
    planted["choices"][0]["message"]["content"] = {"text": secret}
    planted["choices"][0]["finish_reason"] = secret
    row = validate_payload(planted, kind="chat")
    encoded = json.dumps(row)
    assert row["verdict"] == "fail"
    assert secret not in encoded
    assert all("message" not in error for error in row["evidence"]["errors"])


def test_every_sse_json_chunk_is_checked_and_done_is_required():
    second = deepcopy(CHUNK)
    second["choices"][0] = {"index": 0, "delta": {}, "finish_reason": "stop"}
    raw = "data: " + json.dumps(CHUNK) + "\n\ndata: " + json.dumps(second) + "\n\ndata: [DONE]\n\n"
    rows = validate_stream(raw)
    assert [row["verdict"] for row in rows] == ["pass", "pass", "pass"]
    assert rows[0]["check"].endswith("chunk[0]") and rows[1]["check"].endswith("chunk[1]")


def test_bad_chunk_and_bad_framing_each_fail():
    planted = deepcopy(CHUNK)
    planted["created"] = "now"
    raw = "event: message\n\ndata: " + json.dumps(planted) + "\n\n"
    rows = validate_stream(raw)
    assert [row["verdict"] for row in rows] == ["fail", "fail"]


def test_sse_parser_does_not_treat_done_as_a_schema_chunk():
    chunks, done, errors = parse_sse("data: " + json.dumps(CHUNK) + "\n\ndata: [DONE]\n\n")
    assert chunks == [CHUNK] and done and not errors


def test_sse_done_must_be_unique_and_terminal():
    encoded = json.dumps(CHUNK)
    rows = validate_stream(f"data: {encoded}\n\ndata: [DONE]\n\ndata: [DONE]\n\n")
    assert rows[-1]["verdict"] == "fail"
    rows = validate_stream(f"data: [DONE]\n\ndata: {encoded}\n\n")
    assert rows[-1]["verdict"] == "fail"


def test_valid_sse_metadata_and_optional_data_space_are_accepted():
    raw = f"id: 1\nevent: chunk\nretry: 1000\ndata:{json.dumps(CHUNK)}\n\ndata:[DONE]\n\n"
    assert [row["verdict"] for row in validate_stream(raw)] == ["pass", "pass"]


def test_error_envelope_uses_published_error_schema():
    body = {"error": {"message": "bad request", "type": "invalid_request_error", "param": "messages", "code": "invalid_messages"}}
    assert validate_payload(body, kind="error")["verdict"] == "pass"


def test_embedding_response_uses_published_embedding_schema():
    body = {"object": "list", "model": "example/embeddings", "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
    ], "usage": {"prompt_tokens": 2, "total_tokens": 2}}
    assert validate_payload(body, kind="embedding")["verdict"] == "pass"


def test_valid_call_cannot_pass_by_returning_a_well_shaped_error():
    import httpx

    response = httpx.Response(
        503,
        headers={"content-type": "application/json"},
        json={"error": {"message": "busy", "type": "server_error", "param": None, "code": "busy"}},
    )
    row = validate_http_response(
        response, kind="chat", check="schema.chat", expected_status=200, spec=Spec()
    )
    assert row["verdict"] == "fail"
    assert row["evidence"]["status"] == 503


def test_non_json_response_is_one_red_row_not_an_exception():
    import httpx

    response = httpx.Response(502, headers={"content-type": "text/html"}, text="<h1>bad gateway</h1>")
    row = validate_http_response(
        response, kind="chat", check="schema.chat", expected_status=200, spec=Spec()
    )
    assert row["verdict"] == "fail"
    assert row["evidence"]["error_type"] == "non-json-response"
    assert "bad gateway" not in json.dumps(row)


def test_decoded_stream_error_drops_stale_gzip_headers_before_rebuild():
    import httpx

    wire = httpx.Response(
        502,
        headers={
            "content-type": "text/html",
            "content-encoding": "gzip",
            "content-length": "23",
        },
    )
    rebuilt = _buffered_decoded_response(wire, b"<h1>PLANTED-GZIP-ERROR</h1>")
    assert "content-encoding" not in rebuilt.headers
    assert rebuilt.headers["content-length"] == str(len(rebuilt.content))
    row = validate_http_response(
        rebuilt, kind="chat-stream", check="schema.chat-stream", expected_status=200, spec=Spec()
    )
    assert row["verdict"] == "fail"
    assert row["evidence"]["error_type"] == "non-json-response"
    assert "PLANTED-GZIP-ERROR" not in json.dumps(row)


def test_summary_preserves_shared_declared_noop_verdict():
    rows = [{"verdict": "declared-noop"}]
    assert _summary(rows)["evidence"]["declared-noop"] == 1


def test_offline_service_response_meets_required_fields(tmp_path):
    """Offline TestClient mode proves this checks our app, not fixtures alone.
    Until 2026-09-14 this asserted the gap (no choice.logprobs, no
    message.refusal); the service now sends both, so it asserts they hold."""
    from test_skeleton import make

    _, client = make(tmp_path)
    response = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}],
    })
    row = validate_payload(response.json(), kind="chat", check="schema.offline.chat")
    assert row["verdict"] == "pass", row["evidence"]


def test_offline_red_the_old_gap_still_fails_the_schema(tmp_path, monkeypatch):
    """The inverse, so the pass above can't be a validator that stopped looking:
    strip the two fields again and the offline check must go red on both."""
    from chord import server
    from test_skeleton import make

    real = server._conform_choices

    def old_gap(payload, stream):
        payload = real(payload, stream)
        for choice in payload.get("choices") or []:
            if not stream:
                choice.pop("logprobs", None)
                choice.get("message", {}).pop("refusal", None)
        return payload

    monkeypatch.setattr(server, "_conform_choices", old_gap)
    _, client = make(tmp_path)
    response = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}],
    })
    row = validate_payload(response.json(), kind="chat", check="schema.offline.chat")
    assert row["verdict"] == "fail"
    expectations = [error.get("expectation", []) for error in row["evidence"]["errors"]]
    assert any("logprobs" in required for required in expectations)
    assert any("refusal" in required for required in expectations)


def test_endpoint_ledger_is_exhaustive_and_claims_only_the_profile():
    data = ledger()
    assert data["spec_operations"] == 338
    assert data["supported_or_experimental_operations"] == 50  # +POST /embeddings (BGE-M3 via shared inference, 2026-09-21), +POST /moderations as a served no-op, +DELETE /models/{model} as a served no-op, +GET /models/{model} (S02), +POST /completions (S13b), +audio speech/transcriptions (#139), +4 Responses (#146), +8 Conversations, +5 stored Chat (S13e), +cancel, +input_tokens, +compact, +7 ?beta=true aliases, +audio translations, +5 Files (#142), +POST /images/edits, +POST /images/variations, +5 Videos
    assert data["profile_operations"] == 50
    assert "declared_operations" not in data                   # the ambiguous name is retired
    row = next(r for r in data["operations"] if r["operation"] == "DELETE /models/{model}")
    # Was "unsupported-by-design", out of the numerator (#182). On 2026-09-17 it was decided that it
    # answers `deleted: false` for any id rather than refusing, so a harness walking the Models API
    # does not break on it. It is served; it still deletes nothing, which is pinned
    # by tests/test_models_retrieve.py::test_delete_never_deletes_anything, not by this count.
    assert row["status"] == "supported" and row["reason"]   # a served no-op still has to say why
    assert data["counts"] == {"supported": 42, "experimental": 8, "unsupported-by-design": 0, "unimplemented": 288}
    assert sum(data["counts"].values()) == data["spec_operations"]
    assert data["broad_surface_coverage_percent"] == round(100*50/338, 2)


def test_an_unknown_profile_status_is_refused(monkeypatch, tmp_path):
    import json
    from conformance import endpoint_ledger
    profile = json.loads(endpoint_ledger.PROFILE_PATH.read_text())
    for status, extra in (("mostly-works", {}), ("unsupported-by-design", {})):
        bad = json.loads(json.dumps(profile))
        bad["operations"]["POST /embeddings"] = {"status": status, **extra}
        path = tmp_path / f"{status}.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setattr(endpoint_ledger, "PROFILE_PATH", path)
        with pytest.raises(ValueError):
            endpoint_ledger.ledger()


@pytest.mark.parametrize(("schema", "message"), [
    ("SchemaThatDoesNotExist", "is absent from the pinned spec"),
    ("CreateModerationResponse", "is not reachable from the operation responses"),
])
def test_profile_response_schema_must_exist_and_belong_to_the_operation(
        monkeypatch, tmp_path, schema, message):
    """A profile cannot make a compatibility claim against an unrelated shape."""
    from conformance import endpoint_ledger

    profile = json.loads(endpoint_ledger.PROFILE_PATH.read_text())
    profile["operations"]["POST /embeddings"]["response_schema"] = schema
    path = tmp_path / f"profile-{schema}.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(endpoint_ledger, "PROFILE_PATH", path)

    with pytest.raises(ValueError, match=message):
        endpoint_ledger.ledger()


@pytest.fixture
def derived_profile(monkeypatch):
    """Exercise the profile validator with a shaped, synthetic extension set.

    The fixture derives its paths and schemas from a retired deployment profile;
    it is test data, not a capture of a current service response.
    """
    from chord import manifest
    derived = json.loads((Path(__file__).parent / "fixtures" / "profile_extensions_derived.json").read_text())
    real = manifest.load()
    monkeypatch.setattr(manifest, "load", lambda: {**real, "response_extensions": derived["response_extensions"]})


# --- Undeclared fields: strict vs profile (#142) ------------------------------
from conformance.schema import undeclared_paths  # noqa: E402


def _with(chunk_or_chat, **fields):
    out = deepcopy(chunk_or_chat)
    for dotted, value in fields.items():
        node, *rest = dotted.split("__")
        target = out
        for key in [node, *rest][:-1]:
            target = target[key] if not key.isdigit() else target[int(key)]
        last = [node, *rest][-1]
        target[last] = value
    return out


def _delta(**extra):
    chunk = deepcopy(CHUNK)
    chunk["choices"][0]["delta"].update(extra)
    return chunk


def _message(**extra):
    chat = deepcopy(CHAT)
    chat["choices"][0]["message"].update(extra)
    return chat


def test_a_clean_openai_payload_passes_strict_and_profile():
    for mode in ("strict", "profile"):
        assert validate_payload(CHAT, kind="chat", fields=mode)["verdict"] == "pass"
        assert validate_payload(CHUNK, kind="chat-stream", fields=mode)["verdict"] == "pass"


@pytest.mark.usefixtures("derived_profile")
def test_our_declared_extension_fails_strict_but_passes_profile_and_is_reported():
    payload = _message(outcome="chat", trace_id="T", artifacts=[])
    strict = validate_payload(payload, kind="chat", fields="strict")
    profile = validate_payload(payload, kind="chat", fields="profile")
    assert strict["verdict"] == "fail" and "$.choices[*].message.outcome" in strict["evidence"]["undeclared_fields"]
    assert profile["verdict"] == "pass"
    assert set(profile["evidence"]["declared_extensions"]) == {
        "$.choices[*].message.outcome", "$.choices[*].message.trace_id", "$.choices[*].message.artifacts"}


def test_streamed_annotations_are_a_profile_extension_never_strict():
    notes = [{"type": "url_citation", "url_citation": {"url": "u", "title": "t", "start_index": 0, "end_index": 2}}]
    payload = _delta(annotations=notes)
    assert validate_payload(payload, kind="chat-stream", fields="strict")["verdict"] == "fail"
    assert validate_payload(payload, kind="chat-stream", fields="profile")["verdict"] == "pass"
    # non-stream message.annotations is spec-native: strict passes
    assert validate_payload(_message(annotations=notes), kind="chat", fields="strict")["verdict"] == "pass"


@pytest.mark.usefixtures("derived_profile")
@pytest.mark.parametrize("name, payload, kind, path", [
    ("top-level chunk annotations (the b82870e defect)", {**CHUNK, "annotations": []}, "chat-stream", "$.annotations"),
    ("vLLM prompt_text", {**CHUNK, "prompt_text": "x"}, "chat-stream", "$.prompt_text"),
    ("vLLM prompt_token_ids", {**CHUNK, "prompt_token_ids": [1]}, "chat-stream", "$.prompt_token_ids"),
    ("upstream reasoning in message.provider_specific_fields",
     _message(provider_specific_fields={"reasoning": "r", "refusal": None}), "chat", "$.choices[*].message.provider_specific_fields"),
    ("invented key nested in an extension", _message(job={"job_id": "j", "revision": 1, "specialist": "s", "bogus": 1}),
     "chat", "$.choices[*].message.job.bogus"),
    ("an extension at the wrong depth", {**CHUNK, "outcome": "chat"}, "chat-stream", "$.outcome"),
    ("a stream-only extension on a non-stream message", _message(provider_specific_fields={"outcome": "chat"}),
     "chat", "$.choices[*].message.provider_specific_fields"),
])
def test_red_proof_undeclared_fields_fail_even_the_profile(name, payload, kind, path):
    row = validate_payload(payload, kind=kind, fields="profile")
    assert row["verdict"] == "fail", name
    assert path in row["evidence"]["undeclared_fields"], (name, row["evidence"])
    assert "x" not in json.dumps(row["evidence"]["undeclared_fields"]).replace("$.prompt_text", "")  # paths only, no values


def test_a_schema_sanctioned_free_form_map_is_not_an_extra():
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "metadata": {"any_key": "v", "other": "w"}}
    assert undeclared_paths(request, "CreateChatCompletionRequest", Spec()) == []


def test_every_declared_extension_is_an_exact_path_with_a_reason():
    from chord import manifest
    entries = manifest.load()["response_extensions"]
    assert entries, "no extensions declared"
    for e in entries:
        assert e["path"].startswith("$.") and "*" not in e["path"].replace("[*]", ""), e
        assert e["reason"].strip() and e["modes"] and set(e["modes"]) <= {"chat", "chat-stream", "images", "error"}, e
        assert e.get("debt") in (None, "transport"), e
    assert len({(e["path"], tuple(e["modes"])) for e in entries}) == len(entries), "duplicate entries"


def test_offline_service_passes_strict(tmp_path):
    """Since 2026-09-16 the service's chat body is the pinned spec, nothing added
    (see #142): strict passes and the profile has nothing left to declare."""
    from test_skeleton import make

    _, client = make(tmp_path)
    body = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert validate_payload(body, kind="chat", fields="strict")["verdict"] == "pass"
    profile = validate_payload(body, kind="chat", fields="profile")
    assert profile["verdict"] == "pass" and profile["evidence"]["declared_extensions"] == []


@pytest.mark.usefixtures("derived_profile")
def test_field_rows_score_a_whole_stream_as_one_strict_and_one_profile_row():
    from conformance.schema import field_rows
    chunks = [_delta(outcome="chat"), {**CHUNK, "prompt_token_ids": [1]}, CHUNK]
    strict, profile = field_rows(chunks, "chat-stream", "schema.test", Spec())
    assert strict["check"] == "schema.test.strict" and profile["check"] == "schema.test.profile"
    assert set(strict["evidence"]["undeclared_fields"]) == {"$.choices[*].delta.outcome", "$.prompt_token_ids"}
    assert profile["verdict"] == "fail" and profile["evidence"]["undeclared_fields"] == ["$.prompt_token_ids"]
    assert profile["evidence"]["declared_extensions"] == ["$.choices[*].delta.outcome"]


def test_declared_properties_plus_sanctioned_extra_keys_are_not_extras():
    # The one schema in the pinned spec with both `properties` and
    # `additionalProperties: true` (EvalRunOutputItemResult): its extra keys are
    # sanctioned by the schema itself, so they must not be reported.
    item = {"name": "n", "type": "t", "score": 1, "passed": True, "grader_specific": {"any": 1}}
    assert undeclared_paths(item, "EvalRunOutputItemResult", Spec()) == []


@pytest.mark.usefixtures("derived_profile")
@pytest.mark.parametrize("name, extra, rule", [
    ("job as a string (planted case)", {"job": "wrong"}, "type"),
    ("outcome as a number (planted case)", {"outcome": 123}, "type"),
    ("artifacts as a string (planted case)", {"artifacts": "wrong"}, "type"),
    ("outcome outside its values", {"outcome": "maybe"}, "enum"),
    ("job missing a required key", {"job": {"job_id": "j", "specialist": "s"}}, "required"),
    ("an artifact missing its sha256", {"artifacts": [{"id": "a", "type": "image", "mime": "image/png"}]}, "required"),
])
def test_red_proof_a_declared_extension_with_the_wrong_shape_fails_the_profile(name, extra, rule):
    row = validate_payload(_message(**extra), kind="chat", fields="profile")
    assert row["verdict"] == "fail", name
    assert any(e["validator"] == rule for e in row["evidence"]["extension_shape_errors"]), (name, row["evidence"])
    assert "wrong" not in json.dumps(row["evidence"]["extension_shape_errors"])  # rules and paths, never the value


@pytest.mark.usefixtures("derived_profile")
def test_a_correctly_shaped_extension_passes_the_profile():
    good = {"outcome": "completed", "trace_id": "T", "job": {"job_id": "j", "revision": 1, "specialist": "search"},
            "artifacts": [{"id": "a", "type": "image", "mime": "image/png", "sha256": "0" * 64}]}
    row = validate_payload(_message(**good), kind="chat", fields="profile")
    assert row["verdict"] == "pass" and row["evidence"]["extension_shape_errors"] == [], row["evidence"]


def test_every_extension_root_declares_a_shape():
    """Roots carry a schema; children are covered by an ancestor's schema."""
    from chord import manifest
    entries = manifest.load()["response_extensions"]
    with_schema = {(e["path"], tuple(e["modes"])) for e in entries if e.get("schema")}
    for e in entries:
        if e.get("schema"):
            continue
        parent, covered = e["path"], False
        while True:
            if parent.endswith("[*]"):
                parent = parent[:-3]
            elif "." in parent[2:]:
                parent = parent.rsplit(".", 1)[0]
            else:
                break
            if (parent, tuple(e["modes"])) in with_schema:
                covered = True
                break
        assert covered, f"{e['path']} has no schema and no ancestor with one"


def _live(handler):
    import httpx
    from conformance.schema import run_live
    return run_live("http://service.test/v1", "chord-1-poly", None, True, 5, transport=httpx.MockTransport(handler))


def test_run_live_survives_a_200_that_is_not_json_and_keeps_every_row():
    """#143: a 200 HTML body used to raise JSONDecodeError on the reparse and
    throw away every earlier row. Now each arm fails on its own and the run ends."""
    import httpx
    rows = _live(lambda request: httpx.Response(200, text="<html>not json</html>", headers={"content-type": "text/html"}))
    checks = {r["check"]: r["verdict"] for r in rows}
    assert checks["schema.live.models"] == "fail" and checks["schema.live.chat"] == "fail"
    assert "schema.live.chat.error" in checks and "schema.live.images" in checks  # the run reached the last arms


@pytest.mark.usefixtures("derived_profile")
def test_run_live_scores_fields_on_models_and_error_responses_too():
    import httpx

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [], "vendor_extra": 1})
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(400, json={"error": {"message": "m", "type": "invalid_request_error", "param": None,
                                                       "code": None}, "trace_id": "T"})
        return httpx.Response(400, json={"error": {"message": "m", "type": "invalid_request_error", "param": None,
                                                   "code": "c", "surprise": 1}})
    checks = {r["check"]: r for r in _live(handler)}
    assert checks["schema.live.models.fields.strict"]["verdict"] == "fail"
    assert checks["schema.live.models.fields.profile"]["verdict"] == "fail"  # models declares no extensions
    assert "$.error.surprise" in checks["schema.live.chat.error.fields.profile"]["evidence"]["undeclared_fields"]
    # a declared extension on an error: strict fails, profile passes
    assert checks["schema.live.images.fields.strict"]["verdict"] == "fail"
    assert checks["schema.live.images.fields.profile"]["verdict"] == "pass"


def _progress_chunk(marker):
    chunk = deepcopy(CHUNK)
    chunk["choices"][0]["delta"] = {"provider_specific_fields": {"progress": marker}}
    return chunk


@pytest.mark.usefixtures("derived_profile")
@pytest.mark.parametrize("marker", [
    {"capability": "image", "stage": "submitting"},   # ongoing (graph.py:542)
    {"done": True},                                   # closing (server.py:958)
])
def test_the_two_real_progress_shapes_pass_the_profile(marker):
    row = validate_payload(_progress_chunk(marker), kind="chat-stream", fields="profile")
    assert row["verdict"] == "pass", row["evidence"]


@pytest.mark.usefixtures("derived_profile")
@pytest.mark.parametrize("marker", [
    {}, {"capability": "image"}, {"stage": "submit"}, {"done": False},
    # hybrids of the two variants (#143 re-review of 4e0d437): each open
    # branch used to accept the other variant's keys
    {"capability": "image", "stage": "submitting", "done": False},
    {"done": True, "capability": "image"},
    {"done": True, "stage": "submitting"},
])
def test_red_proof_a_malformed_progress_marker_fails_the_profile(marker):
    """#143: with every key optional, all four of these used to score green."""
    row = validate_payload(_progress_chunk(marker), kind="chat-stream", fields="profile")
    assert row["verdict"] == "fail" and row["evidence"]["extension_shape_errors"], row["evidence"]


def test_real_image_stream_passes_strict(tmp_path, monkeypatch):
    """The real image flow scored as one stream: no progress event, no progress
    marker, no image field. Strict passes; the image is markdown in content."""
    from fastapi.testclient import TestClient
    from chord import specialists
    from chord.config import Settings
    from chord.contract import Outcome, Result
    from chord.server import Deps, create_app
    from conformance.schema import field_rows
    import test_progress as tp
    from test_skeleton import AvailableImageBackend

    async def fake_image(job, ctx):
        ctx.progress("preparing")
        ctx.progress("submitting")
        d = ctx.artifacts.register(tp.PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a mug")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)
    settings = Settings(data_dir=tmp_path, router_enabled=True, enabled_routes=frozenset({"image"}))
    chunks = tp.stream_chunks(TestClient(create_app(Deps(settings, upstream=tp.FakeUpstream(),
                                                          model=lambda n: tp.FixedRouter(),
                                                          image_backend=AvailableImageBackend()))))
    strict, profile = field_rows(chunks, "chat-stream", "schema.offline.image", Spec())
    assert strict["verdict"] == "pass", strict["evidence"]
    assert "![image](" in tp.content_of(chunks)
