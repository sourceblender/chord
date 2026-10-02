"""S09 (red team pass 1, 2026-09-15): upstream refusals went to the client as
the gateway's own body: `litellm.*` class names, the internal model group and
its fallbacks, `type: null`, `code: "400"`. The fixtures are DERIVED from the
bodies the red team captured (S-cf-071, S-cf-072, S-cf-086): the deployment's
model-group name is replaced with `example/chat` (11 occurrences) and every other
byte is unchanged, so the shape under test is the captured one. The verbatim
captures are tests/fixtures/red_team_upstream_errors.json at commit 1e9a8dc
(unchanged through c89af1e), sha256
349c7bd5bb4a28557bbe249ca88a1c844e6d300685433582c76315aada0bd0b0, kept in the
private repository's history, not in a public export."""
import json
from pathlib import Path

import openai
import pytest

from chord.server import _upstream_error_body
from chord.upstream import UpstreamError
from test_progress import last_trace
from test_skeleton import FakeUpstream, make

CAPTURED = json.loads((Path(__file__).parent / "fixtures/red_team_upstream_errors.json").read_text())
LEAKS = ("litellm", "model_group", "model group", "fallback", "private/", "OpenAIException", "Received Model Group")


# What each captured case sent: the provenance a rebuilt param must come from.
SENT = {"S-cf-071": {"temperature": 3.0}, "S-cf-072": {"max_completion_tokens": 0}, "S-cf-086": {}}


def envelope(case, sent=None):
    c = CAPTURED[case]
    return _upstream_error_body(UpstreamError(c["status"], c["body"]), SENT[case] if sent is None else sent)["error"]


def no_leaks(err):
    text = json.dumps(err)
    assert not [w for w in LEAKS if w.lower() in text.lower()], text


def test_s_cf_071_temperature_names_the_param_and_keeps_the_backend_sentence():
    err = envelope("S-cf-071")
    no_leaks(err)
    assert err == {"message": "temperature must be between 0 and 2; got 3.0.", "type": "invalid_request_error",
                   "param": "temperature", "code": "invalid_value"}


def test_s_cf_072_names_the_field_the_caller_sent():
    """Live on 2748410: the caller sent max_completion_tokens=0,
    LiteLLM renamed it, and our envelope said max_tokens."""
    err = envelope("S-cf-072")
    no_leaks(err)
    assert err == {"message": "max_completion_tokens must be at least 1; got 0.", "type": "invalid_request_error",
                   "param": "max_completion_tokens", "code": "invalid_value"}
    assert envelope("S-cf-072", {"max_tokens": 0})["param"] == "max_tokens"


@pytest.mark.parametrize("case,sent", [("S-cf-072", {}), ("S-cf-072", {"max_output_tokens": 0}), ("S-cf-071", {})])
def test_a_param_the_request_did_not_carry_is_never_named(case, sent):
    """The upstream's spelling alone is not provenance."""
    assert envelope(case, sent) == {"message": GENERIC_400, "type": "invalid_request_error",
                                    "param": None, "code": "invalid_request"}


def test_s_cf_086_context_overflow_is_context_length_exceeded():
    err = envelope("S-cf-086")
    no_leaks(err)
    assert err["type"] == "invalid_request_error" and err["code"] == "context_length_exceeded"
    assert err["param"] == "messages"
    assert err["message"] == "The input is at least 131073 tokens, over the model's context limit of 131072 tokens."


@pytest.mark.parametrize("status,body,type_,code", [
    (429, "rate limited", "rate_limit_error", "rate_limit_exceeded"),
    (500, "<html>boom</html>", "server_error", "upstream_error"),
    (503, json.dumps({"error": {"message": "litellm.ServiceUnavailableError: model_group=example/chat down"}}),
     "server_error", "upstream_error"),
    (400, json.dumps({"error": {"message": "litellm.BadRequestError: something about example/chat"}}),
     "invalid_request_error", "invalid_request"),
    (400, "not json at all", "invalid_request_error", "invalid_request"),
])
def test_other_shapes_get_a_plain_envelope_and_never_leak(status, body, type_, code):
    err = _upstream_error_body(UpstreamError(status, body), {})["error"]
    no_leaks(err)
    assert err["type"] == type_ and err["code"] == code
    assert set(err) == {"message", "type", "param", "code"} and err["message"]


class Refusing(FakeUpstream):
    def __init__(self, case):
        super().__init__()
        self.case = CAPTURED[case]

    async def complete(self, body):
        raise UpstreamError(self.case["status"], self.case["body"])

    async def stream(self, body):
        raise UpstreamError(self.case["status"], self.case["body"])
        yield  # pragma: no cover


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("stream", [False, True])
def test_backend_auth_refusal_is_our_server_failure(tmp_path, status, stream):
    class AuthRefusing(FakeUpstream):
        async def complete(self, body):
            raise UpstreamError(status, '{"error":{"message":"backend credential rejected"}}')

        async def stream(self, body):
            raise UpstreamError(status, '{"error":{"message":"backend credential rejected"}}')
            yield  # pragma: no cover

    _, client = make(tmp_path, AuthRefusing())
    response = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "The model backend rejected Chord's credentials.",
        "type": "server_error", "param": None, "code": "upstream_auth_error",
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", list(CAPTURED))
def test_endpoint_returns_the_clean_envelope_and_keeps_the_raw_body_in_the_trace(tmp_path, stream, case):
    deps, client = make(tmp_path, Refusing(case))
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
                                                  "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == CAPTURED[case]["status"]
    no_leaks(r.json()["error"])
    assert last_trace(deps.settings)["upstream_error_body"] == CAPTURED[case]["body"][:8000]


GENERIC_400 = "The model backend rejected the request."


@pytest.mark.parametrize("text", [
    # the repros on 59db595: none matched the old blocklist, all leaked.
    "request failed at backend-host.internal:4000",
    "provider rejected api_key=sk-secret-example",
    "malformed input: private user text",
    # A new wording we have never seen, dressed like the known ones.
    "SomeNewGatewayError: tenant=acme route=agent/secret-lane",
    # A recognised shape carrying a parameter outside our allowlist.
    "internal_route must be at least 1, got 0.",
    # A recognised shape whose number is not a number.
    "temperature must be in [0, 2], got sk-secret.",
])
@pytest.mark.parametrize("wrap", ["litellm.BadRequestError: OpenAIException - {} No fallback model group found", "{}"])
def test_unrecognised_upstream_text_is_never_published(text, wrap):
    msg = wrap.format(text)
    err = _upstream_error_body(UpstreamError(400, json.dumps({"error": {"message": msg}})),
                               {"temperature": 1, "internal_route": 0})["error"]
    assert err == {"message": GENERIC_400, "type": "invalid_request_error", "param": None, "code": "invalid_request"}


def test_the_public_sentence_is_ours_even_for_recognised_shapes():
    """Only allowlisted names and plain numbers survive from upstream text."""
    msg = "OpenAIException - top_p must be in [0, 1], got 1.5. (parameter=top_p, value=1.5) secret-host:9 trailing"
    err = _upstream_error_body(UpstreamError(400, json.dumps({"error": {"message": msg}})), {"top_p": 1.5})["error"]
    assert err["message"] == "top_p must be between 0 and 1; got 1.5." and err["param"] == "top_p"


@pytest.mark.parametrize("stream", [False, True])
def test_the_renamed_refusal_names_the_callers_field_through_the_endpoint(tmp_path, stream):
    """The mapping holds end to end, whatever bound the upstream enforces."""
    deps, client = make(tmp_path, Refusing("S-cf-072"))
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
                                                  "max_completion_tokens": 5,
                                                  "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "max_completion_tokens"


# Both public spellings are checked before anything is forwarded, so the name
# in the refusal is the one in the request, not the gateway's rename.
BAD_TOKEN_VALUES = [0, -1, 1.5, True, "5"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", BAD_TOKEN_VALUES)
@pytest.mark.parametrize("param", ["max_tokens", "max_completion_tokens"])
def test_a_bad_token_limit_is_refused_under_the_name_sent(tmp_path, param, value, stream):
    up = FakeUpstream()
    deps, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream, param: value,
                                                  "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"] == {"message": f"{param} must be an integer of at least 1",
                                 "type": "invalid_request_error", "param": param, "code": "invalid_value"}
    assert up.bodies == []                               # never reached the gateway to be renamed


@pytest.mark.parametrize("param", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", [None, 1, 4096])
def test_a_good_token_limit_passes(tmp_path, param, value):
    up = FakeUpstream()
    deps, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", param: value,
                                                  "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and len(up.bodies) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("param", ["max_tokens", "max_completion_tokens"])
def test_official_sdk_sees_the_name_it_sent(tmp_path, param, stream):
    """S-cf-072 through openai-python itself."""
    deps, client = make(tmp_path, FakeUpstream())
    sdk = openai.OpenAI(api_key="test", base_url="http://testserver/v1", http_client=client, max_retries=0)
    with pytest.raises(openai.BadRequestError) as caught:
        sdk.chat.completions.create(model="chord-1-poly", messages=[{"role": "user", "content": "hi"}],
                                    stream=stream, **{param: 0})
    assert caught.value.body["param"] == param and param in caught.value.body["message"]


@pytest.mark.asyncio
async def test_a_non_json_200_is_a_502_upstream_error_not_an_unhandled_crash():
    """A 200 carrying HTML (a proxy in front of a dead backend) used to raise
    JSONDecodeError out of resp.json() -- an unhandled 500 internal_error
    implying a chord bug, while the sibling doors answered 502 for the
    identical condition. decode_embedding_tokens already guarded its parse;
    this extends the guard to every call (review 2026-09-22)."""
    import httpx

    from chord.upstream import Upstream

    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>not json</html>"))
    up = Upstream("http://backend.invalid/v1", "k")
    await up._client.aclose()
    up._client = httpx.AsyncClient(base_url="http://backend.invalid/v1", transport=transport)
    with pytest.raises(UpstreamError) as excinfo:
        await up.complete({"model": "m"})
    assert excinfo.value.status == 502
    assert "not JSON" in str(excinfo.value)
