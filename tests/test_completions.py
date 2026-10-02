"""S13b (red team pass 1, 2026-09-15; S-cmpl-001): POST /v1/completions was
"Invalid URL". Legacy Completions now serves the same public model id, with the
prompt reaching the model as written: no chat template, no base layer, no
router. The response is rebuilt from the pinned CreateCompletionResponse."""
import json
import sys
from pathlib import Path

import httpx
import openai
import pytest

from test_progress import last_trace
from test_skeleton import FakeUpstream, make, make_keyed

sys.path.insert(0, str(Path(__file__).parents[1] / "qa"))
from conformance.schema import validate_payload

FROZEN = {"model": "chord-1-poly", "prompt": "The capital of France is", "max_tokens": 8}
RAW = {  # what LiteLLM/vLLM answers with, extras and all
    "id": "cmpl-upstream", "object": "text_completion", "created": 1, "model": "example/chat",
    "choices": [{"index": 0, "text": " Paris.", "finish_reason": "length", "logprobs": None,
                 "stop_reason": None, "token_ids": None, "prompt_logprobs": None, "routed_experts": None}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13, "completion_tokens_details": None},
    "system_fingerprint": "vllm-0.28.0", "kv_transfer_params": None, "metrics": {"x": 1},
}


class TextUpstream(FakeUpstream):
    """Her model on the legacy endpoint."""

    def __init__(self, data=None, fail=None):
        super().__init__()
        self.data, self.fail, self.text_bodies = data or RAW, fail, []

    async def complete_text(self, body):
        self.text_bodies.append(body)
        if self.fail:
            raise self.fail
        return json.loads(json.dumps(self.data)), {"model-api-base": "http://10.9.8.7:8113/v1"}

    async def stream_text(self, body):
        self.text_bodies.append(body)
        if self.fail:
            raise self.fail
        yield None, {"model-api-base": "http://10.9.8.7:8113/v1"}
        for piece in (" Paris", "."):
            yield {"id": "x", "object": "text_completion", "created": 1, "model": "example/chat",
                   "choices": [{"index": 0, "text": piece, "finish_reason": None, "token_ids": None}]}, {}
        # LiteLLM's real frames (live 2026-09-16): the finish chunk has no text key, and
        # the usage chunk's pseudo-choice carries only an index.
        yield {"id": "x", "object": "text_completion", "created": 1, "model": "example/chat",
               "choices": [{"finish_reason": "length", "index": 0}]}, {}
        yield {"id": "x", "object": "text_completion", "created": 1, "model": "example/chat", "choices": [{"index": 0}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}}, {}


def test_the_frozen_red_team_case_comes_back_as_a_text_completion(tmp_path):
    up = TextUpstream()
    deps, client = make(tmp_path, up)
    r = client.post("/v1/completions", json=FROZEN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion" and body["model"] == "chord-1-poly"
    assert body["choices"] == [{"index": 0, "text": " Paris.", "logprobs": None, "finish_reason": "length"}]
    assert body["usage"]["total_tokens"] == 13
    assert body["system_fingerprint"].startswith("fp_") and "vllm" not in body["system_fingerprint"]   # opaque, S10
    assert set(body) == {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
    assert body["id"].startswith("cmpl-") and r.headers["x-chord-trace-id"] in body["id"]
    sent = up.text_bodies[-1]
    assert sent["prompt"] == FROZEN["prompt"] and sent["max_tokens"] == 8      # the prompt as written
    assert sent["model"] == deps.settings.persona_model                        # the backend model, not the public id
    assert "messages" not in sent and "stream" not in sent                     # no chat template, no base layer
    t = last_trace(deps.settings)
    assert t["endpoint"] == "completions" and t["result_status"] == "completed"


def test_the_response_matches_the_pinned_schema(tmp_path):
    from conformance.schema import Spec
    _, client = make(tmp_path, TextUpstream())
    r = client.post("/v1/completions", json=FROZEN)
    verdict = validate_payload(r.json(), kind="completion", check="schema.completion", fields="strict", spec=Spec())
    assert verdict["verdict"] == "pass", verdict
    assert "trace_id" not in r.json() and "outcome" not in r.json()            # no extension keys (S15)


def test_the_final_streamed_chunk_matches_the_pinned_schema(tmp_path):
    """The pinned CreateCompletionResponse says streamed and non-streamed share
    one shape, with finish_reason required and non-null. A stream can't: every
    chunk before the last carries null, as OpenAI's own legacy stream does. So
    the last chunk is validated strictly, and the earlier ones are asserted by
    shape. The gap is the spec's, and it keeps this operation honest-partial."""
    from conformance.schema import Spec
    _, client = make(tmp_path, TextUpstream())
    with client.stream("POST", "/v1/completions", json={**FROZEN, "stream": True}) as r:
        chunks = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {")]
    verdict = validate_payload(chunks[-1], kind="completion", check="schema.completion.final", fields="strict", spec=Spec())
    assert verdict["verdict"] == "pass", verdict
    for chunk in chunks[:-1]:
        choice = chunk["choices"][0]
        assert set(chunk) == {"id", "object", "created", "model", "choices"}
        assert set(choice) == {"index", "text", "logprobs", "finish_reason"} and choice["finish_reason"] is None


@pytest.mark.parametrize("prompt", ["The capital of France is", ["one", "two"]])
def test_string_and_string_array_prompts_are_served(tmp_path, prompt):
    up = TextUpstream()
    _, client = make(tmp_path, up)
    assert client.post("/v1/completions", json={**FROZEN, "prompt": prompt}).status_code == 200
    assert up.text_bodies[-1]["prompt"] == prompt


@pytest.mark.parametrize("extra,param", [
    ({"prompt": []}, "prompt"),
    ({"prompt": [1, "a"]}, "prompt"), ({"prompt": [[1], []]}, "prompt"), ({"prompt": [True, 2]}, "prompt"), ({"prompt": [-1]}, "prompt"),
    ({"prompt": 42}, "prompt"),
    ({"best_of": 2}, "best_of"),
    ({"n": 0}, "n"), ({"n": 129}, "n"), ({"n": True}, "n"),
    ({"n": 3, "best_of": 2}, "best_of"),                     # best_of below n
    ({"n": 2, "temperature": 0}, "n"),                       # vLLM: n must be 1 with greedy sampling (measured)
    ({"echo": "yes"}, "echo"),
    ({"suffix": "..."}, "suffix"),                           # the tested backend refuses it itself
    ({"logprobs": 6}, "logprobs"), ({"logprobs": True}, "logprobs"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"max_tokens": True}, "max_tokens"),
    ({"stream": "yes"}, "stream"),
    # the pinned types and ranges, bool excluded from every number
    ({"temperature": True}, "temperature"), ({"temperature": 3}, "temperature"), ({"temperature": "one"}, "temperature"),
    ({"top_p": "one"}, "top_p"), ({"top_p": 1.5}, "top_p"),
    ({"frequency_penalty": []}, "frequency_penalty"), ({"frequency_penalty": 3}, "frequency_penalty"),
    ({"presence_penalty": 3}, "presence_penalty"), ({"presence_penalty": True}, "presence_penalty"),
    ({"seed": True}, "seed"), ({"seed": "42"}, "seed"),
    ({"stop": {"a": 1}}, "stop"), ({"stop": ["a", "b", "c", "d", "e"]}, "stop"), ({"stop": [1]}, "stop"),
    ({"stream_options": "yes"}, "stream_options"), ({"stream_options": {"include_usage": True}}, "stream_options"),
    ({"stream_options": {"include_usage": False}}, "stream_options"),   # needs stream: true
    ({"stream": True, "stream_options": {"bogus": True}}, "stream_options"),
    ({"user": {"id": 1}}, "user"),
    ({"logit_bias": []}, "logit_bias"), ({"logit_bias": {"5": True}}, "logit_bias"),
    ({"messages": []}, "messages"),                         # a chat param on the legacy door
    ({"tools": []}, "tools"),
])
def test_what_we_cannot_honour_is_refused_not_ignored(tmp_path, extra, param):
    up = TextUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/completions", json={**FROZEN, **extra})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == param
    assert up.text_bodies == []


@pytest.mark.parametrize("model,status", [("chord-1-open", 404), ("example/chat", 404), (None, 404)])
def test_only_a_served_model_answers(tmp_path, model, status):
    _, client = make(tmp_path, TextUpstream())
    body = {**FROZEN}
    if model is None:
        del body["model"]
    else:
        body["model"] = model
    r = client.post("/v1/completions", json=body)
    assert r.status_code == status and r.json()["error"]["code"] == "model_not_found"


def test_streaming_rebuilds_every_chunk_and_ends_with_done(tmp_path):
    deps, client = make(tmp_path, TextUpstream())
    with client.stream("POST", "/v1/completions", json={**FROZEN, "stream": True}) as r:
        assert r.status_code == 200
        frames = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(f) for f in frames[:-1]]
    assert "".join(c["choices"][0]["text"] for c in chunks) == " Paris."
    assert all(c["object"] == "text_completion" and c["model"] == "chord-1-poly" for c in chunks)
    assert all("token_ids" not in c["choices"][0] for c in chunks)             # upstream extras never cross
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert last_trace(deps.settings)["endpoint"] == "completions"


def test_an_upstream_refusal_is_our_envelope(tmp_path):
    from chord.upstream import UpstreamError
    _, client = make(tmp_path, TextUpstream(fail=UpstreamError(400, json.dumps(
        {"error": {"message": "litellm.BadRequestError: OpenAIException - temperature must be in [0, 2], got 3.0."}}))))
    r = client.post("/v1/completions", json={**FROZEN, "temperature": 1.5})   # valid here; the model refuses it
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["message"] == "temperature must be between 0 and 2; got 3.0." and "litellm" not in json.dumps(err)


@pytest.mark.parametrize("stream", [False, True])
def test_backend_auth_refusal_is_not_a_client_auth_failure(tmp_path, stream):
    from chord.upstream import UpstreamError

    _, client = make(tmp_path, TextUpstream(fail=UpstreamError(401, "backend credential rejected")))
    response = client.post("/v1/completions", json={**FROZEN, "stream": stream})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == "upstream_auth_error"


def test_the_door_requires_the_service_key(tmp_path):
    client = make_keyed(tmp_path)
    assert client.post("/v1/completions", json=FROZEN).status_code == 401


def test_official_sdk_completions_create(tmp_path):
    """The red team's failing call, through openai-python itself."""
    _, client = make(tmp_path, TextUpstream())
    sdk = openai.OpenAI(api_key="test", base_url="http://testserver/v1", http_client=client, max_retries=0)
    out = sdk.completions.create(model="chord-1-poly", prompt="The capital of France is", max_tokens=8)
    assert out.object == "text_completion" and out.choices[0].text == " Paris."
    assert out.model == "chord-1-poly"


@pytest.mark.parametrize("extra", [
    {"temperature": 0}, {"temperature": 2}, {"top_p": 0.1}, {"frequency_penalty": -2}, {"presence_penalty": 2},
    {"seed": 42}, {"stop": "\n"}, {"stop": ["a", "b"]},
    {"logit_bias": {"5": -1}}, {"best_of": 1}, {"n": 1}, {"echo": False},
    {"prompt": [1, 2, 3]}, {"prompt": [[1, 2], [3]]}, {"prompt": ["a", "b"]},
    {"n": 2}, {"n": 2, "best_of": 2}, {"echo": True}, {"logprobs": 0}, {"logprobs": 5},
])
def test_values_the_pinned_request_allows_are_forwarded(tmp_path, extra):
    """The control: valid values reach the model, so the checks refuse shapes,
    not the parameters themselves."""
    up = TextUpstream()
    _, client = make(tmp_path, up)
    assert client.post("/v1/completions", json={**FROZEN, **extra}).status_code == 200, extra
    sent = up.text_bodies[-1]
    for key, value in extra.items():
        if key == "best_of":                          # only ever equal to n: nothing to forward
            assert key not in sent
        else:
            assert sent[key] == value


@pytest.mark.parametrize("stream", [False, True])
def test_completion_user_is_validated_but_never_sent_to_backend(tmp_path, stream):
    up = TextUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/completions", json={**FROZEN, "user": "private-client-id", "stream": stream})
    assert r.status_code == 200, r.text
    assert "user" not in up.text_bodies[-1]


@pytest.mark.parametrize("planted,gone", [
    ({"usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13, "private_counter": 7}}, "private_counter"),
    ({"usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13, "internal_flag": True}}, "internal_flag"),
    ({"usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13,
                "completion_tokens_details": {"reasoning_tokens": 1, "secret": 2}}}, "secret"),
    ({"choices": [{"index": 0, "text": " Paris.", "finish_reason": "length", "token_ids": [1, 2]}]}, "token_ids"),
])
def test_a_field_the_schema_does_not_name_is_dropped(tmp_path, planted, gone):
    """Closed-default: the rebuild copies named fields only (the proof)."""
    data = {**json.loads(json.dumps(RAW)), **planted}
    _, client = make(tmp_path, TextUpstream(data=data))
    r = client.post("/v1/completions", json=FROZEN)
    assert r.status_code == 200 and gone not in r.text
    assert set(r.json()["choices"][0]) == {"index", "text", "logprobs", "finish_reason"}


@pytest.mark.parametrize("planted", [
    {"usage": {"prompt_tokens": 5, "completion_tokens": True, "total_tokens": 13}},   # bool is not a count
    {"usage": {"prompt_tokens": 5, "completion_tokens": 8}},                          # a required count is missing
    {"usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": "13"}},
    {"usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13,
               "completion_tokens_details": {"reasoning_tokens": True}}},
    {"usage": "none"},
    {"choices": [{"index": True, "text": " Paris.", "finish_reason": "length"}]},
    {"choices": [{"index": 0, "text": 42, "finish_reason": "length"}]},
    {"choices": [{"index": 0, "text": 0, "finish_reason": "length"}]},        # falsy, and not a string
    {"choices": [{"index": 0, "text": False, "finish_reason": "length"}]},
    {"choices": [{"index": 0, "text": None, "finish_reason": "length"}]},
    {"choices": [{"index": 0, "finish_reason": "length"}]},                   # missing entirely
    {"choices": [{"index": 0, "text": " Paris.", "finish_reason": "banana"}]},
    {"choices": [{"index": 0, "text": " Paris.", "finish_reason": "length", "logprobs": "no"}]},
    {"choices": "nope"},
    {"choices": ["nope"]},
])
def test_an_unusable_upstream_answer_is_never_published(tmp_path, planted):
    """A shape the pinned schema can't describe fails closed, rather than
    publishing an invalid 200 or a raw 500."""
    data = {**json.loads(json.dumps(RAW)), **planted}
    deps, client = make(tmp_path, TextUpstream(data=data))
    r = client.post("/v1/completions", json=FROZEN)
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
    assert set(r.json()) == {"error"} and "banana" not in r.text
    assert last_trace(deps.settings)["upstream_shape_error"]


def test_an_unusable_streamed_chunk_stops_the_stream_without_publishing_it(tmp_path):
    class BadStream(TextUpstream):
        async def stream_text(self, body):
            self.text_bodies.append(body)
            yield None, {}
            yield {"choices": [{"index": 0, "text": " Paris", "finish_reason": None}]}, {}
            yield {"choices": [{"index": 0, "text": " ok", "finish_reason": "banana", "private": 1}]}, {}

    deps, client = make(tmp_path, BadStream())
    with client.stream("POST", "/v1/completions", json={**FROZEN, "stream": True}) as r:
        frames = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert "[DONE]" not in frames and json.loads(frames[-1])["error"]["code"] == "stream_failed"
    assert "private" not in " ".join(frames) and "banana" not in " ".join(frames)
    assert [json.loads(f)["choices"][0]["text"] for f in frames[:-1]] == [" Paris"]
    t = last_trace(deps.settings)
    assert t["result_status"] == "failed" and t["upstream_shape_error"]


# Regression from 8eec9e0: the same closed-default boundary, one level deeper.
def test_logprobs_is_rebuilt_from_its_pinned_fields(tmp_path):
    data = json.loads(json.dumps(RAW))
    data["choices"][0]["logprobs"] = {
        "tokens": [" Paris"], "token_logprobs": [-0.5], "text_offset": [0],
        "top_logprobs": [{" Paris": -0.5, " Rome": -2.0}], "private_trace": "secret",
    }
    _, client = make(tmp_path, TextUpstream(data=data))
    r = client.post("/v1/completions", json=FROZEN)
    assert r.status_code == 200 and "private_trace" not in r.text and "secret" not in r.text
    logprobs = r.json()["choices"][0]["logprobs"]
    assert set(logprobs) == {"tokens", "token_logprobs", "text_offset", "top_logprobs"}
    assert logprobs["top_logprobs"] == [{" Paris": -0.5, " Rome": -2.0}]


def test_the_echoed_first_token_keeps_its_null_logprob(tmp_path):
    """Live 2026-09-16: with echo the first prompt token has no logprob and vLLM sends
    null, as OpenAI's legacy API does. It passes as null, never an invented number."""
    data = json.loads(json.dumps(RAW))
    data["choices"][0]["logprobs"] = {"tokens": ["The", " capital"], "token_logprobs": [None, -11.6], "text_offset": [0, 3],
                                      "top_logprobs": [None, {" capital": -11.6, "\n": -2.3}]}
    _, client = make(tmp_path, TextUpstream(data=data))
    r = client.post("/v1/completions", json={**FROZEN, "echo": True, "logprobs": 1})
    assert r.status_code == 200, r.text
    logprobs = r.json()["choices"][0]["logprobs"]
    assert logprobs["token_logprobs"] == [None, -11.6] and logprobs["top_logprobs"][0] is None
    assert client.post("/v1/completions", json={**FROZEN, "logprobs": 1}).status_code == 502   # the same null without echo


def test_a_streamed_chunk_with_no_text_and_no_finish_still_fails_closed(tmp_path):
    class NoText(TextUpstream):
        async def stream_text(self, body):
            yield None, {}
            yield {"choices": [{"index": 0, "finish_reason": None}]}, {}

    _, client = make(tmp_path, NoText())
    with client.stream("POST", "/v1/completions", json={**FROZEN, "stream": True}) as r:
        frames = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert frames == [json.dumps({"error": {"message": "the response failed while streaming", "type": "server_error",
                                            "param": None, "code": "stream_failed"}})]


@pytest.mark.parametrize("logprobs", [
    "no", {"tokens": "Paris"}, {"tokens": [1]}, {"token_logprobs": ["-0.5"]}, {"token_logprobs": [True]},
    {"text_offset": [True]}, {"text_offset": "0"}, {"top_logprobs": [{"Paris": "x"}]},
    {"top_logprobs": ["Paris"]}, {"top_logprobs": [{"Paris": True}]},
    {"token_logprobs": [-0.5, None]}, {"top_logprobs": [{"The": -0.1}, None]},   # null past the echoed first token
])
def test_an_unusable_logprobs_shape_fails_closed(tmp_path, logprobs):
    data = json.loads(json.dumps(RAW))
    data["choices"][0]["logprobs"] = logprobs
    _, client = make(tmp_path, TextUpstream(data=data))
    r = client.post("/v1/completions", json=FROZEN)
    assert r.status_code == 502 and set(r.json()) == {"error"}


def test_a_null_finish_reason_is_a_streamed_chunk_exception_only(tmp_path):
    """The pinned response requires a non-null enum; only a mid-stream chunk
    may carry null, as OpenAI's own legacy stream does."""
    data = json.loads(json.dumps(RAW))
    data["choices"][0]["finish_reason"] = None
    deps, client = make(tmp_path, TextUpstream(data=data))
    assert client.post("/v1/completions", json=FROZEN).status_code == 502
    assert last_trace(deps.settings)["upstream_shape_error"]
    # the same null mid-stream is fine: the streaming test above asserts it


@pytest.mark.parametrize("name,field", [
    ("prompt_tokens_details", "text_tokens"), ("prompt_tokens_details", "image_tokens"),
    ("prompt_tokens_details", "cache_write_tokens"), ("prompt_tokens_details", "cached_tokens"),
    ("prompt_tokens_details", "audio_tokens"), ("completion_tokens_details", "text_tokens"),
    ("completion_tokens_details", "reasoning_tokens"), ("completion_tokens_details", "audio_tokens"),
    ("completion_tokens_details", "accepted_prediction_tokens"),
    ("completion_tokens_details", "rejected_prediction_tokens"),
])
def test_every_pinned_usage_detail_survives(tmp_path, name, field):
    """Closed-default must not mean lossy: each pinned breakdown field is kept."""
    data = json.loads(json.dumps(RAW))
    data["usage"][name] = {field: 3, "not_in_the_spec": 9}
    _, client = make(tmp_path, TextUpstream(data=data))
    r = client.post("/v1/completions", json=FROZEN)
    assert r.status_code == 200 and r.json()["usage"][name] == {field: 3}


def test_an_empty_completion_text_is_still_valid(tmp_path):
    """The control for the falsy cases: "" is a legitimate completion."""
    data = json.loads(json.dumps(RAW))
    data["choices"][0]["text"] = ""
    _, client = make(tmp_path, TextUpstream(data=data))
    r = client.post("/v1/completions", json=FROZEN)
    assert r.status_code == 200 and r.json()["choices"][0]["text"] == ""


# --- S13h (2026-09-16): the measured backend capabilities, served -----------------------------

def test_n_echo_logprobs_come_back_strict_through_the_sdk(tmp_path):
    raw = {"id": "x", "object": "text_completion", "created": 1, "model": "example/chat",
           "choices": [{"index": i, "text": f"The capital of France is Paris{i}", "finish_reason": "length",
                        "logprobs": {"tokens": [" Paris"], "token_logprobs": [-0.1], "text_offset": [24],
                                     "top_logprobs": [{" Paris": -0.1, " Lyon": -3.2}]}} for i in range(2)],
           "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}}
    up = TextUpstream(data=raw)
    _, client = make(tmp_path, up)
    body = client.post("/v1/completions", json={**FROZEN, "n": 2, "echo": True, "logprobs": 2}).json()
    from qa.conformance.schema import Spec, validate_payload
    row = validate_payload(body, kind="completion", spec=Spec(), fields="strict")
    assert row["verdict"] == "pass", row["evidence"]
    assert [c["index"] for c in body["choices"]] == [0, 1] and body["choices"][0]["logprobs"]["top_logprobs"][0][" Lyon"] == -3.2
    assert (up.text_bodies[-1]["n"], up.text_bodies[-1]["echo"], up.text_bodies[-1]["logprobs"]) == (2, True, 2)
    sdk = openai.OpenAI(api_key="test", base_url="http://testserver/v1", http_client=client, max_retries=0)
    out = sdk.completions.create(model="chord-1-poly", prompt=[1, 2, 3], n=2, logprobs=2)
    assert len(out.choices) == 2 and out.choices[1].logprobs.tokens == [" Paris"]


class UsageStream(TextUpstream):
    """A backend usage chunk: a text-less pseudo-choice (measured 2026-09-16)."""
    async def stream_text(self, body):
        self.text_bodies.append(body)
        yield None, {}
        for piece in (" Paris", "."):
            yield {"id": "x", "object": "text_completion", "created": 1, "model": "example/chat",
                   "choices": [{"index": 0, "text": piece, "finish_reason": None}]}, {}
        yield {"id": "x", "object": "text_completion", "created": 1, "model": "example/chat",
               "choices": [{"index": 0, "text": "", "finish_reason": "length"}]}, {}
        if (body.get("stream_options") or {}).get("include_usage"):
            yield {"id": "x", "object": "text_completion", "created": 1, "model": "example/chat",
                   "choices": [{"index": 0}], "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}, {}


def test_a_non_integer_usage_count_is_an_in_band_stream_error(tmp_path):
    class BadUsage(UsageStream):
        async def stream_text(self, body):
            async for item in super().stream_text(body):
                chunk = item[0]
                if isinstance(chunk, dict) and chunk.get("usage"):
                    chunk = {**chunk, "usage": {**chunk["usage"], "prompt_tokens": "1"}}
                    yield chunk, item[1]
                else:
                    yield item

    _, client = make(tmp_path, BadUsage())
    body = {**FROZEN, "stream": True, "stream_options": {"include_usage": True}}
    with client.stream("POST", "/v1/completions", json=body) as r:
        assert r.status_code == 200
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert any('"stream_failed"' in line for line in lines)
    assert "data: [DONE]" not in lines


@pytest.mark.parametrize("include", [True, False])
def test_include_usage_ends_with_a_choices_empty_usage_chunk(tmp_path, include):
    up = UsageStream()
    _, client = make(tmp_path, up)
    body = {**FROZEN, "stream": True, **({"stream_options": {"include_usage": True}} if include else {})}
    with client.stream("POST", "/v1/completions", json=body) as r:
        frames = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {")]
    from qa.conformance.schema import Spec, validate_payload
    for f in frames:
        if f["choices"] and f["choices"][0]["finish_reason"] is None:
            continue   # OPEN: the pinned schema requires finish_reason non-null, which no mid-stream chunk can meet
        row = validate_payload(f, kind="completion", spec=Spec(), fields="strict")
        assert row["verdict"] == "pass", (f, row["evidence"])
    usage_frames = [f for f in frames if "usage" in f]
    # We ALWAYS ask the backend for usage; the client's flag decides only whether the
    # usage FRAME is forwarded. Asking only when the client asked used to cost every
    # other client the terminal chunk, because the backend emits finish_reason
    # alongside usage (S13h). This assertion used to say "stream_options" not in the
    # upstream body, which pinned the defect in place.
    assert up.text_bodies[-1]["stream_options"] == {"include_usage": True}
    if include:
        assert usage_frames == [frames[-1]] and frames[-1]["choices"] == [] and frames[-1]["usage"]["total_tokens"] == 20
    else:
        assert usage_frames == []
    # S13h: whatever the client asked about usage, the stream must say why it ended.
    ended = [f for f in frames if f["choices"] and f["choices"][0]["finish_reason"] is not None]
    assert ended, "a streamed completion must carry a terminal finish_reason"
    assert ended[-1]["choices"][0]["finish_reason"] in {"stop", "length", "content_filter"}


# review 2026-09-24 B13: only UpstreamShapeError was caught, so a transport drop or a
# malformed SSE line ended the client stream with no error event and a completed trace.
@pytest.mark.parametrize("boom", [httpx.ReadError("connection reset"), json.JSONDecodeError("Expecting value", "{", 1)])
def test_a_mid_stream_transport_or_parse_failure_ends_with_an_error_event(tmp_path, boom):
    class Drops(TextUpstream):
        async def stream_text(self, body):
            yield None, {}
            yield {"choices": [{"index": 0, "text": " Paris", "finish_reason": None}]}, {}
            raise boom

    deps, client = make(tmp_path, Drops())
    with client.stream("POST", "/v1/completions", json={**FROZEN, "stream": True}) as r:
        frames = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert "[DONE]" not in frames and json.loads(frames[-1])["error"]["code"] == "stream_failed"
    assert [json.loads(f)["choices"][0]["text"] for f in frames[:-1]] == [" Paris"]
    t = last_trace(deps.settings)
    assert t["result_status"] == "failed" and t["stream_error"]
