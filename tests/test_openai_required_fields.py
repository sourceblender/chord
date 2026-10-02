"""Fields the pinned OpenAI spec requires, present on every path we build.

Found by qa/conformance on the live build (2026-09-14): non-stream choices had
no `logprobs`, messages no `refusal`, stream choices no `finish_reason`;
non-stream dropped upstream `logprobs` and `system_fingerprint`; unknown request
fields were forwarded unread. Each test here is one of those findings.
"""
import json


from chord import manifest
from chord.server import CHAT_SPEC_PARAMS
from test_skeleton import FakeUpstream, make

CHAT = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
PARAMS_FILE = "qa/conformance/params/chat_request_params.json"


class RichUpstream(FakeUpstream):
    """Upstream that sends logprobs, a fingerprint and a relocated refusal."""

    async def complete(self, body):
        self.bodies.append(body)
        lp = {"content": [{"token": "hi", "logprob": -0.1, "bytes": [104, 105], "top_logprobs": []}], "refusal": None}
        return (
            {"choices": [{"index": 0, "finish_reason": "stop", "logprobs": lp,
                          "message": {"role": "assistant", "content": "hi there",
                                      "provider_specific_fields": {"refusal": "no"}}}],
             "system_fingerprint": "fp_test", "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
            {},
        )


def test_chat_spec_params_equal_the_pinned_list():
    with open(PARAMS_FILE) as fh:
        assert CHAT_SPEC_PARAMS == set(json.load(fh)["params"])


def test_non_stream_choice_has_logprobs_and_message_has_refusal(tmp_path):
    _, client = make(tmp_path)
    choice = client.post("/v1/chat/completions", json=CHAT).json()["choices"][0]
    assert "logprobs" in choice and choice["logprobs"] is None
    assert "refusal" in choice["message"] and choice["message"]["refusal"] is None


def test_non_stream_passes_upstream_logprobs_fingerprint_and_refusal_through(tmp_path):
    _, client = make(tmp_path, RichUpstream())
    body = client.post("/v1/chat/completions", json={**CHAT, "logprobs": True}).json()
    assert body["choices"][0]["logprobs"]["content"][0]["token"] == "hi"
    assert body["system_fingerprint"].startswith("fp_") and body["system_fingerprint"] != "fp_test"   # opaque, never the raw value
    assert body["choices"][0]["message"]["refusal"] == "no"


class LiveShapedStream(FakeUpstream):
    """Content chunks WITHOUT a finish_reason key, as the live upstream sends
    them (direct probe on 124320a, 2026-09-14: choice keys ['delta', 'index']).
    FakeUpstream sends an explicit None, which hid this defect from the test."""

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for piece in ["hi", " there"]:
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def test_every_stream_chunk_choice_has_finish_reason(tmp_path):
    _, client = make(tmp_path, LiveShapedStream())
    with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as r:
        chunks = [json.loads(line[6:]) for line in r.iter_lines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
    choices = [c for ch in chunks for c in ch.get("choices") or []]
    assert choices and all("finish_reason" in c and "index" in c for c in choices)
    assert sum(1 for c in choices if c["finish_reason"]) == 1


def test_a_declared_noop_is_accepted_but_never_forwarded(tmp_path):
    """The declared-no-op mechanism: accepted and validated at the door, never sent
    upstream. `user` is the exemplar. `prediction` is not: the spec's token counts
    are observable, so that field is refused rather than accepted and dropped."""
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**CHAT, "user": "caller-1"})
    assert r.status_code == 200, r.text
    assert up.bodies[0].get("user") in (None, "caller-1") or "user" not in up.bodies[0]
    bad = client.post("/v1/chat/completions", json={**CHAT, "user": 5})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "user"


def test_unknown_field_is_refused_like_openai_not_forwarded(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**CHAT, "zz_not_an_openai_param": 1})
    err = r.json()["error"]
    assert r.status_code == 400 and err["code"] == "unsupported_parameter" and err["param"] == "zz_not_an_openai_param"
    assert set(err) == {"message", "type", "param", "code"} and up.bodies == []


def test_declared_extension_and_noop_params_are_accepted(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    extra = {"chat_template_kwargs": {"enable_thinking": False}, "metadata": {"k": "v"}, "user": "u"}
    assert client.post("/v1/chat/completions", json={**CHAT, **extra}).status_code == 200
    assert up.bodies


def test_ledger_lists_are_disjoint_and_inside_the_spec():
    m = manifest.load()
    refused, noop, ext = set(m["refused_params"]), set(m["declared_noop_params"]), set(m["extension_params"])
    assert not (refused & noop) and not (set(m["supported_openai_params"]) & (refused | noop))
    assert refused <= CHAT_SPEC_PARAMS and noop <= CHAT_SPEC_PARAMS and not (ext & CHAT_SPEC_PARAMS)
    # An extension is not OpenAI support: /v1/models and LiteLLM model_info
    # publish supported_openai_params as exactly that.
    assert not (ext & set(m["supported_openai_params"]))
    assert set(m["supported_openai_params"]) <= CHAT_SPEC_PARAMS


class EmptyFingerprint(RichUpstream):
    async def complete(self, body):
        data, dep = await super().complete(body)
        data["system_fingerprint"] = ""
        return data, dep


def test_empty_system_fingerprint_is_passed_through_not_dropped(tmp_path):
    """The spec allows any string; truthiness dropped "" (#117 review)."""
    _, client = make(tmp_path, EmptyFingerprint())
    body = client.post("/v1/chat/completions", json=CHAT).json()
    assert "system_fingerprint" in body and body["system_fingerprint"] == ""


def test_the_upstream_provider_block_is_scrubbed_after_its_refusal_is_restored(tmp_path):
    """#142: the inner hop's message.provider_specific_fields (reasoning,
    refusal) is not our contract. Refusal moves to its spec place; the block goes."""
    class ReasoningUpstream(RichUpstream):
        async def complete(self, body):
            payload, extra = await super().complete(body)
            payload["choices"][0]["message"]["provider_specific_fields"] = {"refusal": "no", "reasoning": "thinking"}
            return payload, extra

    _, client = make(tmp_path, ReasoningUpstream())
    message = client.post("/v1/chat/completions", json=CHAT).json()["choices"][0]["message"]
    assert message["refusal"] == "no"
    assert "provider_specific_fields" not in message
