"""The #187 finding: the chat message was built by SUBTRACTION (copy the
upstream message, pop known extras), so any novel upstream field leaked. The
wire is now built from the pinned spec's own field lists at every level: body,
choice, message, stream chunk, delta. Declared extensions only as declared."""
import json

from fastapi.testclient import TestClient

from qa.conformance.schema import Spec, parse_sse, validate_payload
from chord.config import Settings
from chord.server import Deps, create_app
from test_skeleton import FakeUpstream

NOVEL_MESSAGE = {"prompt_logprobs": [0.1], "routed_experts": [3, 7], "kv_transfer_params": {"k": 1}, "vendor_future": "x"}


class Leaky(FakeUpstream):
    async def complete(self, body):
        data, dep = await super().complete(body)
        data["choices"][0]["message"].update(NOVEL_MESSAGE)
        data["choices"][0]["stop_reason"] = 151645
        data["prompt_token_ids"] = [1, 2, 3]
        data["kv_transfer_params"] = None
        return data, dep

    async def stream(self, body):
        yield None, {}
        yield {"object": "chat.completion.chunk", "prompt_token_ids": [1], "choices": [
            {"index": 0, "delta": {"content": "hi", "token_ids": [9], **NOVEL_MESSAGE}, "finish_reason": None, "stop_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": 7}]}, {}


def client(tmp_path):
    return TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=Leaky(), model=lambda n: None)))


def strict(payload, kind):
    row = validate_payload(payload, kind=kind, spec=Spec(), fields="strict")
    assert row["verdict"] == "pass", json.dumps(row["evidence"])[:1500]


def test_novel_upstream_fields_never_reach_a_non_stream_body(tmp_path):
    body = client(tmp_path).post("/v1/chat/completions", json={"model": "chord-1-poly",
                                                               "messages": [{"role": "user", "content": "hi"}]}).json()
    strict(body, "chat")
    assert body["choices"][0]["message"]["content"] == "hi there"
    dumped = json.dumps(body)
    assert not any(k in dumped for k in (*NOVEL_MESSAGE, "prompt_token_ids", "stop_reason"))


def test_novel_upstream_fields_never_reach_a_stream_chunk(tmp_path):
    with client(tmp_path).stream("POST", "/v1/chat/completions", json={"model": "chord-1-poly", "stream": True,
                                                                       "messages": [{"role": "user", "content": "hi"}]}) as r:
        raw = r.read()
    chunks, done, framing = parse_sse(raw)
    assert done and not framing
    for c in chunks:
        strict(c, "chat-stream")
    assert "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices")) == "hi"
    assert not any(k in raw.decode() for k in (*NOVEL_MESSAGE, "prompt_token_ids", "token_ids", "stop_reason"))


def test_declared_reasoning_still_reaches_a_caller_who_turned_thinking_on(tmp_path):
    class Thinks(FakeUpstream):
        async def complete(self, body):
            data, dep = await super().complete(body)
            data["choices"][0]["message"]["reasoning"] = "because"
            data["choices"][0]["message"].pop("reasoning_content", None)
            return data, dep
    c = TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=Thinks(), model=lambda n: None)))
    body = c.post("/v1/chat/completions", json={"model": "chord-1-poly", "chat_template_kwargs": {"enable_thinking": True},
                                                "messages": [{"role": "user", "content": "hi"}]}).json()
    msg = body["choices"][0]["message"]
    assert msg["reasoning_content"] == "because" and "reasoning" not in msg   # one declared name, the upstream alias normalized


# --- one level down (a novel key inside a tool call is still on the wire) ---------------

class LeakyNested(FakeUpstream):
    async def complete(self, body):
        self.bodies.append(body)
        message = {"role": "assistant", "content": "ok", "refusal": None,
                   "tool_calls": [{"id": "call_1", "type": "function", "vllm_trace": "x",
                                   "function": {"name": "f", "arguments": "{}", "parsed": {"a": 1}}}],
                   "annotations": [{"type": "url_citation", "score": 0.9,
                                    "url_citation": {"url": "https://a", "title": "A", "start_index": 0, "end_index": 2, "snippet": "s"}}],
                   "function_call": {"name": "f", "arguments": "{}", "legacy_id": 5}}
        return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]}, {}

    async def stream(self, body):
        yield None, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": None, "delta": {
            "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "vllm_trace": "x",
                            "function": {"name": "f", "arguments": "{}", "parsed": {"a": 1}}}]}}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, {}


NESTED_NOVEL = ("vllm_trace", "parsed", "score", "snippet", "legacy_id")


def nested_client(tmp_path):
    return TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=LeakyNested(), model=lambda n: None)))


def test_novel_keys_inside_nested_objects_are_stripped_non_stream(tmp_path):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    body = nested_client(tmp_path).post("/v1/chat/completions", json={"model": "chord-1-poly", "tools": tools,
                                                                      "messages": [{"role": "user", "content": "hi"}]}).json()
    strict(body, "chat")
    msg = body["choices"][0]["message"]
    assert msg["tool_calls"] == [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    assert msg["annotations"][0]["url_citation"] == {"start_index": 0, "end_index": 2, "url": "https://a", "title": "A"}
    assert not any(k in json.dumps(body) for k in NESTED_NOVEL)


def test_non_nullable_optional_nulls_are_omitted_and_usage_is_closed_default():
    from chord.server import _construct

    body = _construct({
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "hello",
                "refusal": None,
                "annotations": None,
                "function_call": None,
                "tool_calls": None,
            },
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "prompt_tokens_details": None,
            "completion_tokens_details": {"reasoning_tokens": 0, "backend_cache_hits": 9},
            "provider_cost": 0.01,
        },
    }, stream=False)

    message = body["choices"][0]["message"]
    assert message["refusal"] is None
    assert "annotations" not in message
    assert "function_call" not in message
    assert "tool_calls" not in message
    assert body["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def test_novel_keys_inside_streamed_tool_calls_are_stripped(tmp_path):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    with nested_client(tmp_path).stream("POST", "/v1/chat/completions", json={"model": "chord-1-poly", "stream": True, "tools": tools,
                                                                              "messages": [{"role": "user", "content": "hi"}]}) as r:
        raw = r.read()
    chunks, done, framing = parse_sse(raw)
    for c in chunks:
        strict(c, "chat-stream")
    assert not any(k in raw.decode() for k in NESTED_NOVEL)


def test_audio_is_rebuilt_from_its_fields():
    from chord.server import _construct
    out = _construct({"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": [
        {"index": 0, "finish_reason": "stop", "logprobs": None, "message": {"role": "assistant", "content": "x", "refusal": None,
         "audio": {"id": "a", "expires_at": 1, "data": "AA==", "transcript": "x", "voice_internal": "alloy"}}}]}, stream=False)
    assert out["choices"][0]["message"]["audio"] == {"id": "a", "expires_at": 1, "data": "AA==", "transcript": "x"}
