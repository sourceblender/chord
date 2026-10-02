"""Reasoning on Responses: with reasoning.effort above none, the model's thinking
comes back as the spec's reasoning item (reasoning_text), before her answer, in
create, stream and replay. Without it, reasoning stays dropped as on Chat."""
import json

import pytest

from test_responses import MODEL, events_of, make, strict
from test_responses_stream_replay import replay
from test_skeleton import FakeUpstream


class Thinking(FakeUpstream):
    async def complete(self, body):
        self.bodies.append(body)
        return ({"choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Four.", "reasoning_content": "Two plus two is four."}}],
                 "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14,
                           "completion_tokens_details": {"reasoning_tokens": 6}}}, {})

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for piece in ("Two plus two ", "is four."):
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"reasoning_content": piece}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Four."}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def test_effort_returns_a_reasoning_item_before_the_answer(tmp_path):
    deps, client, sdk = make(tmp_path, Thinking())
    body = client.post("/v1/responses", json={"model": MODEL, "input": "2+2?", "reasoning": {"effort": "low"}}).json()
    strict(body, "response")
    assert [i["type"] for i in body["output"]] == ["reasoning", "message"]
    assert body["output"][0]["content"] == [{"type": "reasoning_text", "text": "Two plus two is four."}]
    assert body["usage"]["output_tokens_details"]["reasoning_tokens"] == 6
    # The effort reaches the backend as its thinking switch, not as itself: it is a
    # backend-ism no harness can send, and translating it is ours since LiteLLM left
    # the middle (2026-09-18).
    assert deps.upstream.bodies[0]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
    for e in replay(client, body["id"]):
        strict(e, "response-event")


@pytest.mark.parametrize("reasoning", [None, {"effort": "none"}])
def test_no_effort_means_no_reasoning_item(tmp_path, reasoning):
    deps, client, sdk = make(tmp_path, Thinking())
    body = client.post("/v1/responses", json={"model": MODEL, "input": "2+2?", **({"reasoning": reasoning} if reasoning else {})}).json()
    assert [i["type"] for i in body["output"]] == ["message"] and "Two plus two" not in json.dumps(body)


def test_a_streamed_reasoning_item_is_strict_and_ordered(tmp_path):
    deps, client, sdk = make(tmp_path, Thinking())
    events = events_of(client, {"model": MODEL, "input": "2+2?", "reasoning": {"effort": "high"}})
    for e in events:
        strict(e, "response-event")
    kinds = [e["type"] for e in events]
    assert kinds[:6] == ["response.created", "response.in_progress", "response.output_item.added",
                         "response.reasoning_text.delta", "response.reasoning_text.delta", "response.reasoning_text.done"]
    final = events[-1]["response"]
    assert [i["type"] for i in final["output"]] == ["reasoning", "message"]
    assert final["output"][0]["content"][0]["text"] == "Two plus two is four."
    with sdk.responses.stream(model=MODEL, input="2+2?", reasoning={"effort": "high"}) as s:
        for _ in s:
            pass
        assert s.get_final_response().output_text == "Four."
