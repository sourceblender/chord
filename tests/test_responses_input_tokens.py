"""POST /v1/responses/input_tokens: the exact prompt token count, read from the
model's own usage for a one-token prefill of exactly what create would send."""
import pytest
from fastapi import Response

from test_responses import MODEL, make, strict
from test_skeleton import FakeUpstream


class Counting(FakeUpstream):
    async def complete(self, body):
        data, dep = await super().complete(body)
        data["usage"] = {"prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43}
        return data, dep


def test_the_count_is_the_models_and_the_prefill_is_one_token(tmp_path):
    deps, client, sdk = make(tmp_path, Counting(), router=lambda: (_ for _ in ()).throw(AssertionError("router ran")),
                             router_enabled=True, enabled_routes=frozenset({"image"}))
    r = client.post("/v1/responses/input_tokens", json={"model": MODEL, "input": "hello", "instructions": "Be brief.",
                                                          "tools": [{"type": "image_generation"}]})
    assert r.status_code == 200, r.text
    strict(r.json(), "response-input-tokens")
    assert r.json() == {"object": "response.input_tokens", "input_tokens": 42}
    sent = deps.upstream.bodies[0]
    assert sent["max_completion_tokens"] == 1 and "Be brief." in sent["messages"][0]["content"]
    assert client.get("/v1/chat/completions").json()["data"] == []            # nothing stored
    assert sdk.responses.input_tokens.count(model=MODEL, input="hello").input_tokens == 42


def test_it_counts_a_chained_conversation(tmp_path):
    deps, client, sdk = make(tmp_path, Counting())
    first = sdk.responses.create(model=MODEL, input="my name is Ava")
    sdk.responses.input_tokens.count(model=MODEL, input="and yours?", previous_response_id=first.id)
    roles = [m["role"] for m in deps.upstream.bodies[-1]["messages"] if m["role"] != "system"]
    assert roles == ["user", "assistant", "user"]


def test_an_empty_conversation_is_not_a_500(tmp_path):
    """Creating a conversation with no items is legal. Counting tokens for it
    used to call input_to_messages([]) outside the Refusal handler."""
    _, client, _sdk = make(tmp_path, Counting())
    conv = client.post("/v1/conversations", json={"items": []})
    assert conv.status_code == 200, conv.text
    r = client.post("/v1/responses/input_tokens", json={
        "model": MODEL, "conversation": conv.json()["id"], "input": "hi"})
    assert r.status_code == 200, r.text
    assert r.json()["input_tokens"] == 42


def test_no_usage_is_a_502_not_a_guess(tmp_path):
    class NoUsage(FakeUpstream):
        async def complete(self, body):
            data, dep = await super().complete(body)
            data.pop("usage", None)
            return data, dep
    deps, client, sdk = make(tmp_path, NoUsage())
    r = client.post("/v1/responses/input_tokens", json={"model": MODEL, "input": "hello"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "token_count_unavailable"


def test_early_disconnect_returns_empty_499_without_parsing(tmp_path):
    _, client, _ = make(tmp_path, Counting())

    async def disconnected(*_args, **_kwargs):
        return Response(status_code=499)

    client.app.state.run_chat = disconnected
    result = client.post("/v1/responses/input_tokens", json={"model": MODEL, "input": "hello"})
    assert result.status_code == 499 and result.content == b""


@pytest.mark.parametrize("body,param", [({"personality": "friendly"}, "personality"), ({"stream": True}, "stream"),
                                        ({"previous_response_id": "resp_x"}, "previous_response_id")])
def test_refusals(tmp_path, body, param):
    deps, client, sdk = make(tmp_path, Counting())
    r = client.post("/v1/responses/input_tokens", json={"model": MODEL, "input": "x", **body})
    assert r.status_code == 400 and r.json()["error"]["param"] == param
    assert deps.upstream.bodies == []
