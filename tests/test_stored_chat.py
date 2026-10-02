"""S13a/S13e (red team pass 1): store=true was accepted and nothing was kept
(S-cf-032, S-cf-058), then honestly refused. Since 2026-09-16 it keeps the
completion, and the five stored operations read it back. Official SDK, strict."""
import json

import openai
import pytest

from test_responses import MODEL, make, strict
from test_client_tools_own_the_turn import Calling
from test_skeleton import FakeUpstream
from qa.conformance.schema import Spec, validate_payload

CHAT = [{"role": "user", "content": "hi"}]
ABSENT = object()


class FullUsage(FakeUpstream):
    """The real backend's stream usage carries all three counts (the shared fake only total_tokens)."""
    async def stream(self, body):
        async for chunk, dep in super().stream(body):
            if chunk and chunk.get("usage"):
                chunk = {**chunk, "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
            yield chunk, dep


@pytest.mark.parametrize("stream", [False, True])
def test_store_true_keeps_what_the_caller_got(tmp_path, stream):
    deps, client, sdk = make(tmp_path, FullUsage())
    if stream:
        chunks = list(sdk.chat.completions.create(model=MODEL, messages=CHAT, store=True,
                                                  metadata={"who": "ava"}, stream=True,
                                                  stream_options={"include_usage": True}))
        cid, text = chunks[0].id, "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    else:
        c = sdk.chat.completions.create(model=MODEL, messages=CHAT, store=True, metadata={"who": "ava"})
        cid, text = c.id, c.choices[0].message.content
        assert c.metadata == {"who": "ava"}
    raw = client.get(f"/v1/chat/completions/{cid}").json()
    strict(raw, "chat")
    assert raw["id"] == cid and raw["choices"][0]["message"]["content"] == text == "hi there"
    assert raw["metadata"] == {"who": "ava"}
    assert "store" not in deps.upstream.bodies[0]                                   # ours, never forwarded
    assert "metadata" not in deps.upstream.bodies[0]
    if stream:
        assert raw["usage"]["total_tokens"] == 7


@pytest.mark.parametrize("value", [False, None])
def test_store_false_or_null_keeps_nothing(tmp_path, value):
    deps, client, sdk = make(tmp_path)
    c = client.post("/v1/chat/completions", json={"model": MODEL, "messages": CHAT, "store": value}).json()
    assert client.get(f"/v1/chat/completions/{c['id']}").status_code == 404
    assert client.get("/v1/chat/completions").json()["data"] == []


def test_list_filters_update_messages_delete(tmp_path):
    deps, client, sdk = make(tmp_path)
    a = sdk.chat.completions.create(model=MODEL, messages=CHAT, store=True, metadata={"team": "kitchen"})
    b = sdk.chat.completions.create(model=MODEL, store=True, metadata={"team": "bath"},
                                    messages=[{"role": "system", "content": "Be kind."},
                                              {"role": "user", "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "there"}]}])
    listing = client.get("/v1/chat/completions").json()
    strict(listing, "chat-list")
    assert [c["id"] for c in listing["data"]] == [a.id, b.id]                          # default order asc
    assert [c.id for c in sdk.chat.completions.list(metadata={"team": "bath"}).data] == [b.id]
    assert [c.id for c in sdk.chat.completions.list(order="desc", limit=1).data] == [b.id]
    assert sdk.chat.completions.list(model="chord-1-other").data == []

    updated = sdk.chat.completions.update(a.id, metadata={"team": "living-room"})
    assert updated.id == a.id
    assert [c.id for c in sdk.chat.completions.list(metadata={"team": "living-room"}).data] == [a.id]

    msgs = client.get(f"/v1/chat/completions/{b.id}/messages").json()
    # The pinned spec types these items as ChatCompletionResponseMessage, whose role
    # enum is only "assistant", for a list of the REQUEST's messages. We report the
    # true role; every other rule is held strict.
    row = validate_payload(msgs, kind="chat-messages", spec=Spec(), fields="strict")
    errors = row["evidence"].get("errors", [])
    assert not row["evidence"].get("undeclared_fields")
    assert all(e["json_path"].endswith(".role") and e["validator"] == "enum" and e["expectation"] == ["assistant"] for e in errors), errors
    assert len(errors) == 2
    assert [(m["role"], m["content"], m["content_parts"] is not None) for m in msgs["data"]] == [
        ("system", "Be kind.", False), ("user", "hello there", True)]
    assert [m.id for m in sdk.chat.completions.messages.list(b.id).data] == [f"{b.id}-0", f"{b.id}-1"]

    deleted = sdk.chat.completions.delete(a.id)
    assert deleted.deleted is True
    with pytest.raises(openai.NotFoundError):
        sdk.chat.completions.retrieve(a.id)


def test_a_streamed_tool_call_is_stored_assembled(tmp_path):
    deps, client, sdk = make(tmp_path, Calling())
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    chunks = list(sdk.chat.completions.create(model=MODEL, messages=CHAT, tools=tools, store=True, stream=True))
    stored = sdk.chat.completions.retrieve(chunks[0].id)
    assert [t.function.arguments for t in stored.choices[0].message.tool_calls] == ['{"city": "Paris"}', '{"city": "Oslo"}']
    assert stored.choices[0].finish_reason == "tool_calls"


def test_the_responses_door_does_not_store_chat_completions(tmp_path):
    deps, client, sdk = make(tmp_path)
    sdk.responses.create(model=MODEL, input="hi")
    assert client.get("/v1/chat/completions").json()["data"] == []


@pytest.mark.parametrize("method,path,body,status,param", [
    ("POST", "/v1/chat/completions/chatcmpl-x", {"metadata": {}}, 404, None),
    ("GET", "/v1/chat/completions/chatcmpl-x/messages", None, 404, None),
    ("DELETE", "/v1/chat/completions/chatcmpl-x", None, 404, None),
    ("GET", "/v1/chat/completions?limit=0", None, 400, "limit"),
])
def test_stored_errors(tmp_path, method, path, body, status, param):
    deps, client, sdk = make(tmp_path)
    r = client.request(method, path, json=body)
    assert r.status_code == status and r.json()["error"]["param"] == param


def test_metadata_is_returned_on_the_stored_object_and_an_update_shows(tmp_path):
    """The S13e gate (2026-09-17): an update answered 200 with no metadata on the object, so nothing
    showed it had taken. The pinned CreateChatCompletionResponse has `metadata`."""
    deps, client, sdk = make(tmp_path)
    c = sdk.chat.completions.create(model=MODEL, messages=CHAT, store=True, metadata={"team": "kitchen"})
    got = client.get(f"/v1/chat/completions/{c.id}").json()
    strict(got, "chat")
    assert got["metadata"] == {"team": "kitchen"}
    updated = client.post(f"/v1/chat/completions/{c.id}", json={"metadata": {"team": "bath", "k": "v"}}).json()
    strict(updated, "chat")
    assert updated["metadata"] == {"team": "bath", "k": "v"}
    assert client.get(f"/v1/chat/completions/{c.id}").json()["metadata"] == {"team": "bath", "k": "v"}   # read back
    listing = client.get("/v1/chat/completions").json()
    strict(listing, "chat-list")
    assert listing["data"][0]["metadata"] == {"team": "bath", "k": "v"}
    plain = sdk.chat.completions.create(model=MODEL, messages=CHAT, store=True)
    assert plain.metadata == {}
    assert client.get(f"/v1/chat/completions/{plain.id}").json()["metadata"] == {}


@pytest.mark.parametrize("metadata", [
    {f"k{i}": "v" for i in range(17)},
    {"k" * 65: "v"},
    {"k": "v" * 513},
])
def test_create_rejects_metadata_the_store_cannot_honour(tmp_path, metadata):
    deps, client, sdk = make(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": MODEL, "messages": CHAT,
                                                   "store": True, "metadata": metadata})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "metadata"
    assert deps.upstream.bodies == []


def test_null_metadata_is_canonical_on_create_and_retrieve(tmp_path):
    deps, client, sdk = make(tmp_path)
    created = client.post("/v1/chat/completions", json={"model": MODEL, "messages": CHAT,
                                                         "store": True, "metadata": None}).json()
    retrieved = client.get(f"/v1/chat/completions/{created['id']}").json()
    assert created["metadata"] == retrieved["metadata"] == {}
    assert "metadata" not in deps.upstream.bodies[0]


@pytest.mark.parametrize("metadata", [ABSENT, None, {"k": "v"}], ids=["absent", "null", "present"])
@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("n", [None, 2], ids=["single", "n=2"])
def test_metadata_response_matrix(tmp_path, metadata, store, n):
    deps, client, sdk = make(tmp_path)
    request = {"model": MODEL, "messages": CHAT, "store": store}
    if metadata is not ABSENT:
        request["metadata"] = metadata
    if n is not None:
        request["n"] = n
    created = client.post("/v1/chat/completions", json=request).json()
    expected = None if metadata is ABSENT and not store else ({} if metadata in (ABSENT, None) else metadata)
    if expected is None:
        assert "metadata" not in created
    else:
        assert created["metadata"] == expected
    if store:
        retrieved = client.get(f"/v1/chat/completions/{created['id']}").json()
        assert retrieved["metadata"] == created["metadata"]
    assert all("metadata" not in upstream for upstream in deps.upstream.bodies)


def test_update_answers_404_when_the_row_vanishes_mid_request(tmp_path, monkeypatch):
    """A DELETE (or the retention boundary) between the existence check and the
    read-back is a 404, not a TypeError on None through the generic 500
    (review 2026-09-22)."""
    from chord.responses_store import ResponseStore

    deps, client, sdk = make(tmp_path)
    c = sdk.chat.completions.create(model=MODEL, messages=CHAT, store=True)
    real = ResponseStore.get_chat
    calls = {"n": 0}

    def vanishing(self, completion_id):
        calls["n"] += 1
        return real(self, completion_id) if calls["n"] == 1 else None

    monkeypatch.setattr(ResponseStore, "get_chat", vanishing)
    r = client.post(f"/v1/chat/completions/{c.id}", json={"metadata": {"x": "y"}})
    assert r.status_code == 404, r.text
    strict(r.json(), "error")


# review 2026-09-24 B15: a stored streamed completion kept usage only when the client
# asked include_usage, and dropped logprobs, annotations, system_fingerprint and the
# legacy function_call, while the non-stream store kept them.
LOGPROBS = [{"token": "hi", "logprob": -0.1, "bytes": [104, 105], "top_logprobs": []},
            {"token": " there", "logprob": -0.2, "bytes": None, "top_logprobs": []}]


class HonestUsage(FakeUpstream):
    """A backend as LiteLLM/vLLM behave: usage only when stream_options asks for it,
    per-chunk logprobs, and a system_fingerprint on every chunk."""

    async def complete(self, body):
        self.bodies.append(body)
        return ({"choices": [{"index": 0, "finish_reason": "stop", "logprobs": {"content": LOGPROBS, "refusal": None},
                              "message": {"role": "assistant", "content": "hi there"}}],
                 "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                 "system_fingerprint": "vllm-0.28.0"}, {})

    async def stream(self, body):
        self.bodies.append(body)
        asked = (body.get("stream_options") or {}).get("include_usage") is True
        extra = {"usage": None} if asked else {}
        yield None, {}
        for piece, lp in zip(["hi", " there"], LOGPROBS):
            yield {"object": "chat.completion.chunk", "system_fingerprint": "vllm-0.28.0", **extra,
                   "choices": [{"index": 0, "delta": {"content": piece}, "logprobs": {"content": [lp]}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "system_fingerprint": "vllm-0.28.0", **extra,
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}
        if asked:
            yield {"object": "chat.completion.chunk", "system_fingerprint": "vllm-0.28.0", "choices": [],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}, {}


def test_a_stored_streamed_completion_carries_what_the_non_stream_one_does(tmp_path):
    deps, client, sdk = make(tmp_path, HonestUsage())
    plain = sdk.chat.completions.create(model=MODEL, messages=CHAT, store=True, logprobs=True)
    with client.stream("POST", "/v1/chat/completions", json={"model": MODEL, "messages": CHAT, "store": True,
                                                              "logprobs": True, "stream": True}) as r:
        frames = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    streamed_id = json.loads(frames[0])["id"]
    assert all('"usage"' not in f for f in frames), "the client did not ask for usage and must not get it"
    want = client.get(f"/v1/chat/completions/{plain.id}").json()
    got = client.get(f"/v1/chat/completions/{streamed_id}").json()
    strict(got, "chat")
    assert got["usage"] == want["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert got["system_fingerprint"] == want["system_fingerprint"] and got["system_fingerprint"].startswith("fp_")
    assert got["choices"][0]["logprobs"] == want["choices"][0]["logprobs"]
    assert got["choices"][0]["logprobs"]["content"] == LOGPROBS
    assert set(got) == set(want)


def test_completion_from_chunks_keeps_annotations_and_a_legacy_function_call():
    from chord.stored_chat import completion_from_chunks
    note = {"type": "url_citation", "url_citation": {"start_index": 0, "end_index": 2, "url": "https://a.example", "title": "A"}}
    head = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    got = completion_from_chunks([
        {**head, "choices": [{"index": 0, "delta": {"content": "hi", "annotations": [note]}, "finish_reason": None}]},
        {**head, "choices": [{"index": 0, "delta": {"function_call": {"name": "get_", "arguments": '{"a"'}}, "finish_reason": None}]},
        {**head, "choices": [{"index": 0, "delta": {"function_call": {"name": "weather", "arguments": ": 1}"}}, "finish_reason": None}]},
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]},
    ])
    assert got is not None
    message = got["choices"][0]["message"]
    assert message["annotations"] == [note]
    assert message["function_call"] == {"name": "get_weather", "arguments": '{"a": 1}'}
    assert got["choices"][0]["finish_reason"] == "function_call"
