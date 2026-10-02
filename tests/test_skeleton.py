import base64
import hashlib
import asyncio
import json
import re
import struct
import time
import zlib

import pytest
from fastapi.testclient import TestClient

from chord import artifact_links, registry
from chord.artifacts import ArtifactError, ArtifactStore
from chord.config import Settings
from chord.router import parse
from chord.server import load_specialists, Deps, create_app, create_internal_app
from chord.upstream import UpstreamError

# A real 1x1 PNG (IHDR, IDAT, IEND): the artifact store parses PNGs and drops
# ancillary chunks (T01), so a signature followed by filler is no longer accepted.
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
       b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xdf\xc0\x00\x00\x04\x01\x01\x80\xc5*\x18]\x00\x00\x00\x00IEND\xaeB`\x82")


class AvailableImageBackend:
    """Marks a configured provider while a test substitutes the image specialist."""

    async def render(self, prompt):
        raise AssertionError("the substituted specialist must handle this render")

    async def aclose(self):
        pass


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


# A 66-byte PNG whose IHDR claims 14000x14000 -- 196M pixels, past Pillow's hard
# limit of 178,956,970. Image.open raises DecompressionBombError while reading the
# IHDR, before any pixel data, so the IDAT only needs a valid CRC, not valid
# pixels. DecompressionBombError inherits from Exception alone, so an image door
# catching (UnidentifiedImageError, OSError) turns this into a bare 500: every
# route that accepts an image must refuse it as a 400 (review 2026-09-22).
BOMB_PNG = (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 14000, 14000, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(b"\x00"))
            + _png_chunk(b"IEND", b""))


class FakeUpstream:
    """Records what the service sent; replies like a real deployment."""

    def __init__(self, fail_status: int | None = None, usage_choices=None):
        self.bodies = []
        self.fail_status = fail_status
        # OpenAI sends choices=[] on the usage chunk; LiteLLM sends [{"index":0,"delta":{}}].
        self.usage_choices = [] if usage_choices is None else usage_choices

    async def complete(self, body):
        # A real upstream call suspends. A fake that returns without yielding means the event loop
        # never interleaves concurrent turns, so NO concurrency defect can appear in any test that
        # uses it: `n` above 1 stampeded the STT cache and the suite could not have caught it
        # (bug bounty 2026-09-17). One yield point is the difference.
        await asyncio.sleep(0)
        self.bodies.append(body)
        if self.fail_status:
            raise UpstreamError(self.fail_status, json.dumps({"error": {"message": "bad param"}}))
        return (
            {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi there", "reasoning_content": "thinking"}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
            {"model-api-base": "http://10.9.8.7:8113/v1", "model-group": "example/chat"},
        )

    async def stream(self, body):
        await asyncio.sleep(0)   # suspends, as a real stream does (see complete)
        self.bodies.append(body)
        if self.fail_status:
            raise UpstreamError(self.fail_status, json.dumps({"error": {"message": "bad param"}}))
        yield None, {"model-api-base": "http://10.9.8.7:8113/v1"}
        for piece in ["hi", " there"]:
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}
        yield {"object": "chat.completion.chunk", "choices": self.usage_choices, "usage": {"total_tokens": 7}}, {}

    async def aclose(self):
        pass


def make(tmp_path, upstream=None):
    settings = Settings(data_dir=tmp_path,
                        persona_model="example-persona", router_model="example-router",
                        persona_base_url="http://persona.test/v1", router_base_url="http://router.test/v1",
                        stt_model="example-stt", tts_model="example-tts",
                        persona_thinking_mode="qwen_chat_template")
    deps = Deps(settings, upstream=upstream or FakeUpstream(), model=lambda name: None)
    return deps, TestClient(create_app(deps))


def test_models_lists_generic(tmp_path):
    _, client = make(tmp_path)
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "chord-1-poly" in ids


def test_unknown_model_is_404(tmp_path):
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-open", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize("part", [
    {"type": "file", "file": {"file_data": "data:application/pdf;base64,AA==", "filename": "a.pdf"}},
    {"type": "input_file", "file_data": "AA=="},
])
def test_unsupported_parts_are_refused_not_dropped(tmp_path, part):
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}, part]}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_content_part"


def test_audio_modality_needs_its_audio_params(tmp_path):
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "modalities": ["text", "audio"], "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_audio_request"


def test_plain_chat_forwards_params_and_marks_outcome(tmp_path):
    up = FakeUpstream()
    deps, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "temperature": 0.3, "reasoning_effort": "high", "max_tokens": 50, "seed": 7,
        "messages": [{"role": "system", "content": "You are Ava now."}, {"role": "user", "content": "hello"}],
    })
    assert r.status_code == 200, r.text
    sent = up.bodies[0]
    # parameters reach the model untouched; the model id is the persona deployment
    assert (sent["temperature"], sent["max_tokens"], sent["seed"]) == (0.3, 50, 7)
    # reasoning_effort is not forwarded: it IS the backend's thinking switch, and we
    # translate it ourselves now that nothing sits in the middle doing it for us.
    assert "reasoning_effort" not in sent
    assert sent["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "high"}
    assert sent["model"] == deps.settings.persona_model
    # one system message (templates allow only one): our base, then the client's text, untouched and last
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"].endswith("\n\nYou are Ava now.")
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg["content"] == "hi there"
    assert "reasoning_content" not in msg  # OpenAI never returns reasoning unless the client opts in (#143)
    assert set(msg) <= {"role", "content", "refusal", "tool_calls", "annotations", "audio", "function_call"}
    assert body["usage"]["total_tokens"] == 7
    assert body["model"] == "chord-1-poly"
    trace_id = r.headers["x-chord-trace-id"]
    assert r.headers["x-request-id"] == trace_id  # the header OpenAI callers read; never in the body
    record = TestClient(create_internal_app(deps)).get(f"/internal/traces/{trace_id}").json()
    assert record["persona_deployment"]["model-api-base"] == "http://10.9.8.7:8113/v1"
    assert record["client_instruction_messages"] == 1
    assert record["route_decision"] == "chat"


@pytest.mark.parametrize("usage_choices", [[], [{"index": 0, "delta": {}}]], ids=["openai", "litellm"])
def test_stream_order_finish_before_usage_then_done(tmp_path, usage_choices):
    _, client = make(tmp_path, FakeUpstream(usage_choices=usage_choices))
    with client.stream("POST", "/v1/chat/completions", json={"model": "chord-1-poly", "stream": True,
                                                            "stream_options": {"include_usage": True},
                                                            "messages": [{"role": "user", "content": "hi"}]}) as r:
        assert r.status_code == 200
        frames = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(f) for f in frames[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))
    assert text == "hi there"
    finish, usage = chunks[-2], chunks[-1]
    assert finish["choices"][0]["finish_reason"] == "stop" and finish["choices"][0]["delta"] == {}
    assert sum(1 for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")) == 1
    # The spec's usage chunk has choices: [] whichever shape the upstream sent
    # (S12, red team 2026-09-15). LiteLLM's [{"index":0,"delta":{}}] used to
    # pass through with an explicit null finish_reason added.
    assert usage["usage"]["total_tokens"] == 7
    assert usage["choices"] == []
    assert all(c["model"] == "chord-1-poly" for c in chunks)


def test_upstream_refusal_before_stream_is_a_real_http_error(tmp_path):
    _, client = make(tmp_path, FakeUpstream(fail_status=400))
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400


def test_artifact_register_validates_and_hashes(tmp_path):
    store = ArtifactStore(tmp_path)
    d = store.register(PNG, "image/png")
    assert d.sha256 == hashlib.sha256(PNG).hexdigest()
    assert store.locate(d.id)[1] == "image/png"
    with pytest.raises(ArtifactError):
        store.register(b"not a png", "image/png")
    assert store.locate("../etc") is None


def test_capability_routable_only_when_certification_matches(tmp_path):
    caps = registry.load()
    image = caps["image"]
    assert not image.routable
    image.certified = {"passed_at": "2026-09-11", "model": image.model, "prompt_version": image.prompt_version}
    assert image.routable
    image.certified["model"] = "example/other"
    assert not image.routable


def test_router_unparseable_falls_back_to_chat():
    d = parse("sure, I'll render that!", {"image"})
    assert d.route == "chat" and d.parse_error
    d = parse('{"route": "image", "intent": "selfie", "constraints": "same outfit"}', {"image"})
    assert d.route == "image" and d.constraints == ["same outfit"]
    assert parse('{"route": "launch_missiles"}', {"image"}).parse_error


class UnreachableUpstream(FakeUpstream):
    async def complete(self, body):
        import httpx
        raise httpx.ConnectTimeout("no route")


def test_unreachable_gateway_is_an_openai_502_not_a_bare_500(tmp_path):
    _, client = make(tmp_path, UnreachableUpstream())
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_unreachable"
    assert r.headers["x-chord-trace-id"]


def make_keyed(tmp_path):
    settings = Settings(data_dir=tmp_path, service_api_key="s3cret")
    deps = Deps(settings, upstream=FakeUpstream(), model=lambda name: None)
    return TestClient(create_app(deps))


def test_key_required_when_configured_but_health_is_open(tmp_path):
    client = make_keyed(tmp_path)
    req = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=req).status_code == 401
    assert client.post("/v1/chat/completions", json=req, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json=req, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/health").json() == {"status": "ok"}          # S10: no detail without the key
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).json() == {"status": "ok"}
    assert "revision" in client.get("/health", headers={"Authorization": "Bearer s3cret"}).json()


def test_bogus_reasoning_effort_refused_valid_forwarded(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    base = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json={**base, "reasoning_effort": "bogus"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_reasoning_effort"
    assert up.bodies == []  # refused before anything reached the model
    assert client.post("/v1/chat/completions", json={**base, "reasoning_effort": "none"}).status_code == 200
    # "none" means do not think, and reaches the backend as the switch, not as itself.
    assert "reasoning_effort" not in up.bodies[0]
    assert up.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_current_reasoning_efforts_reach_the_backend(tmp_path, effort):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": effort}

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert up.bodies[0]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": effort}


@pytest.mark.parametrize("body", [
    [{"model": "chord-1-poly"}],
    {"model": "chord-1-poly", "messages": [None]},
    {"model": "chord-1-poly", "messages": [{"content": "no role"}]},
    {"model": "chord-1-poly", "messages": [{"role": "user", "content": 42}]},
    {"model": "chord-1-poly", "messages": [{"role": "user", "content": ["not a part"]}]},
    {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": {}},
    {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}], "modalities": "text"},
], ids=["top-level-array", "null-message", "no-role", "int-content", "bare-string-part", "dict-effort", "string-modalities"])
def test_malformed_shapes_are_structured_400s_not_500s(tmp_path, body):
    settings = Settings(data_dir=tmp_path)
    deps = Deps(settings, upstream=FakeUpstream(), model=lambda name: None)
    client = TestClient(create_app(deps), raise_server_exceptions=False)
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400, r.text
    assert "error" in r.json()


def test_models_are_the_openai_model_object_only(tmp_path):
    """S15: capabilities live in the manifest and LiteLLM's model_info, not on the wire.

    The field set is the spec's Model object exactly, on BOTH routes — list and
    retrieve. `shutdown_date` is optional in the schema and present in the spec's
    example for each route, as required by the 2026-09-17 decision."""
    _, client = make(tmp_path)
    listed = client.get("/v1/models").json()["data"][0]
    retrieved = client.get(f"/v1/models/{listed['id']}").json()
    fields = {"id", "object", "created", "owned_by", "shutdown_date"}
    for entry in (listed, retrieved):
        assert set(entry) == fields
        assert entry["shutdown_date"] is None
    assert retrieved == listed


def test_model_created_is_a_real_unix_timestamp(tmp_path):
    """We served `created: 0` — a valid integer that renders as 1970-01-01 in any
    client that formats it. A live endpoint check caught this; 0 is the
    one value that passes a type check and still means nothing."""
    _, client = make(tmp_path)
    for entry in (client.get("/v1/models").json()["data"][0],
                  client.get("/v1/models/chord-1-poly").json()):
        created = entry["created"]
        assert isinstance(created, int) and not isinstance(created, bool)
        # After this repo's first commit and not in the future: a date, not a placeholder.
        assert 1_757_000_000 < created <= int(time.time())


def test_the_refused_params_list_is_pinned_by_contents_not_by_emptiness(tmp_path):
    """A refusal is a capability claim in the negative, so it may never appear OR
    DISAPPEAR silently. This test pins the exact contents of `refused_params`, and
    both directions are load-bearing.

    Written 2026-09-18 asserting `== []`, which is how it spent its short life
    guarding a bug. `moderation` had just been moved OUT of this list by applying
    The decision about the `POST /v1/moderations` route and Chat's `moderation`
    PARAMETER — two different objects. The route's no-op answers visibly
    (`flagged: false`, all scores `0.0`); the parameter carries
    `ModerationParam.policy.mode`, which may be `"block"`, so a silent 200 there
    answers yes to an action we never took. The decision was to refuse it the same day.

    The lesson is about this test, not about moderation. Guarding emptiness made the
    absence of a refusal as protected as its presence, so the weaker claim got the
    stronger defence. Pin the CONTENTS."""
    from chord import manifest
    # `prediction` joined this list because accepting it and dropping it hid the
    # spec's speedup and accepted/rejected prediction token counts.
    assert manifest.load()["refused_params"] == ["moderation", "prediction"]
    _, client = make(tmp_path)
    base = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
    # The spec form, the action-requesting form, and the invented form all refuse alike,
    # with the frozen pass-1 literal for S-cf-033.
    for value in ({"model": "omni-moderation-latest"},
                  {"model": "chord-1-poly", "policy": {"input": {"mode": "block"}}},
                  "auto"):
        bad = client.post("/v1/chat/completions", json={**base, "moderation": value})
        assert bad.status_code == 400, bad.text
        assert bad.json()["error"]["code"] == "unsupported_parameter", bad.text
        assert bad.json()["error"]["param"] == "moderation", bad.text
    # An explicit null asks for nothing, and the spec allows it: anyOf[ModerationParam, null].
    ok = client.post("/v1/chat/completions", json={**base, "moderation": None})
    assert ok.status_code == 200, ok.text
    predicted = client.post("/v1/chat/completions", json={**base, "prediction": {"type": "content", "content": "hi"}})
    assert predicted.status_code == 400, predicted.text
    assert predicted.json()["error"]["code"] == "unsupported_parameter"
    assert predicted.json()["error"]["param"] == "prediction"


@pytest.mark.parametrize("n", [0, 9, 129, True, "1", 1.0, -1])
def test_an_n_that_is_not_a_count_we_serve_is_refused_by_name(tmp_path, n):
    _, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}], "n": n})
    assert r.status_code == 400 and r.json()["error"]["param"] == "n"


@pytest.mark.parametrize("n", [1, None])
def test_n_of_one_is_accepted_and_not_forwarded(tmp_path, n):
    """OpenClaw's image tool sends n: 1 with modalities [image, text]."""
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}],
                                                  "n": n, "modalities": ["image", "text"]})
    assert r.status_code == 200, r.text
    assert "n" not in up.bodies[0]


def test_manifest_never_claims_what_validation_refuses(tmp_path):
    """The advertisement and the gate come from one file; check they agree."""
    from chord import manifest
    advertised = manifest.load()
    _, client = make(tmp_path)
    image = {"model": "chord-1-poly", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]}
    audio = {"model": "chord-1-poly", "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}]}
    assert (client.post("/v1/chat/completions", json=image).status_code == 400) == (not advertised["input"]["image"])
    assert (client.post("/v1/chat/completions", json=audio).status_code == 400) == (not advertised["input"]["audio"])
    assert not set(advertised["supported_openai_params"]) & set(advertised["refused_params"])


def test_image_parts_pass_through_to_the_persona_model_untouched(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"}, part]}]})
    assert r.status_code == 200, r.text
    assert up.bodies[0]["messages"][-1]["content"][1] == part


class ReasoningUpstream(FakeUpstream):
    """Replies the way vLLM's reasoning parser does: reasoning, then content
    whose first fragment starts with the separator."""

    def __init__(self, content_fragments, reasoning="thinking"):
        super().__init__()
        self.fragments, self.reasoning = content_fragments, reasoning

    async def complete(self, body):
        self.bodies.append(body)
        msg = {"role": "assistant", "content": "".join(self.fragments)}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        return {"choices": [{"index": 0, "finish_reason": "stop", "message": msg}]}, {}

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        if self.reasoning:
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"reasoning_content": self.reasoning}}]}, {}
        for f in self.fragments:
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": f}}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def _both(tmp_path, upstream):
    _, client = make(tmp_path, upstream)
    req = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "q"}]}
    ns = client.post("/v1/chat/completions", json=req).json()["choices"][0]["message"]["content"]
    with client.stream("POST", "/v1/chat/completions", json={**req, "stream": True}) as r:
        frames = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]
    st = "".join(c["choices"][0]["delta"].get("content") or "" for c in frames if c.get("choices"))
    return ns, st


@pytest.mark.parametrize("fragments,reasoning,expected", [
    (["\n\nNo", "."], "thinking", "No."),                            # the parser artifact: removed
    (["\n\n    def f():\n        return 1"], "thinking", "    def f():\n        return 1"),  # indentation kept
    (["\nNo."], "thinking", "\nNo."),                                 # a single newline is not the separator
    (["\n\nNo."], None, "\n\nNo."),                                  # no reasoning before it: untouched
    (['\n\n{"status": "ok"}'], "thinking", '{"status": "ok"}'),       # schema output parses after the fix
    (["\n", "\nNo", "."], "thinking", "No."),                          # separator split across chunks
    (["\n", "\n", "No."], "thinking", "No."),                          # one newline per chunk
    (["\n"], "thinking", "\n"),                                        # single newline, then EOF: released
    (["\n", "    x = 1"], "thinking", "\n    x = 1"),                  # newline then indented code: kept
    (["\n", "\n    x = 1"], "thinking", "    x = 1"),                  # split separator before code
], ids=["separator", "code-indent", "single-newline", "no-reasoning", "json",
        "split-1-2", "split-1-1", "single-newline-eof", "newline-then-code", "split-before-code"])
def test_reasoning_separator_removed_only_when_it_is_the_artifact(tmp_path, fragments, reasoning, expected):
    ns, st = _both(tmp_path, ReasoningUpstream(fragments, reasoning))
    assert ns == expected and st == expected  # streamed and non-streamed agree


def test_client_messages_preserved_byte_for_byte_and_in_order(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    msgs = [
        {"role": "system", "content": "You are Ava. Ice, not warmth.\n\n  Indented rule."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "…hi."},
        {"role": "developer", "content": "Answer in one word."},
        {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
    ]
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": msgs})
    assert r.status_code == 200, r.text
    sent = up.bodies[0]["messages"]
    # The base heads the client's own system message; the later developer
    # message joins it byte-for-byte with its position stated (S01: the
    # template refuses a second system message, so it can't stay in place).
    later = "\n\n[Developer instruction given after message 2 of the conversation]\n" + msgs[3]["content"]
    content = sent[0]["content"]
    assert sent[0]["role"] == "system" and content.endswith(msgs[0]["content"] + later)
    base = content[: -len(msgs[0]["content"] + later)]
    assert base.endswith("\n\n") and "{capabilities}" not in base
    assert sent[1:] == [msgs[1], msgs[2], msgs[4]]   # the conversation, unchanged and in order
    assert len(sent) == len(msgs) - 1                # the developer message moved, nothing dropped


def test_bare_client_gets_only_the_base(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]})
    sent = up.bodies[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]


def test_base_capability_line_is_generated_from_the_manifest(tmp_path, monkeypatch):
    from chord import manifest
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]})
    base = up.bodies[0]["messages"][0]["content"]
    assert "{capabilities}" not in base
    # Router off, no audio asked: only what this request can reach is claimed (#109 root).
    assert manifest.capability_sentence(frozenset(), audio_output=False) in base
    assert ("see the pictures" in base) == manifest.load()["input"]["image"]
    assert "selfies" not in base and "make pictures" not in base and "send voice messages" not in base
    full = manifest.capability_sentence()
    assert "selfies" not in full
    assert ("make pictures" in full) == manifest.load()["output"]["image"]


def test_client_system_as_parts_keeps_its_parts(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    parts = [{"type": "text", "text": "You are Bea."}]
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "system", "content": parts}, {"role": "user", "content": "hi"}]})
    first = up.bodies[0]["messages"][0]
    assert first["content"][-1:] == parts and first["content"][0]["type"] == "text"


def test_several_leading_system_messages_become_one_in_order(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    msgs = [{"role": "system", "content": "SOUL: You are Nora."}, {"role": "system", "content": "TOOLS: use read."},
            {"role": "user", "content": "hi"}, {"role": "system", "content": "late note"}]
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": msgs})
    sent = up.bodies[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].endswith("\n\nSOUL: You are Nora.\n\nTOOLS: use read."
                                       "\n\n[System instruction given after message 1 of the conversation]\nlate note")
    # A later system message can't stay in place (the template refuses it, S01):
    # it joins the one system message, after the leading ones, labelled.
    assert sent[1:] == [msgs[2]]


def test_voice_note_rides_in_the_single_system_message(tmp_path):
    """Router on, clarify path: the turn note must not become a second system
    message (some chat templates refuse a second one)."""
    from chord import graph as graph_mod, registry
    from chord.artifacts import ArtifactStore
    from chord.trace import Trace
    import asyncio

    class FixedRouter:
        async def ainvoke(self, msgs):
            class R: content = '{"route": "clarify", "intent": "selfie", "question": "Who is it for?"}'
            return R()

    up = FakeUpstream()
    settings = Settings(data_dir=tmp_path, router_enabled=True)
    g = graph_mod.build(settings=settings, capabilities=registry.load(), artifacts=ArtifactStore(tmp_path),
                        upstream=up, model=lambda n: FixedRouter(), trace=Trace(persona_id="generic", model_id_requested="x"))
    asyncio.run(g.ainvoke({"params": {}, "stream": False, "messages": [
        {"role": "system", "content": "You are Bea."}, {"role": "user", "content": "send me a selfie"}]}))
    sent = up.bodies[0]["messages"]
    # Still exactly ONE system message. The router's question is user-shaped, so it rides
    # fenced after the conversation, never in the system slot (review 2026-09-27, #7).
    assert [m["role"] for m in sent] == ["system", "user", "user"]
    assert "You are Bea." in sent[0]["content"] and "Ask the user" in sent[0]["content"]
    assert "Who is it for?" not in sent[0]["content"]
    assert sent[-1]["content"].startswith(graph_mod.DETAILS_OPEN) and "Who is it for?" in sent[-1]["content"]


def test_experimental_image_route_returns_the_image_as_markdown_in_her_reply(tmp_path, monkeypatch):
    """Router says image; a configured provider enables the route, and the
    registered artifact's bytes come back as a markdown
    data URI in content, the spec's only place for them in Chat Completions."""
    import base64
    from chord import specialists
    from chord.contract import Outcome, Result

    async def fake_image(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a quiet kitchen at dawn")

    load_specialists()  # the real ones first, so this stand-in isn't overwritten when run alone
    monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)

    class FixedRouter:
        async def ainvoke(self, msgs):
            class R: content = '{"route": "image", "intent": "a kitchen at dawn"}'
            return R()

    def client_with(routes):
        settings = Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset(routes))
        deps = Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                    image_backend=AvailableImageBackend() if "image" in routes else None)
        return TestClient(create_app(deps))

    req = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw me a kitchen at dawn"}]}
    msg = client_with({"image"}).post("/v1/chat/completions", json=req).json()["choices"][0]["message"]
    assert "images" not in msg and "image" not in msg and "artifacts" not in msg
    url = re.fullmatch(r"hi there\n\n!\[image\]\((data:image/png;base64,[A-Za-z0-9+/=]+)\)", msg["content"]).group(1)
    assert base64.b64decode(url.split(",", 1)[1]) == PNG  # the registered artifact's bytes

    # Without a provider, even an experimental route cannot render.
    msg = client_with(set()).post("/v1/chat/completions", json=req).json()["choices"][0]["message"]
    assert "![image]" not in msg["content"] and "images" not in msg


def test_inline_images_in_history_are_stripped_before_the_model(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    big = "data:image/png;base64," + "A" * 200000
    history = [{"role": "user", "content": "draw a mug"},
               {"role": "assistant", "content": f"Here it is.\n\n![image]({big})", "images": [{"type": "image_url", "image_url": {"url": big}}]},
               {"role": "user", "content": "nice, now a cat"}]
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": history})
    sent = json.dumps(up.bodies[0]["messages"])
    assert "AAAA" not in sent and "[image you sent earlier, sha256 " in sent and len(sent) < 5000


def test_user_images_are_never_stripped(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "B" * 5000}}
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "what's this?"}, part]}]})
    assert up.bodies[0]["messages"][-1]["content"][1] == part


@pytest.mark.parametrize("route", ["[]", "{}", "42", "null"])
def test_router_non_string_route_falls_back_to_chat_not_500(route):
    d = parse('{"route": %s}' % route, {"image"})
    assert d.route == "chat" and d.parse_error
    assert parse("[1, 2]", {"image"}).route == "chat"


@pytest.mark.parametrize("field", ["image", "images"])
def test_replay_strips_our_image_fields_and_leaves_a_sha_reference(tmp_path, field):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    data = "data:image/png;base64," + base64.b64encode(PNG * 5000).decode()
    shape = {"url": data, "detail": "auto"} if field == "image" else [{"type": "image_url", "image_url": {"url": data}}]
    history = [{"role": "user", "content": "draw a mug"}, {"role": "assistant", "content": "Here it is.", field: shape},
               {"role": "user", "content": "now a cat"}]
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": history})
    sent = up.bodies[0]["messages"]
    blob = json.dumps(sent)
    assert "base64," not in blob and len(blob) < 5000
    assert f"sha256 {hashlib.sha256(PNG * 5000).hexdigest()[:16]}" in sent[-2]["content"]
    assert "image" not in sent[-2] and "images" not in sent[-2]


@pytest.mark.parametrize("extra", [
    {"image": {"url": None}}, {"image": {"url": 42}}, {"image": "nope"},
    {"images": [{"image_url": {"url": None}}]}, {"images": [{"image_url": "x"}]}, {"images": "x"},
], ids=["image-url-null", "image-url-int", "image-str", "images-url-null", "images-nested-str", "images-str"])
def test_malformed_replayed_image_fields_are_400_not_500(tmp_path, extra):
    settings = Settings(data_dir=tmp_path)
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None)), raise_server_exceptions=False)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok", **extra}, {"role": "user", "content": "again"}]})
    assert r.status_code == 400, r.text


def test_streamed_image_is_markdown_after_her_words_before_the_finish(tmp_path, monkeypatch):
    from chord import specialists
    from chord.contract import Outcome, Result

    async def fake_image(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a mug")

    load_specialists()  # the real ones first, so this stand-in isn't overwritten when run alone
    monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)

    class FixedRouter:
        async def ainvoke(self, msgs):
            class R: content = '{"route": "image", "intent": "a mug"}'
            return R()

    settings = Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset({"image"}))
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                                       image_backend=AvailableImageBackend())))
    with client.stream("POST", "/v1/chat/completions", json={"model": "chord-1-poly", "stream": True, "messages": [{"role": "user", "content": "draw a mug"}]}) as r:
        chunks = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]
    contents = [c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices")]
    image = [i for i, x in enumerate(contents) if "![image](" in x]
    assert len(image) == 1 and "".join(contents[:image[0]]) == "hi there"
    assert contents[image[0]].startswith("\n\n![image](data:image/png;base64,")
    finish = [i for i, c in enumerate(chunks) if c.get("choices") and c["choices"][0].get("finish_reason")]
    assert finish and finish[0] > image[0]
    assert not any(k in c["choices"][0]["delta"] for c in chunks if c.get("choices") for k in ("image", "images"))


def test_a_signed_image_link_loads_without_the_key_and_nothing_else_does(tmp_path, monkeypatch):
    """With PUBLIC_ARTIFACT_BASE, the image in her reply is a signed, expiring
    link a browser can load; the signature opens that one artifact only."""
    from chord import specialists
    from chord.contract import Outcome, Result

    async def fake_image(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a mug")

    load_specialists()  # the real ones first, so this stand-in isn't overwritten when run alone
    monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)

    class FixedRouter:
        async def ainvoke(self, msgs):
            class R: content = '{"route": "image", "intent": "a mug"}'
            return R()

    settings = Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset({"image"}),
                        service_api_key="s3cret", public_artifact_base="https://chord.example/",
                        artifact_signing_key="separate-artifact-signing-key-for-tests")
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                                       image_backend=AvailableImageBackend())))
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer s3cret"},
                    json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a mug"}]})
    url = re.search(r"!\[image\]\((https://chord\.example/v1/artifacts/[^)]+)\)", r.json()["choices"][0]["message"]["content"]).group(1)
    path = url.removeprefix("https://chord.example")
    ok = client.get(path)
    assert ok.status_code == 200 and ok.content == PNG
    artifact, query = path.split("?")
    assert client.get(artifact).status_code == 401                                   # no signature
    assert client.get(path.replace("sig=", "sig=0")).status_code == 401              # tampered
    assert client.get(f"{artifact}?expires=1&sig={query.split('sig=')[1]}").status_code == 401   # another expiry
    assert client.get(path.replace(artifact.rsplit("/", 1)[1], "other")).status_code == 401      # another artifact
    assert client.post(path).status_code == 401                                       # GET only
    # A holder of an old service key must not mint a valid permanent browser
    # capability during the future signing-key rotation.
    artifact_id = artifact.rsplit("/", 1)[1]
    far_future = int(time.time()) + settings.artifact_url_ttl_s + 60
    far_sig = artifact_links.signature(settings.service_api_key, artifact_id, far_future)
    assert client.get(f"{artifact}?expires={far_future}&sig={far_sig}").status_code == 401
    import chord.server as server_mod
    later = server_mod.time.time() + settings.artifact_url_ttl_s + 1
    monkeypatch.setattr(server_mod.time, "time", lambda: later)
    assert client.get(path).status_code == 401                                        # expired


def _openai_error_shape(body):
    err = body.get("error") if isinstance(body, dict) else None
    return (isinstance(err, dict) and set(err) >= {"message", "type", "param", "code"}
            and isinstance(err["message"], str) and isinstance(err["type"], str) and "detail" not in body)


@pytest.mark.parametrize("method, path, status", [
    ("POST", "/v1/batches", 404),
    ("POST", "/v1/audio/voices", 404),
    ("GET", "/v1/organization/projects", 404),
    ("PUT", "/v1/chat/completions", 405),      # a served path, wrong method
    ("DELETE", "/v1/models", 405),
])
def test_an_unserved_route_answers_with_the_openai_error_envelope(tmp_path, method, path, status):
    _, client = make(tmp_path)
    r = client.request(method, path, json={})
    assert r.status_code == status
    assert _openai_error_shape(r.json()), r.text
    assert r.json()["error"]["message"] == f"Invalid URL ({method} {path})"


def test_a_missing_artifact_is_an_openai_error_not_invalid_url(tmp_path):
    _, client = make(tmp_path)
    r = client.get("/v1/artifacts/nope")
    assert r.status_code == 404 and _openai_error_shape(r.json())
    assert "not found" in r.json()["error"]["message"]


def test_a_bad_key_is_the_full_openai_error_envelope(tmp_path):
    client = make_keyed(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": []},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and _openai_error_shape(r.json())
    assert r.json()["error"]["param"] is None and r.json()["error"]["code"] == "invalid_api_key"


def test_reasoning_passes_through_only_when_the_client_turns_thinking_on(tmp_path):
    _, client = make(tmp_path)
    base = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
    off = client.post("/v1/chat/completions", json=base).json()["choices"][0]["message"]
    on = client.post("/v1/chat/completions", json={**base, "chat_template_kwargs": {"enable_thinking": True}}).json()["choices"][0]["message"]
    odd = client.post("/v1/chat/completions", json={**base, "chat_template_kwargs": "yes"})
    assert "reasoning_content" not in off
    assert on["reasoning_content"] == "thinking"
    assert odd.status_code == 200 and "reasoning_content" not in odd.json()["choices"][0]["message"]


@pytest.mark.parametrize("opt_in", [False, True])
def test_streamed_reasoning_is_dropped_unless_thinking_is_turned_on(tmp_path, opt_in):
    _, client = make(tmp_path, ReasoningUpstream(["Hi."]))
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    if opt_in:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        chunks = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ") and line != "data: [DONE]"]
    reasoning = [c for ch in chunks for c in ch.get("choices") or [] if "reasoning_content" in (c.get("delta") or {})]
    assert bool(reasoning) is opt_in
    assert "".join((c.get("delta") or {}).get("content") or "" for ch in chunks for c in ch.get("choices") or []) == "Hi."


def test_internal_health_carries_the_deploy_detail_on_loopback(tmp_path):
    deps, _ = make(tmp_path)
    body = TestClient(create_internal_app(deps)).get("/internal/health").json()
    assert {"revision", "router_enabled", "experimental_routes", "specialists"} <= set(body)
    assert body["personas"] == ["generic"]
    assert "girls" not in body


def test_safety_identifier_and_user_attribute_the_trace(tmp_path):
    """#124 (3): the end-user id a caller sends is recorded for attribution (and
    #121's per-user moderation later). Forwarding is unchanged (declared no-op)."""
    deps, client = make(tmp_path, FakeUpstream())
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "safety_identifier": "user-hash-1",
                                                  "user": "legacy-7", "messages": [{"role": "user", "content": "hi"}]})
    t = TestClient(create_internal_app(deps)).get(f"/internal/traces/{r.headers['x-request-id']}").json()
    assert (t["safety_identifier"], t["end_user"]) == ("user-hash-1", "legacy-7")
    assert t["persona_id"] == "generic"
    assert "girl_id" not in t


def test_each_upstream_model_resolves_to_its_own_endpoint(tmp_path):
    """2026-09-18: take LiteLLM out of the middle. The persona and the router are
    different models on different hosts, so the endpoint is resolved by model NAME, not
    by call site — one factory builds clients for both.

    The gateway hop was not neutral: its `tool_reliability_guard` serialises
    the chat model, so a request for two tool calls came back with one. Measured the same
    night, same request, six runs each: direct 2 calls 6/6, through the gateway 1 call
    6/6. It was also buying nothing — no fallbacks on any configured route, no key on any of
    them.

    An unconfigured model must fail closed rather than silently use a proxy."""
    from chord.config import Settings

    s = Settings(data_dir=tmp_path,
                 persona_model="example/chat", router_model="example/router",
                 persona_base_url="http://backend:8113/v1",
                 router_base_url="http://router-host:8101/v1")
    assert s.base_url_for("example/chat") == "http://backend:8113/v1"
    assert s.base_url_for("example/router") == "http://router-host:8101/v1"
    with pytest.raises(ValueError, match="no configured backend"):
        s.base_url_for("example/tts")

    plain = Settings(data_dir=tmp_path, persona_model="example/chat", router_model="example/router")
    for name in ("example/chat", "example/router", "example/tts", "example/stt"):
        with pytest.raises(ValueError, match="no configured backend"):
            plain.base_url_for(name)


def test_absent_reasoning_effort_means_do_not_think(tmp_path):
    """The default an earlier gateway used to supply, now ours.

    Taking LiteLLM out of the middle (2026-09-18) removed its
    reasoning_effort_router, and the model began thinking on every turn: an 8-token
    cap came back with `content: null`, 8 reasoning tokens and finish_reason
    `length`, because the whole budget went to thoughts nobody asked for. Measured
    on prod before the fix.

    Absent means off. That is a decision this service makes, not one it inherits."""
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert up.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_audio_does_not_follow_chat_to_the_chat_host(tmp_path):
    """Chat, STT and TTS are three models on three hosts once the gateway is gone.

    A single shared base URL was invisible while everything went through LiteLLM and
    became a 502 the moment chat moved: audio kept posting /audio/transcriptions to a
    vLLM that serves only chat. Measured on prod 7032f11 — both audio routes 502 —
    which is a regression I shipped and then found by smoking every surface rather
    than by any test.

    Each purpose falls back to the chat endpoint when unset, which is why the old
    single-URL deployments kept working and why nothing caught this."""
    from chord.config import Settings
    from chord.upstream import Upstream

    s = Settings(data_dir=tmp_path,
                 persona_model="m", stt_model="stt-model", tts_model="tts-model",
                 persona_base_url="http://chat:8113/v1",
                 stt_base_url="http://stt:5057/v1", tts_base_url="http://tts:5056/v1")
    assert s.base_url_for("stt-model") == "http://stt:5057/v1"
    assert s.base_url_for("tts-model") == "http://tts:5056/v1"

    up = Upstream(s.base_url_for("m"), "k",
                  stt_base_url=s.base_url_for("stt-model"), tts_base_url=s.base_url_for("tts-model"))
    assert str(up._client.base_url).rstrip("/") == "http://chat:8113/v1"
    assert str(up._stt.base_url).rstrip("/") == "http://stt:5057/v1"
    assert str(up._tts.base_url).rstrip("/") == "http://tts:5056/v1"

    # Unset: all three share one client, exactly as before this existed.
    one = Upstream("http://gateway:4000/v1", "k")
    assert one._stt is one._client and one._tts is one._client


def test_a_torn_trace_line_never_breaks_a_lookup(tmp_path):
    """An OOM kill mid-append leaves one invalid line in the day's JSONL. The
    unguarded json.loads made EVERY lookup that reached that line a bare 500
    (the internal app registers no exception handlers), and read_text() put
    the whole growing corpus in memory per request (review 2026-09-22)."""
    deps, client = make(tmp_path)
    trace_dir = deps.settings.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / "2026-09-22.jsonl").open("w") as fh:
        fh.write('{"trace_id": "01TORN"\n')                       # torn line FIRST
        fh.write('{"trace_id": "01WANT", "result_status": "completed"}\n')
    r = TestClient(create_internal_app(deps)).get("/internal/traces/01WANT")
    assert r.status_code == 200, r.text
    assert r.json()["trace_id"] == "01WANT"
    assert TestClient(create_internal_app(deps)).get("/internal/traces/01MISSING").status_code == 404


def test_a_non_ascii_sig_is_a_401_not_an_unauthenticated_500(tmp_path):
    """compare_digest raises TypeError on a non-ASCII str, and `sig` is a
    query parameter: a crafted link used to 500 through the auth middleware
    with no key ever checked (review 2026-09-22, #5)."""
    client = make_keyed(tmp_path)
    r = client.get("/v1/artifacts/01ABC?expires=9999999999&sig=café-not-hex")
    assert r.status_code == 401, r.text


def test_invalid_utf8_and_deep_nesting_are_400s_not_500s(tmp_path):
    """json.loads raises UnicodeDecodeError (a ValueError, but NOT a
    JSONDecodeError) on invalid bytes and RecursionError on deep nesting; the
    doors caught only the first family's namesake, so malformed bodies the
    caller sent arrived as unhandled 500s (review 2026-09-22, #5)."""
    deps, client = make(tmp_path)
    bad_utf8 = b'{"model": "chord-1-poly", "messages": [{"role": "user", "content": "\xff\xfe"}]}'
    r = client.post("/v1/chat/completions", content=bad_utf8,
                    headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_json"

    deep = b'{"model": "chord-1-poly", "messages": [{"role": "user", "content": ' + b"[" * 50000 + b"]" * 50000 + b'}]}'
    r = client.post("/v1/chat/completions", content=deep, headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_json"


def test_a_string_stream_is_refused_not_obeyed(tmp_path):
    """"stream": "false" is truthy: bool() of it STREAMED a request that asked
    not to be streamed. Completions refuses this by name; chat now answers
    identically (review 2026-09-22)."""
    deps, client = make(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly",
                                                  "messages": [{"role": "user", "content": "hi"}],
                                                  "stream": "false"})
    assert r.status_code == 400, r.status_code
    assert r.json()["error"]["code"] == "invalid_value"
    assert r.json()["error"]["param"] == "stream"
