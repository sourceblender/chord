"""S12 (red team pass 1, 2026-09-15; S-cf-034, S-cf-080). With
stream_options.include_usage the pinned spec says the usage chunk's "choices
field will always be an empty array", and "all other chunks will also include
a usage field, but with a null value". Live, the usage chunk kept LiteLLM's
pseudo-choice and 1 of 18 chunks had a usage key."""
import json
import sys
from pathlib import Path

import openai
import pytest

from test_search import ASK, UsageCitedUpstream, app as search_app
from test_skeleton import FakeUpstream, make

sys.path.insert(0, str(Path(__file__).parents[1] / "qa"))
from conformance.schema import validate_stream

SHAPES = {"openai": [], "litellm": [{"index": 0, "delta": {}}]}


def frames(client, **extra):
    body = {"model": "chord-1-poly", "stream": True, "messages": [{"role": "user", "content": "hi"}], **extra}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        raw = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert raw[-1] == "[DONE]"
    return [json.loads(f) for f in raw[:-1]]


class MixedFinish(FakeUpstream):
    """One chunk carries the last tokens, their logprobs, the finish, and usage."""

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop", "logprobs": {"content": [{"token": "hi"}]}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}, {}


def test_a_finish_chunk_does_not_repeat_logprobs_or_leak_unrequested_usage(tmp_path):
    _, client = make(tmp_path, MixedFinish())
    chunks = frames(client)
    carrying = [c for c in chunks if c.get("choices") and c["choices"][0].get("logprobs")]
    assert len(carrying) == 1
    assert all(c.get("usage") is None for c in chunks)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_with_include_usage_the_usage_chunk_is_last_with_empty_choices_and_every_other_chunk_has_null(tmp_path, shape):
    _, client = make(tmp_path, FakeUpstream(usage_choices=SHAPES[shape]))
    chunks = frames(client, stream_options={"include_usage": True})
    *rest, last = chunks
    assert last["choices"] == [] and last["usage"]["total_tokens"] == 7
    assert rest and all("usage" in c and c["usage"] is None for c in rest)
    assert any(c["choices"] and c["choices"][0].get("finish_reason") == "stop" for c in rest)   # finish before usage


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("options", [None, {}, {"include_usage": False}])
def test_without_include_usage_no_chunk_has_a_usage_key(tmp_path, options, shape):
    """The upstream still sends its usage chunk here (the fake always does):
    unrequested, it never reaches the client, but the trace keeps it (#172)."""
    extra = {} if options is None else {"stream_options": options}
    deps, client = make(tmp_path, FakeUpstream(usage_choices=SHAPES[shape]))
    chunks = frames(client, **extra)
    assert chunks and not any("usage" in c for c in chunks)
    from test_progress import last_trace
    assert last_trace(deps.settings)["stream_usage"]["total_tokens"] == 7


def test_the_search_path_conforms_too(tmp_path, monkeypatch):
    """The live LiteLLM shape under a held, cited answer."""
    from chord.specialists import search as S
    hits = [S.Hit("Dreamforce", "https://www.salesforceben.com/df26", "Usher and Gwen Stefani headline Dreamfest.")]
    c, up, settings, _ = search_app(tmp_path, monkeypatch, hits, upstream=UsageCitedUpstream())
    with c.stream("POST", "/v1/chat/completions", json={"model": "chord-1-poly", "stream": True,
                                                      "stream_options": {"include_usage": True},
                                                      "messages": [{"role": "user", "content": ASK}]}) as r:
        raw = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    *rest, last = [json.loads(f) for f in raw[:-1]]
    assert last["choices"] == [] and last["usage"]["total_tokens"] == 30
    assert all(c["usage"] is None for c in rest)


class LiveUsageUpstream(FakeUpstream):
    """S-cf-034's captured usage chunk: LiteLLM's pseudo-choice, full usage."""

    async def stream(self, body):
        async for chunk, dep in super().stream(body):
            if chunk and chunk.get("usage"):
                chunk = {**chunk, "choices": SHAPES["litellm"], "usage": {
                    "completion_tokens": 2, "prompt_tokens": 119, "total_tokens": 121,
                    "completion_tokens_details": {"reasoning_tokens": 0}}}
            yield chunk, dep


@pytest.mark.parametrize("include", [True, False])
def test_the_stream_passes_the_pinned_schema(tmp_path, include):
    """A regression guard, not the falsifier: the pinned schema neither forbids
    the pseudo-choice nor requires usage: null, so this passes on main too."""
    _, client = make(tmp_path, LiveUsageUpstream())
    body = {"model": "chord-1-poly", "stream": True, "stream_options": {"include_usage": include},
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        text = "".join(f"{line}\n" for line in r.iter_lines())
    verdicts = list(validate_stream(text))
    assert verdicts and all(v["verdict"] == "pass" for v in verdicts), verdicts


def test_official_sdk_reads_usage_from_the_last_chunk(tmp_path):
    _, client = make(tmp_path, FakeUpstream(usage_choices=SHAPES["litellm"]))
    sdk = openai.OpenAI(api_key="test", base_url="http://testserver/v1", http_client=client, max_retries=0)
    chunks = list(sdk.chat.completions.create(model="chord-1-poly", messages=[{"role": "user", "content": "hi"}],
                                              stream=True, stream_options={"include_usage": True}))
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 7
    assert all(ch.usage is None for ch in chunks[:-1])
