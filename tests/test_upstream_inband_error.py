"""An error the backend reports INSIDE a stream is a failure, never a reply.

vLLM and LiteLLM report a failure that happens after the 200 as an SSE frame,
`data: {"error": {...}}`, and then usually `[DONE]`. The chat client parsed that
frame like any chunk. It has no `choices`, so the wire builder kept an empty
chunk, dropped the error, and the stream then closed with a manufactured
`finish_reason: "stop"`; Responses said `response.completed`. The client got
truncated text dressed as a finished answer (a failure is never
dressed up as a reply). Review 2026-09-27.

These drive the REAL `Upstream` over an httpx MockTransport, because every
FakeUpstream yields parsed chunks and so never crosses the parser where the
error frame arrives.
"""
import json

import httpx
import pytest

from chord.upstream import Upstream, UpstreamError
from test_skeleton import make

BASE = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
ERROR_FRAME = {"error": {"message": "CUDA out of memory", "type": "InternalServerError", "code": 500}}


def _sse(*frames) -> bytes:
    out = []
    for f in frames:
        out.append("data: " + (f if isinstance(f, str) else json.dumps(f)) + "\n\n")
    return "".join(out).encode()


def _chunk(text: str) -> dict:
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def _completion_chunk(text: str) -> dict:
    return {"id": "c1", "object": "text_completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": None}]}


def _upstream(stream_body: bytes, text_body: bytes | None = None) -> Upstream:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/completions") and not request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, content=text_body or stream_body,
                                  headers={"content-type": "text/event-stream"})
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    up = Upstream("http://up/v1", "k")
    up._client = httpx.AsyncClient(base_url="http://up/v1", transport=httpx.MockTransport(handler))
    return up


def _data_frames(text: str) -> list[str]:
    return [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]


MID_STREAM = _sse(_chunk("starting"), ERROR_FRAME, "[DONE]")


async def _drain(gen):
    return [item async for item in gen]


@pytest.mark.parametrize("method", ["stream", "stream_text"])
def test_the_client_raises_on_an_in_band_error_frame(method):
    import asyncio
    first = _completion_chunk("starting") if method == "stream_text" else _chunk("starting")
    up = _upstream(_sse(first, ERROR_FRAME, "[DONE]"))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(_drain(getattr(up, method)({"model": "m"})))
    assert exc.value.status == 500


def test_a_non_http_error_code_is_still_a_server_failure():
    import asyncio
    up = _upstream(_sse(_chunk("a"), {"error": {"message": "x", "code": "boom"}}))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(_drain(up.stream({"model": "m"})))
    assert exc.value.status == 502


def test_an_error_beside_partial_choices_still_fails():
    """An explicit error wins even when the frame also carries choices (#346)."""
    import asyncio
    both = {**_chunk("partial"), **ERROR_FRAME}
    up = _upstream(_sse(_chunk("a"), both, "[DONE]"))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(_drain(up.stream({"model": "m"})))
    assert exc.value.status == 500


@pytest.mark.parametrize("empty", [{}, ""])
def test_an_empty_error_is_still_an_error(empty):
    """Falsey is not absent: an empty error object or string still fails (#346)."""
    import asyncio
    up = _upstream(_sse(_chunk("a"), {**_chunk("b"), "error": empty}, "[DONE]"))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(_drain(up.stream({"model": "m"})))
    assert exc.value.status == 502


def test_a_null_error_field_is_an_ordinary_chunk():
    import asyncio
    ok = {**_chunk("fine"), "error": None}
    up = _upstream(_sse(ok, "[DONE]"))
    frames = asyncio.run(_drain(up.stream({"model": "m"})))
    assert frames[-1][0]["choices"][0]["delta"]["content"] == "fine"


def test_chat_stream_with_error_beside_choices_never_ends_in_stop(tmp_path):
    both = {**_chunk(" more"), **ERROR_FRAME}
    _, client = make(tmp_path, _upstream(_sse(_chunk("starting"), both, "[DONE]")))
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    frames = _data_frames(r.text)
    assert any('"stream_failed"' in f for f in frames), frames
    finishes = [c.get("finish_reason") for f in frames if f.startswith("{")
                for c in json.loads(f).get("choices") or []]
    assert "stop" not in finishes


def test_chat_stream_reports_the_failure_and_never_a_stop_finish(tmp_path):
    _, client = make(tmp_path, _upstream(MID_STREAM))
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    frames = _data_frames(r.text)
    assert "starting" in r.text, "the text already sent is not taken back"
    assert any('"stream_failed"' in f for f in frames), frames
    finishes = [c.get("finish_reason") for f in frames if f.startswith("{")
                for c in json.loads(f).get("choices") or []]
    assert "stop" not in finishes, "a failed stream must not end like a finished answer"
    assert "CUDA" not in r.text, "the backend's own error text is not relayed"


def test_responses_stream_does_not_say_completed(tmp_path):
    _, client = make(tmp_path, _upstream(MID_STREAM))
    r = client.post("/v1/responses", json={"model": "chord-1-poly", "input": "hi", "stream": True})
    events = [json.loads(f).get("type") for f in _data_frames(r.text) if f.startswith("{")]
    assert "response.completed" not in events, events
    assert events[-1] in {"response.failed", "error"}, events


def test_an_error_before_any_content_is_an_http_error_not_a_200(tmp_path):
    _, client = make(tmp_path, _upstream(_sse(ERROR_FRAME, "[DONE]")))
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert r.status_code >= 500, (r.status_code, r.text[:300])
    assert "CUDA" not in r.text
