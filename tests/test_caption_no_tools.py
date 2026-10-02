"""#147 (operator's Open WebUI test, 2026-09-14, trace 01M2H4STRRQP0T5RZYA6CH5F2W):
Open WebUI offered create_image, chord rendered, and the caption call forwarded
the client's tools, so her model called create_image on a turn that already
carried our picture. The failed call became "the image tool is acting up".

After our picture is delivered the caption offers no tools, and a tool call that
comes back anyway is stripped and recorded, never relayed or failed."""
import json
import re

import pytest
from fastapi.testclient import TestClient

from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.graph import CAPTION_FALLBACK, TOOL_PARAMS
from chord.server import Deps, create_app, load_specialists
from test_progress import FixedRouter, last_trace
from test_skeleton import AvailableImageBackend, PNG, FakeUpstream

load_specialists()

CREATE_IMAGE = {"type": "function", "function": {"name": "create_image", "parameters": {"type": "object"}}}
TIMESTAMP = {"type": "function", "function": {"name": "get_current_timestamp", "parameters": {"type": "object"}}}
ASK = {"role": "user", "content": "Alex in the grocery store, picking out peaches."}
# An earlier turn of her own agent loop: a tool call and its result, then a new ask.
HISTORY = [
    {"role": "user", "content": "What time is it?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_current_timestamp", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "2026-09-14T18:02:00Z"},
    {"role": "assistant", "content": "Just after six."},
]
CALL = {"id": "call_9", "type": "function", "function": {"name": "create_image", "arguments": '{"prompt": "peaches"}'}}
CAPTION = "Here she is."


class Voice(FakeUpstream):
    """Her model, calling create_image whether or not it was offered."""
    caption, call = CAPTION, CALL

    async def complete(self, body):
        data, dep = await super().complete(body)
        data["choices"][0]["message"] = {"role": "assistant", "content": self.caption or None,
                                         "tool_calls": [self.call], "function_call": None}
        data["choices"][0]["finish_reason"] = "tool_calls"
        return data, dep

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {"model-api-base": "http://10.9.8.7:8113/v1"}
        yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}, {}
        for char in self.caption:
            yield {"choices": [{"index": 0, "delta": {"content": char}}]}, {}
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": self.call["id"], "type": "function",
             "function": {"name": self.call["function"]["name"], "arguments": ""}}]}}]}, {}
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": self.call["function"]["arguments"]}}]}}]}, {}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, {}


def client(tmp_path, monkeypatch, status=Outcome.completed, router_enabled=True, voice=None):
    renders = []

    async def render(job, ctx):
        renders.append(job)
        artifacts = [ctx.artifacts.register(PNG, "image/png")] if status is Outcome.completed else []
        return Result(job_id=job.job_id, revision=job.revision, status=status, artifacts=artifacts)

    monkeypatch.setitem(specialists.SPECIALISTS, "image", render)
    settings = Settings(data_dir=tmp_path, router_enabled=router_enabled,
                        experimental_routes=frozenset({"image"}))
    up = voice or Voice()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda _: FixedRouter(),
                                      image_backend=AvailableImageBackend()))), up, settings, renders


DELIVERED_IMAGE = re.compile(r"\n\n!\[image\]\(data:image/png;base64,[A-Za-z0-9+/=]+\)")


def split_images(text):
    """Her words, and the images the service delivered after them as markdown."""
    return DELIVERED_IMAGE.sub("", text), DELIVERED_IMAGE.findall(text)


def post(c, stream, messages, **extra):
    """(text, images, tool calls, finish reasons) as the client received them."""
    body = {"model": "chord-1-poly", "stream": stream, "messages": messages, **extra}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    if not stream:
        choice = r.json()["choices"][0]
        message = choice["message"]
        calls = [message[k] for k in ("tool_calls", "function_call") if message.get(k)]
        text, images = split_images(message.get("content") or "")
        return text, images, calls, [choice["finish_reason"]]
    chunks = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")]
    choices = [c for chunk in chunks for c in chunk.get("choices") or []]
    deltas = [c.get("delta") or {} for c in choices]
    text, images = split_images("".join(d.get("content") or "" for d in deltas))
    calls = [d[k] for d in deltas for k in ("tool_calls", "function_call") if d.get(k)]
    return text, images, calls, [c["finish_reason"] for c in choices if c.get("finish_reason")]


CHOICES = {
    "offered": {},
    "auto": {"tool_choice": "auto", "parallel_tool_calls": True},
    "none": {"tool_choice": "none"},
}
# A forced call is the client's answer, so there is no picture of ours to caption
# (R1a, red team pass 1 S03, 2026-09-15). These rows rendered and stripped the
# forced call until then.
FORCED = {
    "required": {"tool_choice": "required"},
    "forced": {"tool_choice": {"type": "function", "function": {"name": "create_image"}}},
}


@pytest.mark.parametrize("choice", list(CHOICES))
@pytest.mark.parametrize("stream", [False, True])
def test_declared_tools_mean_no_picture_of_ours_to_caption(tmp_path, monkeypatch, stream, choice):
    """S04 (2026-09-16) removed #147's cause: a request that declares tools is
    never routed, so there is no render of ours for her tool to collide with."""
    c, up, settings, renders = client(tmp_path, monkeypatch)
    text, images, calls, finishes = post(c, stream, HISTORY + [ASK], tools=[TIMESTAMP, CREATE_IMAGE], **CHOICES[choice])
    assert renders == [] and not images
    # tool_choice "none" forwards no tools at all (S-cf-045: with them in the prompt the reply came back empty)
    assert up.bodies[-1].get("tools") == (None if choice == "none" else [TIMESTAMP, CREATE_IMAGE])
    assert last_trace(settings)["router"] == "skipped_client_tools"


@pytest.mark.parametrize("stream", [False, True])
def test_after_our_picture_a_call_invented_from_tool_history_is_stripped(tmp_path, monkeypatch, stream):
    """No tools declared, but her earlier loop is in the history: we render, and
    a tool call her caption invents anyway is stripped and recorded, never relayed."""
    c, up, settings, renders = client(tmp_path, monkeypatch)
    text, images, calls, finishes = post(c, stream, HISTORY + [ASK])
    assert len(renders) == 1 and images and text == CAPTION
    assert not TOOL_PARAMS & up.bodies[-1].keys()
    assert calls == [] and finishes == ["stop"]
    t = last_trace(settings)
    assert t["caption_tool_calls_stripped"] >= 1 and t["caption_tool_call_names"] == ["create_image"]


@pytest.mark.parametrize("history", [False, True], ids=["fresh", "tool-history"])
@pytest.mark.parametrize("choice", list(FORCED))
@pytest.mark.parametrize("stream", [False, True])
def test_a_forced_client_call_is_returned_not_rendered_over(tmp_path, monkeypatch, stream, choice, history):
    c, up, settings, renders = client(tmp_path, monkeypatch)
    messages = (HISTORY if history else []) + [ASK]
    text, images, calls, finishes = post(c, stream, messages, tools=[TIMESTAMP, CREATE_IMAGE], **FORCED[choice])
    assert renders == []                                  # the client's call, not our render
    assert up.bodies[-1]["tools"] == [TIMESTAMP, CREATE_IMAGE]
    assert up.bodies[-1]["tool_choice"] == FORCED[choice]["tool_choice"]
    assert calls and not images and finishes == ["tool_calls"]
    t = last_trace(settings)
    assert t["router"] == "skipped_client_constraint" and "caption_tool_calls_stripped" not in t

# The live shape (the chat backend, 1 of 18, 2026-09-14): with tool history and no
# tools offered, the whole reply was an invented generate_image call.
INVENTED = {"id": "chatcmpl-tool-924f", "type": "function", "function": {"name": "generate_image", "arguments": "{}"}}


@pytest.mark.parametrize("caption", ["", "\n\n"], ids=["empty", "whitespace"])
@pytest.mark.parametrize("stream", [False, True])
def test_a_tool_only_caption_becomes_a_true_line_not_an_empty_stop(tmp_path, monkeypatch, stream, caption):
    voice = Voice()
    voice.caption, voice.call = caption, INVENTED
    c, up, settings, renders = client(tmp_path, monkeypatch, voice=voice)
    text, images, calls, finishes = post(c, stream, HISTORY + [ASK])
    assert images and calls == [] and finishes == ["stop"]
    assert text.strip() == CAPTION_FALLBACK
    t = last_trace(settings)
    assert t["caption_empty_fallback"] is True and t["caption_tool_call_names"] == ["generate_image"]


@pytest.mark.parametrize("stream", [False, True])
def test_plain_chat_still_carries_her_tools_both_ways(tmp_path, monkeypatch, stream):
    """The pass-through her agent loop depends on is unchanged when we render nothing."""
    c, up, settings, renders = client(tmp_path, monkeypatch, router_enabled=False)
    text, images, calls, finishes = post(c, stream, [ASK], tools=[TIMESTAMP, CREATE_IMAGE], tool_choice="auto")
    assert renders == []
    assert up.bodies[-1]["tools"] == [TIMESTAMP, CREATE_IMAGE] and up.bodies[-1]["tool_choice"] == "auto"
    assert calls and finishes == ["tool_calls"]
    t = last_trace(settings)
    assert "caption_client_tools_removed" not in t and "caption_tool_calls_stripped" not in t


def test_a_tool_only_stream_with_no_finish_chunk_still_gets_the_line(tmp_path, monkeypatch):
    class Cut(Voice):
        caption, call = "", INVENTED

        async def stream(self, body):
            async for chunk, dep in super().stream(body):
                if chunk and any(c.get("finish_reason") for c in chunk.get("choices") or []):
                    return
                yield chunk, dep
    c, up, settings, renders = client(tmp_path, monkeypatch, voice=Cut())
    text, images, calls, finishes = post(c, True, HISTORY + [ASK])
    assert images and calls == [] and text == CAPTION_FALLBACK and finishes == ["stop"]
    assert last_trace(settings)["caption_empty_fallback"] is True
