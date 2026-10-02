"""A router that fails or stalls must not hold up, or break, a chat turn.

2026-09-11: the router backend went down. Every chat turn hung until
LiteLLM gave up at 180 s, and a router exception was a bare HTTP 500.
"""
import asyncio
import json
import time
from pathlib import Path

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from chord import graph as graph_mod
from chord import registry, specialists
from chord.artifacts import ArtifactStore
from chord.config import Settings
from chord.server import Deps, create_app, load_specialists
from chord.trace import Trace

from test_skeleton import FakeUpstream

load_specialists()  # so a test's stand-in isn't overwritten by the first Deps()


class RaisingRouter:
    async def ainvoke(self, msgs):
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://router-host/v1/chat/completions"))


class StalledRouter:
    async def ainvoke(self, msgs):
        await asyncio.Event().wait()  # never answers


def last_trace(settings):
    lines = [l for p in sorted(Path(settings.trace_dir).glob("*.jsonl")) for l in p.read_text().splitlines()]
    return json.loads(lines[-1])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("router,expected", [(RaisingRouter, "failed (APITimeoutError)"), (StalledRouter, "timed out")])
def test_router_failure_falls_back_to_chat_within_our_deadline(tmp_path, monkeypatch, stream, router, expected):
    submitted = []

    async def image(job, ctx):
        submitted.append(job)
        raise AssertionError("no specialist may start when the router failed")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        experimental_routes=frozenset({"image"}), router_timeout_s=0.3)
    up = FakeUpstream()
    client = TestClient(create_app(Deps(settings, upstream=up, model=lambda n: router())))
    t0 = time.monotonic()
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
                                                  "messages": [{"role": "user", "content": "draw me a mug"}]})
    elapsed = time.monotonic() - t0

    assert r.status_code == 200, r.text
    if stream:
        chunks = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
        said = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in chunks)
    else:
        said = r.json()["choices"][0]["message"]["content"]
    assert said == "hi there"                         # chat reached: her voice answered
    assert len(up.bodies) == 1 and not submitted      # one voice call, no specialist
    assert elapsed < 5                                # our deadline, not the gateway's
    t = last_trace(settings)
    assert t["route_decision"] == "chat" and t["router_call_error"] == f"router call {expected}"
    assert t["trace_id"] in r.text                    # the trace is the one for this turn


def test_cancellation_during_routing_propagates_and_never_starts_a_voice_call(tmp_path):
    """The fallback catches failures, not cancellation: a turn cancelled while
    the router is thinking must end, not fall through to a fresh chat call."""
    up = FakeUpstream()
    settings = Settings(data_dir=tmp_path, router_enabled=True, router_timeout_s=30)
    g = graph_mod.build(settings=settings, capabilities=registry.load(), artifacts=ArtifactStore(tmp_path),
                        upstream=up, model=lambda n: StalledRouter(), trace=Trace(persona_id="generic", model_id_requested="x"))

    async def run():
        task = asyncio.create_task(g.ainvoke({"params": {}, "stream": False, "messages": [{"role": "user", "content": "hi"}]}))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)

    asyncio.run(run())
    assert up.bodies == []


def test_forced_search_missing_from_registry_becomes_unavailable_without_specialist(tmp_path, monkeypatch):
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("a missing search capability must never start a specialist")

    monkeypatch.setitem(specialists.SPECIALISTS, "search", forbidden)
    trace = Trace(persona_id="generic", model_id_requested="chord-1-poly")
    up = FakeUpstream()
    graph = graph_mod.build(
        settings=Settings(data_dir=tmp_path), capabilities={}, artifacts=ArtifactStore(tmp_path),
        upstream=up, model=lambda _name: None, trace=trace,
    )
    result = asyncio.run(graph.ainvoke({
        "params": {"web_search_options": {}}, "stream": False,
        "messages": [{"role": "user", "content": "Find the weather"}],
    }))
    assert result["unavailable"] == "search"
    assert trace.fields["route_not_registered"] == "search"
    assert len(up.bodies) == 1


def test_the_router_prompt_keeps_both_audio_overread_rules():
    """Two measured over-read classes, each pinned by the rule that closed it:
    a question about a sent clip (S-cf-084, 2026-09-17: audio 10/10 before,
    chat 10/10 after) and "say X" (#322, 2026-09-23: audio 10/10 before, with
    8/10 replies claiming the caller asked for a voice message). The prompt is
    the router's whole program; a rule silently deleted is the defect
    returned. This pins the rule's presence -- the behavioral N>=10 evidence
    lives in the dated evidence directories, per each issue's acceptance."""
    from chord.registry import PROMPTS_DIR

    text = (PROMPTS_DIR / "router.md").read_text()
    assert "A question about something the user sent" in text        # the 09-17 rule
    assert "asks for those words as the assistant's reply" in text   # the #322 rule
    assert "never asks for audio" in text
