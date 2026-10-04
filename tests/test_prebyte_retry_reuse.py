"""Batch 3 (the entry-edge design): the pre-byte retry re-draws the PERSONA, not the turn.

The retry used to re-invoke the whole graph on the original state_in: the router
re-decided the turn -- and the router is stochastic, so a search turn could come
back `chat` and silently drop its own citations, a semantic drift the wire never
shows -- and a specialist turn re-ran finished work, an image turn burning a
second GPU render and orphaning the first one's artifact. The element that
leaked the unauthorized call is the persona pass; everything upstream of it was
done, deterministic, and already traced.

The fix branches at the edge out of START, so route() and specialist() stay
byte-identical and "every certification pin passes unmodified" holds
structurally instead of by hope: entry() takes the shortcut only when the chat
door's retry set `retry_draw` AND draw 1's finished work arrived complete in
the carry; anything less is a full re-run, TRACED.

Two traps this file pins:
- LangGraph silently drops input keys not declared on the state schema. An
  undeclared retry_draw would never reach entry(), and the whole fix would be a
  quiet no-op (the probe: the node saw ['a', 'b'] of {'a','b','retry_draw'}).
  test_retry_draw_reaches_the_entry_edge exists for exactly that.
- A fallback nobody can see is how you get a green that means nothing: the
  incomplete-carry re-run must trace prebyte_retry_full_rerun and why.
"""
import copy

import pytest

from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app
from fastapi.testclient import TestClient
from test_canonical_trace_row import CALL, TEXT, Draws, _the_row
from test_skeleton import PNG, FakeUpstream


def call_draw():
    # NEVER share the imported dicts with the graph: speak() rewrites chunk
    # deltas in place (the citation hold sets content to ""), and a mutated
    # module-level TEXT silently broke test_canonical_trace_row's recovery
    # test when this file ran first. Deep-copy per draw.
    return copy.deepcopy(CALL)


def text_draw():
    return copy.deepcopy(TEXT)

BASE = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "a search ask"}]}


class CountingRouter:
    """Routes where it is told and counts how often it was asked."""

    def __init__(self, route: str):
        self.route = route
        self.calls = 0

    async def ainvoke(self, msgs):
        self.calls += 1
        from types import SimpleNamespace
        return SimpleNamespace(content='{"route": "%s", "intent": "the ask"}' % self.route)


def _client(tmp_path, monkeypatch, route: str, up):
    """A router-enabled app whose specialist runs exactly once per real run."""
    router = CountingRouter(route)
    runs: list = []

    async def fake_specialist(job, ctx):
        runs.append(job.job_id)
        if route == "search":
            return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                          summary="Found one blue mug.",
                          provenance={"kind": "search", "query": "blue mug",
                                      "sources": [{"id": 1, "url": "https://ex.example/mugs",
                                                   "title": "Mugs"}]})
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[d], summary="a mug")

    # The real ones first, so the stand-in is not overwritten when this file
    # runs alone (the trap test_skeleton documents): Deps' load_specialists
    # imports the modules, and a fresh import would re-register over the patch.
    from chord.server import load_specialists
    load_specialists()
    monkeypatch.setitem(specialists.SPECIALISTS, route, fake_specialist)
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        enabled_routes=frozenset({route}))
    deps = Deps(settings, upstream=up, model=lambda name: router)
    return TestClient(create_app(deps)), router, runs, deps


def test_a_specialist_retry_reuses_the_finished_work(tmp_path, monkeypatch):
    """The search ran, the persona leaked a call pre-byte, and the retry must
    re-ask the PERSONA only: one router call, one specialist run, draw 2's text
    delivered, and the reused decision on the trace. On the old code the whole
    graph re-ran -- router 2, specialist 2 -- and the re-route could even come
    back a different route, silently dropping the search."""
    cited = {"content": "It is sunny in Paris [[cite:1]]."}   # the citation format
    up = Draws([call_draw()], [cited])
    client, router, runs, deps = _client(tmp_path, monkeypatch, "search", up)

    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert r.status_code == 200, r.text
    assert "It is sunny in Paris" in r.text, "draw 2's prose never reached the client"
    assert "weather" not in r.text, "draw 1's invalid call leaked"
    assert len(up.bodies) == 2, f"exactly two persona draws, got {len(up.bodies)}"
    assert router.calls == 1, f"the router re-decided a finished turn: {router.calls} calls"
    assert len(runs) == 1, f"the specialist re-ran finished work: {len(runs)} runs"

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "prebyte_retry"
    assert row["prebyte_retry_recovered"] is True
    assert row["prebyte_retry_reused_decision"] == "search"
    assert row.get("prebyte_retry_full_rerun") is None, "the carry was complete; no re-run"


def test_a_chat_route_retry_does_not_recall_the_router(tmp_path, monkeypatch):
    """Even the cheap route paid a second router call for one persona slip."""
    up = Draws([call_draw()], [text_draw()])
    client, router, runs, deps = _client(tmp_path, monkeypatch, "chat", up)

    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert r.status_code == 200, r.text
    assert "It is sunny in Paris." in r.text
    assert router.calls == 1, f"router recalled on retry: {router.calls}"
    row = _the_row(tmp_path)
    assert row["prebyte_retry_reused_decision"] == "chat"


def test_a_fresh_turn_never_takes_the_entry_shortcut(tmp_path, monkeypatch):
    """The positive control: entry() must be invisible on ordinary turns --
    routing happens, no retry fields exist, and the row's disposition is none."""
    up = Draws([text_draw()])
    client, router, runs, deps = _client(tmp_path, monkeypatch, "search", up)

    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert r.status_code == 200, r.text
    assert router.calls == 1 and len(runs) == 1, "a fresh turn must route and run its specialist"
    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "none"
    assert "prebyte_retry_reused_decision" not in row
    assert "prebyte_retry_full_rerun" not in row


def _graph(tmp_path, router):
    from chord import graph as graph_mod, registry
    from chord.artifacts import ArtifactStore
    from chord.trace import Trace

    up = FakeUpstream()
    trace = Trace(persona_id="generic", model_id_requested=None)
    g = graph_mod.build(
        settings=Settings(data_dir=tmp_path, router_enabled=True),
        capabilities=registry.load(), artifacts=ArtifactStore(tmp_path / "artifacts"),
        upstream=up, model=lambda name: router, trace=trace)
    return g, trace


def _state(**extra):
    base = {"params": {}, "messages": [{"role": "user", "content": "hi"}], "stream": False,
            "verbosity": None, "persona_model": None, "audio_output": False,
            "allowed_routes": None, "count_only": False}
    return {**base, **extra}


@pytest.mark.asyncio
async def test_retry_draw_reaches_the_entry_edge(tmp_path):
    """the channel trap: StateGraph(TurnState) silently drops input keys the
    schema does not declare, so an undeclared retry_draw never reaches entry()
    and the fix is a quiet no-op. This test fails -- by recalling the router --
    the moment the declaration disappears OR the entry branch does."""
    from chord.router import RouteDecision

    router = CountingRouter("chat")
    g, trace = _graph(tmp_path, router)
    out = await g.ainvoke(_state(retry_draw=True, decision=RouteDecision(route="chat")))
    assert router.calls == 0, "entry() never saw retry_draw: the channel dropped it"
    assert out.get("text"), "the chat node did not speak"
    assert trace.fields.get("prebyte_retry_reused_decision") == "chat"


@pytest.mark.asyncio
async def test_an_incomplete_carry_falls_back_and_says_so(tmp_path):
    """retry_draw without a decision is a broken carry, not a shortcut: full
    re-run, and the fallback is TRACED with its reason -- a silent fallback is
    a green that means nothing."""
    router = CountingRouter("chat")
    g, trace = _graph(tmp_path, router)
    out = await g.ainvoke(_state(retry_draw=True))          # no decision carried
    assert router.calls == 1, "an incomplete carry must take the full re-run"
    assert out.get("text")
    assert trace.fields.get("prebyte_retry_full_rerun") is True
    assert trace.fields.get("prebyte_retry_carry_incomplete") == "no decision"
