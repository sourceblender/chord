"""Responses event enumeration gate (#189 finding side).

Every event the service emits, across every stream shape, must be EXACTLY the
pinned spec's ResponseStreamEvent: strict-clean (no undeclared field) AND every
required field present. `validate_payload(..., fields="strict")` fails in both
directions, so a single `verdict == "pass"` is the two-sided gate.

This is the red gate for constructing Responses events from per-type spec field
lists (the #189 structural recommendation): a constructed event that drops a
required field, keeps an off-spec one, or wires a wrong field list goes red here.
It passes on today's hand-assembled emission (verified strict-clean at #196), so
its job is to LOCK that invariant across the refactor and across every shape.
"""
import pathlib
import tempfile


from qa.conformance.schema import Spec, validate_payload
from chord import specialists
from chord.contract import Outcome, Result
from test_client_tools_own_the_turn import Calling
from test_progress import FixedRouter
from test_responses import MODEL, WEATHER, events_of, make
from test_skeleton import PNG

SPEC = Spec()


def _tmp():
    return pathlib.Path(tempfile.mkdtemp())


def _assert_exactly_spec(events, label):
    """Each event is exactly ResponseStreamEvent, and sequence_number is contiguous."""
    assert events, f"{label}: emitted no events"
    for e in events:
        row = validate_payload(e, kind="response-event", spec=SPEC, fields="strict")
        assert row["verdict"] == "pass", (
            f"{label}: {e.get('type')} is not spec-exact -> {row['evidence']}")
    seqs = [e["sequence_number"] for e in events]
    assert seqs == list(range(len(events))), f"{label}: sequence_number not 0..n: {seqs}"
    return {e["type"] for e in events}


def _text():
    _, client, _ = make(_tmp())
    return events_of(client, {"model": MODEL, "input": "hello"})


def _function_call():
    _, client, _ = make(_tmp(), Calling())
    return events_of(client, {"model": MODEL, "input": "weather in Paris and Oslo?", "tools": [WEATHER]})


def _image(monkeypatch):
    async def image(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")], summary="a mug")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    _, client, _ = make(_tmp(), router=FixedRouter, router_enabled=True,
                        enabled_routes=frozenset({"image"}))
    return events_of(client, {"model": MODEL, "input": "draw a mug", "tools": [{"type": "image_generation"}]})


def _web_search(monkeypatch):
    async def search(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      summary="Paris is sunny.",
                      provenance={"kind": "search", "query": "paris weather today",
                                  "sources": [{"id": 1, "url": "https://weather.example/paris",
                                               "title": "Paris weather"}]})

    class SearchRouter:
        async def ainvoke(self, msgs):
            class R:
                content = '{"route": "search", "intent": "paris weather today"}'
            return R()

    monkeypatch.setitem(specialists.SPECIALISTS, "search", search)
    _, client, _ = make(_tmp(), router=SearchRouter, router_enabled=True,
                        enabled_routes=frozenset({"search"}))
    return events_of(client, {"model": MODEL, "input": "weather in Paris?", "tools": [{"type": "web_search"}]})


def test_text_stream_events_are_exactly_spec():
    _assert_exactly_spec(_text(), "text")


def test_function_call_stream_events_are_exactly_spec():
    _assert_exactly_spec(_function_call(), "function-call")


def test_image_generation_stream_events_are_exactly_spec(monkeypatch):
    _assert_exactly_spec(_image(monkeypatch), "image")


def test_web_search_stream_events_are_exactly_spec(monkeypatch):
    _assert_exactly_spec(_web_search(monkeypatch), "web-search")


def _spec_event_types():
    """Every event `type` ResponseStreamEvent declares, derived from the pinned
    spec at runtime -- no hand-maintained list to rot."""
    doc = SPEC.document
    schemas = doc["components"]["schemas"]
    rse = schemas["ResponseStreamEvent"]
    out = set()
    for variant in rse.get("oneOf") or rse.get("anyOf") or []:
        ref = variant.get("$ref", "").split("/")[-1]
        prop = schemas.get(ref, {}).get("properties", {}).get("type", {})
        for key in ("const", "enum"):
            if key in prop:
                vals = prop[key] if isinstance(prop[key], list) else [prop[key]]
                out.update(vals)
    return out


# A small, stable lifecycle invariant: every real stream opens, produces an item,
# and closes. This does NOT enumerate optional events (annotation.added needs a
# citing model, audio needs modalities) -- the subset check below guards those.
CORE_LIFECYCLE = {
    "response.created", "response.in_progress",
    "response.output_item.added", "response.output_item.done",
    "response.completed",
}


def test_every_emitted_event_type_is_spec_declared_and_lifecycle_is_covered(monkeypatch):
    seen = set()
    seen |= _assert_exactly_spec(_text(), "text")
    seen |= _assert_exactly_spec(_function_call(), "function-call")
    seen |= _assert_exactly_spec(_image(monkeypatch), "image")
    seen |= _assert_exactly_spec(_web_search(monkeypatch), "web-search")
    # Non-decaying: nothing the service emits may be unknown to the pinned spec.
    unknown = seen - _spec_event_types()
    assert not unknown, f"emitted event types not declared by ResponseStreamEvent: {sorted(unknown)}"
    # And the shapes actually exercise the core lifecycle (so the gate isn't empty).
    assert CORE_LIFECYCLE <= seen, f"core lifecycle events never emitted: {sorted(CORE_LIFECYCLE - seen)}"
    # The finding's own event must be exercised and spec-exact, or this gate is theatre.
    assert "response.function_call_arguments.done" in seen
