"""EVERY response that traverses the normalization boundary carries the canonical row.

Every response, not every *successful* one. The first implementation recorded inside the
SSE generator and after a completed non-stream turn, which meant the pre-header refusals
— the population we most want to count — were the ones missing. "Every response" quietly
meant "every response that got far enough to succeed" (2026-09-19).

The row is the DENOMINATOR. Recording only repairs and refusals gives numerators with no
base population, which is exactly why `2/222` could not price a neighbouring branch — and
it cannot be fixed later, because the clean rows were never written.

Each cell reads the trace the response actually produced, by id, and asserts the required
fields are present. Asserting the fields exist in `normalize.canonical_row()` would prove
the schema and not the wiring, which is the distinction that made the bind sit unreachable
with thirteen passing unit tests.

2026-09-19.
"""
import asyncio
import json
import pathlib

import pytest

from test_skeleton import FakeUpstream, make

REQUIRED = ("transport", "tool_parameter_state", "call_observation",
            "client_sse_data_emitted", "client_content_delta_emitted",
            "normalization_disposition")

STRICT = {"type": "object", "properties": {"city": {"type": "string"}},
          "required": ["city"], "additionalProperties": False}
TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": STRICT}}
BASE = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}


def _rows(root: pathlib.Path) -> list[dict]:
    return [json.loads(line)
            for f in root.rglob("*.jsonl")
            for line in f.read_text().splitlines() if line.strip()]


def _the_row(root: pathlib.Path) -> dict:
    """The single trace for this turn. One row per response is itself the property: a
    trace that can appear twice destroys the denominator, because you can no longer count
    responses by counting rows."""
    rows = _rows(root)
    assert len(rows) == 1, f"expected exactly one trace row, got {len(rows)}"
    row = rows[0]
    missing = [f for f in REQUIRED if f not in row]
    assert not missing, f"canonical fields missing from the row: {missing}"
    return row


class Returns(FakeUpstream):
    """Returns a chosen message, or streams content and then a call."""

    def __init__(self, message=None, *, stream_call=None, stream_content="I'll check."):
        super().__init__()
        self.message = message
        self.stream_call = stream_call
        self.stream_content = stream_content

    async def complete(self, body):
        await asyncio.sleep(0)
        self.bodies.append(body)
        finish = "tool_calls" if (self.message or {}).get("tool_calls") else "stop"
        return ({"choices": [{"index": 0, "finish_reason": finish, "message": self.message}],
                 "usage": {"total_tokens": 2}}, {"model-api-base": "http://x/v1"})

    async def stream(self, body):
        await asyncio.sleep(0)
        self.bodies.append(body)
        yield None, {"model-api-base": "http://x/v1"}
        if self.stream_content:
            yield {"object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {"content": self.stream_content}, "finish_reason": None}]}, {}
        if self.stream_call:
            yield {"object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "c0", "type": "function", "function": self.stream_call}]},
                 "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk", "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def _call(name, arguments):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c0", "type": "function",
                            "function": {"name": name, "arguments": arguments}}]}


# ------------------------------------------------------------------ the clean row

def test_a_clean_non_stream_response_carries_the_denominator(tmp_path):
    """Nothing went wrong, and the row still exists. THIS is the cell that matters most:
    the clean rows are the base population, and they are the ones a repair-only
    instrument silently omits."""
    _, client = make(tmp_path, Returns({"role": "assistant", "content": "hello"}))
    client.post("/v1/chat/completions", json={**BASE, "tools": [TOOL]})

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "none"
    assert row["call_observation"] == "none"
    assert row["tool_parameter_state"] == "nonempty"
    assert row["transport"] == "nonstream"


def test_a_clean_stream_response_carries_the_denominator(tmp_path):
    _, client = make(tmp_path, Returns(stream_call=None))
    client.post("/v1/chat/completions", json={**BASE, "tools": [TOOL], "stream": True})

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "none"
    assert row["transport"] == "stream"
    assert row["client_content_delta_emitted"] is True


# ------------------------------------------------------------------ repaired

def test_a_repaired_response_records_what_the_BACKEND_sent(tmp_path):
    """The observation is what happened; the disposition is what we did about it.

    The client received `get_weather`. The row must still carry the name the backend
    actually chose, or the telemetry records our repair instead of the defect it repaired
    -- and the population we are trying to measure disappears into its own fix."""
    _, client = make(tmp_path, Returns(_call("weather_getCurrent_2604", '{"city": "Paris"}')))
    r = client.post("/v1/chat/completions", json={**BASE, "tools": [TOOL]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "sole_tool_bind"
    observed = row["call_observation"]["calls"][0]
    assert observed["function_name"] == "weather_getCurrent_2604", \
        "the row recorded the repair rather than the thing repaired"
    assert observed["membership"] == "no_match"


# ------------------------------------------------------------------ pre-header refusals

def test_a_non_stream_refusal_carries_the_row(tmp_path):
    """Refusals are the population we most want to count, and they return BEFORE the
    completed-response path that used to be the only place recording anything."""
    _, client = make(tmp_path, Returns(_call("weather_getCurrent_2604", '{"city": 7}')))
    r = client.post("/v1/chat/completions", json={**BASE, "tools": [TOOL]})
    assert r.status_code == 502

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "unrepaired"
    assert row["client_sse_data_emitted"] is False, "nothing had crossed the boundary"
    assert row["call_observation"]["calls"][0]["function_name"] == "weather_getCurrent_2604"


def test_a_buffered_stream_refusal_carries_the_row(tmp_path):
    """Tools declared, so the stream is quarantined whole and refused before a byte."""
    _, client = make(tmp_path, Returns(stream_call={"name": "weather_getCurrent_2604",
                                                    "arguments": '{"city": 7}'},
                                       stream_content=None))
    r = client.post("/v1/chat/completions", json={**BASE, "tools": [TOOL], "stream": True})
    assert r.status_code == 502

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "unrepaired"
    assert row["transport"] == "stream"
    assert row["client_sse_data_emitted"] is False


def test_a_no_tools_prebyte_refusal_carries_the_row(tmp_path):
    """No tool parameter, the call arrives before any prose, and the retry fails too.

    `Returns` gives the same stream every draw, so this exercises a backend that does it
    twice. The disposition is `unrepaired`, NOT `prebyte_retry` -- the enum names the
    OUTCOME, and a retry that was attempted and failed did not normalize anything.
    Reserving `prebyte_retry` for success is the same distinction as `drop` versus
    `extension_error`: the action taken is not the result obtained."""
    _, client = make(tmp_path, Returns(stream_call={"name": "weather", "arguments": "{}"},
                                       stream_content=None))
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert r.status_code == 502

    row = _the_row(tmp_path)
    assert row["tool_parameter_state"] == "absent"
    assert row["normalization_disposition"] == "unrepaired"
    assert row["prebyte_retry_attempted"] is True, "the bounded retry must still have run"
    assert row["prebyte_retry_recovered"] is False
    assert row["client_sse_data_emitted"] is False


# ------------------------------------------------------------------ post-prose

def test_a_post_prose_suppression_carries_the_row(tmp_path):
    """Prose shipped, then an unauthorised call. The one case where the refusal has to
    be in-band, and the row must show that bytes -- and content specifically -- had
    already crossed."""
    _, client = make(tmp_path, Returns(stream_call={"name": "weather", "arguments": "{}"}))
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert r.status_code == 200 and "[DONE]" not in r.text

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "extension_error", \
        "`drop` is the intermediate action; the terminal outcome is the public error " \
        "contract requires, and the enum keeps both so they stay distinguishable"
    assert row["client_sse_data_emitted"] is True
    assert row["client_content_delta_emitted"] is True, \
        "prose crossing is the predicate the orphaned-promise concern rests on"
    assert row["call_observation"]["calls"][0]["function_name"] == "weather"


# ------------------------------------------------------------------ the shape axis

def test_a_mixed_shape_response_is_recorded_as_mixed(tmp_path):
    """`tool_calls` AND a legacy `function_call` in one response. Derived from what came
    back -- never inferred from what the request declared, which has no vote on what the
    backend chose to send."""
    message = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "c0", "type": "function",
                               "function": {"name": "get_weather", "arguments": '{"city": "P"}'}}],
               "function_call": {"name": "get_weather", "arguments": '{"city": "P"}'}}
    _, client = make(tmp_path, Returns(message))
    client.post("/v1/chat/completions", json={**BASE, "tools": [TOOL]})

    row = _the_row(tmp_path)
    assert row["call_observation"]["call_shape"] == "mixed", \
        "a response carrying both shapes is the boundary this field exists for"
    assert len(row["call_observation"]["calls"]) == 2


def test_a_streamed_mixed_shape_with_a_malformed_fragment_records_each_shape(tmp_path):
    """The stream-path shape computation was the asymmetry: it stamped the
    appended MALFORMED sentinel modern and the legacy call modern when both
    legacy and malformed co-occurred. Pinned here so a regression is caught
    at the boundary this field exists for, not in a postmortem."""
    from chord.chat_api import _stream_calls_shaped

    events = [
        ("custom", {"chunk": {"choices": [{
            "delta": {"tool_calls": [
                {"index": 0, "function": {"name": "a", "arguments": "{}"}},
                {"index": 1, "function": {"name": "b", "arguments": "{}"}},
                # malformed: index must be int (per chat_api._stream_calls);
                # a string index triggers the malformed flag without raising.
                {"index": "not-int", "function": {"name": "x", "arguments": "{}"}},
            ]},
        }]}}),
        ("custom", {"chunk": {"choices": [{
            "delta": {"function_call": {"name": "legacy", "arguments": "{}"}},
        }]}}),
    ]

    shaped = _stream_calls_shaped(events)
    shapes = [shape for shape, _, _ in shaped]
    # Order is: modern slots, legacy call, MALFORMED sentinel -- the order
    # `_stream_calls` returns. Each shape is stamped by POSITION in that
    # ordering, so the fix is to subtract malformed from the modern count.
    assert shapes == ["modern", "modern", "legacy", "modern"], \
        f"legacy call stamped wrong or MALFORMED stamped modern; got {shapes}"


@pytest.mark.parametrize("declared, expected", [
    (None, "absent"),
    ([], "empty"),
    ([TOOL], "nonempty"),
])
def test_the_declaration_axis_is_three_states_on_the_row(tmp_path, declared, expected):
    """`bool(declared)` was F1. Collapsing the axis in the INSTRUMENT is worse than in the
    guard: a guard with a bad trigger fails a test, an instrument that cannot see a
    distinction emits confident wrong numbers forever."""
    body = dict(BASE)
    if declared is not None:
        body["tools"] = declared
    _, client = make(tmp_path, Returns({"role": "assistant", "content": "hi"}))
    client.post("/v1/chat/completions", json=body)

    assert _the_row(tmp_path)["tool_parameter_state"] == expected


def test_a_call_only_bind_reports_the_bytes_it_then_emits(tmp_path):
    """THE SHARPEST FALSIFIER on write timing, and it was a real defect.

    A call-only stream has no prose. Every upstream fragment is stripped and the repaired
    call is emitted AFTER the loop -- so a row written in the inner `finally` recorded
    `client_sse_data_emitted=False` and then the generator yielded the call. The row
    described a boundary that had not been crossed yet.

    The write now lives in a finally that encloses every yield."""
    _, client = make(tmp_path, Returns(stream_call={"name": "weather_getCurrent_2604",
                                                    "arguments": '{"city": "Paris"}'},
                                       stream_content=None))
    r = client.post("/v1/chat/completions", json={**BASE, "tools": [TOOL], "stream": True})
    assert r.status_code == 200
    assert "get_weather" in r.text and "weather_getCurrent_2604" not in r.text

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "sole_tool_bind"
    assert row["client_sse_data_emitted"] is True, \
        "the repaired call went out; a row claiming nothing crossed is describing a " \
        "moment before the response it is supposed to describe"
    assert row["client_content_delta_emitted"] is False, "there was no prose in this turn"


def test_a_failure_before_any_frame_still_reports_the_error_it_emitted(tmp_path):
    """The `stream_failed` error object is client-visible data.

    Failing before any ordinary frame used to leave the row saying nothing crossed while
    the error went out — the same accounting gap as the repaired call, on the path least
    likely to be exercised, which is where an unexercised gap survives longest."""
    class Explodes(FakeUpstream):
        async def stream(self, body):
            await asyncio.sleep(0)
            self.bodies.append(body)
            yield None, {"model-api-base": "http://x/v1"}
            # Real content, so priming COMPLETES and the response is committed. An
            # empty string is not a content delta, so the raise landed during priming and
            # became an ordinary HTTP error -- a different path from the one under test.
            yield {"object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {"content": "starting"}, "finish_reason": None}]}, {}
            raise RuntimeError("upstream died mid-stream")

    _, client = make(tmp_path, Explodes())
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert "stream_failed" in r.text, "the error frame is what this cell is about"

    row = _the_row(tmp_path)
    assert row["client_sse_data_emitted"] is True, \
        "the error object crossed the boundary; a row claiming nothing did is wrong " \
        "about the only thing the client received"


# ------------------------------------------------------------------ the bounded retry

class Draws(FakeUpstream):
    """A different stream per draw, so a retry can be told from a repeat."""

    def __init__(self, *draws):
        super().__init__()
        self.draws = list(draws)
        self.n = 0

    async def stream(self, body):
        await asyncio.sleep(0)
        self.bodies.append(body)
        chunks = self.draws[min(self.n, len(self.draws) - 1)]
        self.n += 1
        yield None, {"model-api-base": "http://x/v1"}
        for delta in chunks:
            yield {"object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}

    async def complete(self, body):
        """The same draw, unstreamed: its deltas assembled into one message, so one
        `Draws` states a sequence both transports are asked the same way."""
        await asyncio.sleep(0)
        self.bodies.append(body)
        deltas = self.draws[min(self.n, len(self.draws) - 1)]
        self.n += 1
        content = "".join(d["content"] for d in deltas if d.get("content")) or None
        calls = [{k: v for k, v in c.items() if k != "index"}
                 for d in deltas for c in d.get("tool_calls") or []]
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        return ({"choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                              "message": message}],
                 "usage": {"total_tokens": 2}}, {"model-api-base": "http://x/v1"})


CALL = {"tool_calls": [{"index": 0, "id": "c0", "type": "function",
                        "function": {"name": "weather", "arguments": "{}"}}]}
TEXT = {"content": "It is sunny in Paris."}


def test_a_call_only_first_draw_recovers_when_the_second_is_text(tmp_path):
    """THE RETRY, proven at the door rather than by a trace bit.

    A `retry_attempted` flag false-greens this: the graph captured the original stop
    event at build time, so handing the retry a NEW event left the node checking the old,
    already-set one. Two upstream draws, an empty second stream, and `recovered: True` on
    nothing at all (a review predicted the capture; my own probe produced the false green).

    So this asserts the TEXT comes back and that exactly two draws happened."""
    up = Draws([CALL], [TEXT])
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})

    assert r.status_code == 200
    assert "It is sunny in Paris." in r.text, "the retry's prose never reached the client"
    assert "weather" not in r.text, "the first draw's invalid call leaked"
    assert len(up.bodies) == 2, f"expected exactly two upstream draws, got {len(up.bodies)}"

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "prebyte_retry"
    assert row["prebyte_retry_recovered"] is True
    observed = row["call_observation"]
    assert observed != "none", \
        "a successful retry erased its own numerator: the row must still carry the " \
        "invalid call that caused it, or repairs are reported against no observed defect"
    assert observed["calls"][0]["function_name"] == "weather"


def test_an_adversarial_second_draw_fails_pre_header(tmp_path):
    """Draw two emits PROSE and then a call. Call-only is not sufficient to test this.

    Primed until-decisive, the prose would be released, the response committed, and the
    turn would end in the in-band error -- when the amendment says a failed bounded retry
    is still pre-byte and returns an ordinary HTTP error. The retry is buffered whole so
    a call ANYWHERE in draw two is a pre-header failure."""
    up = Draws([CALL], [TEXT, CALL])
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})

    assert r.status_code == 502, \
        f"a failed retry must refuse BEFORE headers, got {r.status_code}: {r.text[:200]}"
    assert "It is sunny in Paris." not in r.text, \
        "draw two's prose was released before its call was known -- the retry was not " \
        "quarantined whole"
    assert len(up.bodies) == 2

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "unrepaired"
    assert row["prebyte_retry_recovered"] is False
    assert row["client_sse_data_emitted"] is False


def test_an_empty_second_draw_is_not_a_recovery(tmp_path):
    """Nothing to send is not prose. Recovery is positive -- content present and no call
    -- because "no call" alone is satisfied by a stream that produced nothing, which is
    an empty success handed to a caller who needed an actionable error."""
    up = Draws([CALL], [])
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})

    assert r.status_code == 502, f"an empty retry was treated as a recovery: {r.text[:200]}"
    assert _the_row(tmp_path)["prebyte_retry_recovered"] is False


# ------------------------------------------------------------------ the retry, unstreamed

def test_a_non_stream_no_tools_call_retries_once_for_text_as_the_stream_does(tmp_path):
    """review 2026-09-24 B14. The same request, no tools declared, first draw a call and
    second draw prose: streamed it retried and answered 200, unstreamed it refused 502 on
    the first draw. One request, two answers, decided by the transport alone. Non-stream
    now takes the same single bounded retry, and records it the same way."""
    up = Draws([CALL], [TEXT])
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json=BASE)

    assert r.status_code == 200, f"no retry before refusing: {r.status_code} {r.text[:200]}"
    message = r.json()["choices"][0]["message"]
    assert message["content"] == "It is sunny in Paris."
    assert not message.get("tool_calls"), "the first draw's invalid call leaked"
    assert len(up.bodies) == 2, f"expected exactly two upstream draws, got {len(up.bodies)}"

    row = _the_row(tmp_path)
    assert row["transport"] == "nonstream"
    assert row["normalization_disposition"] == "prebyte_retry"
    assert row["prebyte_retry_attempted"] is True
    assert row["prebyte_retry_recovered"] is True
    observed = row["call_observation"]
    assert observed != "none", "a successful retry erased the call that caused it"
    assert observed["calls"][0]["function_name"] == "weather"


@pytest.mark.parametrize("second", [[CALL], [TEXT, CALL], []],
                         ids=["call-again", "prose-then-call", "empty"])
def test_a_non_stream_retry_that_is_not_text_only_is_still_the_502(tmp_path, second):
    """review 2026-09-24 B14: ONE retry, and recovery is positive -- prose present and no
    call. A second call, or nothing at all, answers the same 502 as before the retry."""
    up = Draws([CALL], second)
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json=BASE)

    assert r.status_code == 502, f"a failed retry must still refuse: {r.text[:200]}"
    assert r.json()["error"]["code"] == "upstream_tool_call_contract_violation"
    assert len(up.bodies) == 2, f"exactly one retry, got {len(up.bodies)} draws"

    row = _the_row(tmp_path)
    assert row["normalization_disposition"] == "unrepaired"
    assert row["prebyte_retry_attempted"] is True
    assert row["prebyte_retry_recovered"] is False
    assert row["call_observation"]["calls"][0]["function_name"] == "weather"
