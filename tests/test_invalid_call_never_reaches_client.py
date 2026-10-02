"""THE INVARIANT: for every emission path, an invalid call never reaches the client.

One property, N paths, DERIVED rather than enumerated. The rule came from the failure:
a hand-listed matrix is a curated list wearing a cross-product's clothes, and **a
missing cell is indistinguishable from a passing one**. Her model-id drift test
hand-listed four doors, none of them the one that had drifted, and passed with the
defect re-planted.

So the cases are generated over four dimensions and the cell count is asserted. If a
dimension gains or loses a value, the count fails and names it instead of the matrix
silently shrinking while staying green.

Why this file exists at all: A review held #304 at 21351cd REQUEST CHANGES. The
`if declared:` shortcut a review predicted was fixed on the non-stream path and survived on
the stream path, where `bool(declared)` was ALSO the buffering trigger -- the same
shortcut twice in series, on the one path I was not looking at. And MALFORMED refused
on stream while passing non-stream. **Two paths disagreeing in opposite directions is
the tell that the property was being asserted per-path instead of about the response.**

2026-09-19.
"""
import asyncio
import itertools
import json

import pytest

from test_skeleton import FakeUpstream, make

BASE = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "w",
                                         "parameters": {"type": "object", "properties": {}}}}

SHAPES = ("modern", "legacy")            # tool_calls vs function_call
TRANSPORTS = ("nonstream", "stream")
DECLARED = ("absent", "empty", "nonempty")

# RETURNED-CALL CLASS -- the shape of what came back, never its expected verdict.
# `match_shaped` was called `valid` and the generator then DROPPED it wherever nothing
# was declared, reasoning that a valid name with no declarations is nonsense. It is the
# most real input there is and its correct answer is REFUSE, so filtering it made the
# live defect unrepresentable a second time, after we had named that exact failure.
# Two of those dropped cells were genuinely broken and nothing was testing them.
#
# The verdict is DERIVED per cell from class x declaration state, below. An axis named
# by its outcome cannot express a cell whose outcome is the thing in question.
RETURNED = ("match_shaped", "undeclared", "malformed")

# The full Cartesian product. NEVER a filtered expression: `assert observed == intended`
# only catches drift FROM the intention, so an intention computed by a filter certifies
# the filtered population -- a guard against a shrinking matrix, satisfied by the shrunk
# matrix (a corrected suggestion).
ALL = sorted(itertools.product(SHAPES, TRANSPORTS, DECLARED, RETURNED))

# Omission must cost a NAMED entry with a written reason, outside the generator. Its
# value is not what it holds: filling it in is what forces you to discover it should be
# empty. Every one of the 36 is a real input.
EXCLUDED: dict[tuple, str] = {}

# THE SIX CELLS THAT WERE OPEN ARE CLOSED. They are ordinary passing cells now, and
# this dict is empty rather than deleted -- the same reason `EXCLUDED` is empty and
# present: filling it in is what forces you to discover it should be empty, and a
# structure that disappears when it is satisfied takes its history with it.
#
# What they measured: a stream whose request carried no tool parameter was not buffered,
# so an invalid call could only be found after content had crossed the boundary. Closed
# on 2026-09-19 by option D's no-tools path -- priming stops at the first decisive delta,
# the call is suppressed, and the turn ends in an explicit in-band error rather than a
# terminator that asserts clean completion.
#
# They closed by XPASS(strict), which is the point of strict: the moment the defect
# stopped reproducing the suite refused to stay green about it, and the reasons named the
# defect predicate rather than a remedy, so they recognised a fix that arrived by a route
# nobody had listed when they were written.
KNOWN_OPEN: dict[tuple, str] = {}

CASES = [
    pytest.param(*c, marks=pytest.mark.xfail(
        strict=True,
        reason=f"REAL DEFECT against the invariant, not an out-of-profile cell: "
               f"{KNOWN_OPEN[c]}. Disposition open; remedies live in the issue, not "
               f"here. strict=True prevents silent closure; it does not dispose of it."))
    if c in KNOWN_OPEN else pytest.param(*c)
    for c in ALL
]


def test_the_matrix_is_the_full_cross_product():
    """The SET against an UNFILTERED product, plus every exclusion justified in writing.

    A count dies to a duplicate replacing a missing cell; a set compared against a
    filtered intention dies to the filter. Both happened here, in that order."""
    intended = set(itertools.product(SHAPES, TRANSPORTS, DECLARED, RETURNED))
    assert len(intended) == 36, len(intended)
    observed = {tuple(c.values) for c in CASES}
    assert observed == intended - set(EXCLUDED), (
        f"missing {sorted(intended - set(EXCLUDED) - observed)}, "
        f"extra {sorted(observed - intended)}")
    assert all(reason.strip() for reason in EXCLUDED.values()), "every exclusion needs a reason"
    assert all(reason.strip() for reason in KNOWN_OPEN.values()), "every known-open cell needs a reason"
    assert set(KNOWN_OPEN) <= intended, "known-open cells must be real cells of the matrix"
    assert KNOWN_OPEN == {}, (
        f"a cell was re-opened: {sorted(KNOWN_OPEN)}. All 36 pass as of 2026-09-19; "
        f"re-opening one is a real regression and must be argued, not assumed.")

    # A tripwire for the rule above, not a proof of it: no wordlist can decide that a
    # sentence describes a defect rather than a repair. It catches the specific way the
    # menu got in last time -- prescriptive vocabulary -- and it fails loudly enough to
    # send the next author to the comment. Deleting a remedy sentence is the fix; adding
    # a word here to get past it is not.
    PRESCRIPTIVE = ("needs ", "requires ", "closing it", "close it", "should be fixed",
                    "option ", "the fix", "decision")
    for cell, reason in KNOWN_OPEN.items():
        found = [w for w in PRESCRIPTIVE if w in reason.lower()]
        assert not found, (
            f"{cell}: an xfail reason states the defect predicate, never the remedy -- "
            f"a reason that names the answer cannot recognise a different one. Found "
            f"{found}. Put the options in the issue.")


def _name_for(response):
    return {"match_shaped": "get_weather", "undeclared": "weather_getCurrent_2604"}.get(response)


class Planted(FakeUpstream):
    """Returns the planted call on whichever transport is asked for, fragmenting the
    name across deltas on the stream path and emitting content first.

    PROVENANCE, corrected 2026-09-19: content-first is an ADVERSARIAL CONSTRUCTED
    shape, not an observed one. The live evidence establishes only content beside a
    DECLARED call (`S-cf-060`, `book_table`). Across all 227 rows of
    `evidence/2026-09-18-undeclared-tool-name-289/calls.jsonl`, both undeclared calls
    carried `content: None` -- prose and an invalid name never co-occurred. Planting
    the intersection is correct, because it is the shape that defeats call-only
    buffering; claiming it was measured is not."""

    def __init__(self, shape, response):
        super().__init__()
        self.shape, self.response = shape, response

    def _fn(self):
        name = _name_for(self.response)
        return "not-an-object" if name is None else {"name": name, "arguments": "{}"}

    async def complete(self, body):
        await asyncio.sleep(0)
        self.bodies.append(body)
        msg = {"role": "assistant", "content": "I'll check both cities."}
        fn = self._fn()
        if self.shape == "legacy":
            msg["function_call"] = fn
        else:
            msg["tool_calls"] = [{"id": "c0", "type": "function", "function": fn}]
        return ({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}],
                 "usage": {"total_tokens": 2}},
                {"model-api-base": "http://x/v1"})

    async def stream(self, body):
        await asyncio.sleep(0)
        self.bodies.append(body)
        yield None, {"model-api-base": "http://x/v1"}
        # content FIRST: valid when emitted, and unretractable once it is
        yield {"object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": {"content": "I'll check both cities."},
                            "finish_reason": None}]}, {}
        fn = self._fn()
        if isinstance(fn, str):
            frags = [{"function": fn}]
        else:
            n = fn["name"]
            frags = [{"function": {"name": n[:4]}}, {"function": {"name": n[4:], "arguments": "{}"}}]
        for i, frag in enumerate(frags):
            delta = ({"function_call": frag.get("function")} if self.shape == "legacy"
                     else {"tool_calls": [{"index": 0, "id": "c0", "type": "function", **frag}]})
            yield {"object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, {}


def _body(shape, declared, stream):
    body = {**BASE, "stream": stream}
    if declared == "empty":
        body["tools" if shape == "modern" else "functions"] = []
    elif declared == "nonempty":
        body["tools" if shape == "modern" else "functions"] = (
            [TOOL] if shape == "modern" else [TOOL["function"]])
    return body


@pytest.mark.parametrize("shape,transport,declared,response", CASES)
def test_an_invalid_call_never_reaches_the_client(tmp_path, shape, transport, declared, response):
    up = Planted(shape, response)
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json=_body(shape, declared, transport == "stream"))
    blob = r.text
    # The verdict is DERIVED, not carried by the axis: a match-shaped name is only
    # valid when something was actually declared.
    if response == "match_shaped" and declared == "nonempty":
        assert r.status_code == 200, blob
        # A fragmented name arrives AS FRAGMENTS -- the client assembles them, which is
        # correct SSE. My first assertion looked for the joined string and failed on the
        # stream cells, on main as well as here: the matrix caught a defect in itself,
        # which is what asserting a property rather than a transcript buys.
        if transport == "stream":
            parts = [json.loads(l[6:]) for l in blob.splitlines()
                     if l.startswith("data: ") and l[6:] != "[DONE]"]
        else:
            parts = [r.json()]
        emitted = ""
        for part in parts:
            for c in part.get("choices") or []:
                piece = c.get("delta") or c.get("message") or {}
                emitted += (piece.get("function_call") or {}).get("name", "")
                for tc in piece.get("tool_calls") or []:
                    emitted += (tc.get("function") or {}).get("name", "")
        assert emitted == "get_weather", f"declared call not delivered intact: {emitted!r}\n{blob[:300]}"
        return
    # undeclared or malformed, on any path, with any declared set: never delivered.
    # THE INVARIANT. Everything below is about HOW the refusal is expressed; this is the
    # thing that must hold in every cell, and it is checked before any of that.
    bad = _name_for(response) or "not-an-object"
    assert bad not in blob, (
        f"{response} call reached the client on {transport}/{shape}/{declared}:\n{blob[:400]}")

    # HOW it refuses depends on whether bytes had already crossed, and that is a fact
    # about the transport rather than a second invariant.
    #
    # `Planted` emits content FIRST on the stream path, and a request declaring no tool
    # parameter is not buffered -- deliberately: every returned call is invalid there
    # whatever its spelling, so the decision needs no assembly and ordinary chat keeps its
    # latency. Prose is therefore already delivered when the call arrives, headers are
    # long gone, and a status code is no longer available to say anything with.
    #
    # The 2026-09-19 disposition: an explicit in-band SSE error object, then
    # stop -- no `[DONE]`, no terminal `finish_reason: stop`. Silent termination is what
    # the official clients accept as a normal completion (`openai-python` 3.13.0 ends the
    # iterator with `finish_reason=None` and raises nothing), so it would hand a harness a
    # compromised answer that looks finished.
    post_prose = transport == "stream" and declared == "absent"
    if post_prose:
        assert r.status_code == 200, f"headers were already sent; got {r.status_code}\n{blob[:300]}"
        frames = [l[6:] for l in blob.splitlines() if l.startswith("data: ")]
        assert frames and "[DONE]" not in frames, (
            "an error frame after `[DONE]` is unreachable: both official SDKs check the "
            f"terminator first and return.\n{blob[:400]}")
        err = json.loads(frames[-1]).get("error") or {}
        assert err.get("code") == "upstream_tool_call_contract_violation", (
            f"the published category for this event, reused rather than renamed\n{blob[:400]}")
        assert not any('"finish_reason": "stop"' in f for f in frames), \
            "a normal terminal reason asserts clean completion for a compromised response"
        return

    assert r.status_code == 502, f"expected a contract-violation refusal, got {r.status_code}"
    assert r.json()["error"]["code"] == "upstream_tool_call_contract_violation", blob
