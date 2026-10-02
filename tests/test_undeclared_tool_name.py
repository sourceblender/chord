"""#289: a returned tool_call naming a function the request never declared must refuse.

Measured twice in 222 byte-identical firings, with two names that differ IN KIND --
`weather` (truncation-shaped) and `weather_getCurrent_2604` (convention-shaped, matching
nothing in the tree). That pair is why the check is **exact membership in the request's
own declared set** and nothing cleverer: an alias map or fuzzy repair catches the first
and waves the second straight through.

Falsifiers 1-4 were specified before the implementation froze; 5 (content before
the call) proves call-only buffering insufficient. The positive controls are
load-bearing: every refusal-direction test above would pass a guard
that refuses everything.

2026-09-19.
"""

import pytest

from test_skeleton import FakeUpstream, make

CHAT = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "w",
                                            "parameters": {"type": "object", "properties": {}}}}
BOOK = {"type": "function", "function": {"name": "book_table", "description": "b",
                                         "parameters": {"type": "object", "properties": {}}}}


class Calls(FakeUpstream):
    """Replies with exactly the tool calls it was built with, plus optional content."""

    def __init__(self, names, content=None, legacy=False):
        super().__init__()
        self.names, self.content, self.legacy = names, content, legacy

    async def complete(self, body):
        import asyncio
        await asyncio.sleep(0)
        self.bodies.append(body)
        msg = {"role": "assistant", "content": self.content}
        if self.legacy:
            msg["function_call"] = {"name": self.names[0], "arguments": "{}"}
        else:
            msg["tool_calls"] = [{"id": f"c{i}", "type": "function",
                                  "function": {"name": n, "arguments": "{}"}}
                                 for i, n in enumerate(self.names)]
        return ({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
                {"model-api-base": "http://x/v1", "model-group": "example/chat"})


def _post(tmp_path, up, **extra):
    _, client = make(tmp_path, up)
    return client.post("/v1/chat/completions", json={**CHAT, **extra})


def _refused(r):
    return (r.status_code == 502
            and r.json()["error"]["code"] == "upstream_tool_call_contract_violation")


# ---- the two observed names, as NAMED regressions (Falsifier 4) --------------------------

@pytest.mark.parametrize("name", ["weather", "weather_getCurrent_2604"])
def test_both_observed_undeclared_names_refuse(tmp_path, name):
    """They exist to fail together or not at all: `weather` is the one any cheap fix
    catches, `weather_getCurrent_2604` is the one that kills the cheap fix."""
    r = _post(tmp_path, Calls([name]), tools=[WEATHER])
    assert _refused(r), r.text


# ---- Falsifier 1: the declared set comes from THIS request ------------------------------

def test_the_declared_set_is_read_from_this_request_not_remembered(tmp_path):
    """My first analysis table compared every returned name to a remembered
    `get_weather` and flagged `book_table` as undeclared -- a fabricated positive. A
    validator resolving the set from a constant, a cache or the previous turn
    reproduces that at runtime and REJECTS VALID CALLS."""
    up = Calls(["book_table"])
    _, client = make(tmp_path, up)
    first = client.post("/v1/chat/completions", json={**CHAT, "tools": [WEATHER]})
    assert _refused(first), "book_table is not declared by the FIRST request"
    second = client.post("/v1/chat/completions", json={**CHAT, "tools": [BOOK]})
    assert second.status_code == 200, f"second request declares book_table: {second.text}"


# ---- Falsifier 2: empty or absent tools means an empty membership set -------------------

@pytest.mark.parametrize("extra", [{}, {"tools": []}], ids=["tools-absent", "tools-empty"])
def test_a_call_with_no_declared_tools_refuses(tmp_path, extra):
    """Nothing is a member of an empty set. The natural implementation reads `tools` and
    SKIPS validation when absent, which is the exact inverse of correct -- and is what I
    wrote first (`if declared:`), exactly as a review predicted before seeing the code."""
    r = _post(tmp_path, Calls(["get_weather"]), **extra)
    assert _refused(r), r.text


# ---- Falsifier 3: exact membership, no normalisation -----------------------------------

@pytest.mark.parametrize("name", ["Get_Weather", "get_weather ", " get_weather", "GET_WEATHER"])
def test_membership_is_case_and_whitespace_exact(tmp_path, name):
    """Any normalising comparison is a fuzzy match wearing a different name, and the
    measurement forbids fuzzy."""
    r = _post(tmp_path, Calls([name]), tools=[WEATHER])
    assert _refused(r), f"{name!r} is not `get_weather`: {r.text}"


# ---- shape coverage: legacy functions, multi-call --------------------------------

def test_the_legacy_function_call_shape_is_validated_too(tmp_path):
    r = _post(tmp_path, Calls(["weather"], legacy=True), functions=[WEATHER["function"]])
    assert _refused(r), r.text


def test_a_mixed_response_refuses_as_a_whole(tmp_path):
    """One valid call does not license an invalid sibling."""
    r = _post(tmp_path, Calls(["get_weather", "weather_getCurrent_2604"]), tools=[WEATHER])
    assert _refused(r), r.text


# ---- POSITIVE CONTROLS: a guard that refuses everything passes all the above --

def test_a_declared_call_still_passes(tmp_path):
    r = _post(tmp_path, Calls(["get_weather"]), tools=[WEATHER])
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_content_before_a_declared_call_still_passes(tmp_path):
    """a probe measured this shape live: content and a tool call in one response. It is
    legitimate and must not be collateral."""
    r = _post(tmp_path, Calls(["get_weather"], content="I'll check both cities."), tools=[WEATHER])
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "I'll check both cities."


def test_a_prose_only_response_still_passes(tmp_path):
    r = _post(tmp_path, FakeUpstream(), tools=[WEATHER])
    assert r.status_code == 200, r.text


def test_a_prose_only_response_with_no_tools_still_passes(tmp_path):
    """The no-tools path must not become a refusal for ordinary chat."""
    r = _post(tmp_path, FakeUpstream())
    assert r.status_code == 200, r.text
