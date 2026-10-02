"""S03 (red team pass 1, 2026-09-15). Routing a forced call to chat was not
enough: live on 6185afd, tool_choice required / named and a legacy
function_call came back as prose, and a direct call to vLLM :8113 did the same
(the matched control). Whatever the backend does, a forced call either
returns a valid declared call, or the client gets a 502 in our envelope. Prose
alone is never a success. Streams are read whole before any header, because a
first tool-call delta does not prove the call finishes valid."""
import json

import openai
import pytest

from chord.server import FORCED_CALL_CODE
from test_progress import last_trace
from test_skeleton import FakeUpstream, make

BOOK = {"type": "function", "function": {"name": "book_table", "parameters": {"type": "object"}}}
TIME = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}
FORCES = {
    "required": ({"tools": [BOOK, TIME], "tool_choice": "required"}, "tool_choice"),
    "named": ({"tools": [BOOK, TIME], "tool_choice": {"type": "function", "function": {"name": "book_table"}}}, "tool_choice"),
    "legacy": ({"functions": [BOOK["function"], TIME["function"]], "function_call": {"name": "book_table"}}, "function_call"),
}


class Scripted(FakeUpstream):
    """Her model's answer: content, then calls as (name, arguments), then a
    finish reason. Streamed arguments arrive in two fragments."""

    def __init__(self, content=None, calls=(), finish="stop", legacy=False):
        super().__init__()
        self.content, self.calls, self.finish, self.legacy = content, list(calls), finish, legacy

    async def complete(self, body):
        data, dep = await super().complete(body)
        msg = {"role": "assistant", "content": self.content}
        if self.calls and self.legacy:
            msg["function_call"] = {"name": self.calls[0][0], "arguments": self.calls[0][1]}
        elif self.calls:
            msg["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
                                 for i, (n, a) in enumerate(self.calls)]
        data["choices"][0]["message"], data["choices"][0]["finish_reason"] = msg, self.finish
        return data, dep

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        if self.content:
            yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": self.content}}]}, {}
        for i, (name, args) in enumerate(self.calls):
            half = len(args) // 2
            if self.legacy:
                yield {"choices": [{"index": 0, "delta": {"function_call": {"name": name, "arguments": args[:half]}}}]}, {}
                yield {"choices": [{"index": 0, "delta": {"function_call": {"arguments": args[half:]}}}]}, {}
            else:
                yield {"choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": i, "id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args[:half]}}]}}]}, {}
                yield {"choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": i, "function": {"arguments": args[half:]}}]}}]}, {}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": self.finish}]}, {}


def test_null_stream_options_do_not_crash_a_forced_repair(tmp_path):
    deps, client = make(tmp_path, Scripted(**FAILURES["prose only"]))
    r = client.post("/v1/chat/completions", json={
        "model": "chord-1-poly", "stream": True, "stream_options": None,
        "messages": [{"role": "user", "content": "Hello!"}],
        "tools": [BOOK], "tool_choice": "required",
    })
    assert r.status_code != 500, r.text
    assert "Internal Server Error" not in r.text


def post(client, stream, force):
    extra, _ = FORCES[force]
    return client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
                                                     "messages": [{"role": "user", "content": "Hello!"}], **extra})


# Each way the answer can fail a force. The two red proofs are the second and third.
FAILURES = {
    "prose only": dict(content="Hello! How can I help?", finish="stop"),
    "prose then tool_calls finish, no call": dict(content="Hello! 👋", finish="tool_calls"),
    "call with cut-off arguments": dict(calls=[("book_table", '{"people": ')], finish="tool_calls"),
    "call to an undeclared function": dict(calls=[("delete_everything", "{}")], finish="tool_calls"),
    "empty function name": dict(calls=[("", "{}")], finish="tool_calls"),
}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("force", list(FORCES))
@pytest.mark.parametrize("failure", list(FAILURES))
def test_a_force_the_answer_does_not_satisfy_is_a_502_before_any_byte(tmp_path, failure, force, stream):
    deps, client = make(tmp_path, Scripted(**FAILURES[failure], legacy=force == "legacy"))
    r = post(client, stream, force)
    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")     # not an SSE 200
    assert "data:" not in r.text
    err = r.json()["error"]
    assert err["code"] == FORCED_CALL_CODE and err["type"] == "server_error" and err["param"] == FORCES[force][1]
    assert "Hello" not in err["message"] and "book_table" not in err["message"]   # our sentence, not hers
    t = last_trace(deps.settings)
    assert t["forced_call_violation"] and t["forced_call_param"] == FORCES[force][1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("force", ["named", "legacy"])
def test_a_named_force_answered_with_another_declared_function_is_a_502(tmp_path, force, stream):
    deps, client = make(tmp_path, Scripted(calls=[("get_time", "{}")], finish="tool_calls", legacy=force == "legacy"))
    r = post(client, stream, force)
    assert r.status_code == 502 and r.json()["error"]["code"] == FORCED_CALL_CODE
    assert "not the forced" in last_trace(deps.settings)["forced_call_violation"]


@pytest.mark.parametrize("stream", [False, True])
def test_required_is_satisfied_by_any_declared_function(tmp_path, stream):
    deps, client = make(tmp_path, Scripted(calls=[("get_time", "{}")], finish="tool_calls"))
    assert post(client, stream, "required").status_code == 200


def calls_in(r, stream, legacy):
    key = "function_call" if legacy else "tool_calls"
    if not stream:
        msg = r.json()["choices"][0]["message"]
        return msg.get("content") or "", msg.get(key)
    chunks = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
    deltas = [c.get("delta") or {} for k in chunks for c in k.get("choices") or []]
    return "".join(d.get("content") or "" for d in deltas), [d[key] for d in deltas if d.get(key)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("force", list(FORCES))
@pytest.mark.parametrize("content", [None, "", "Booking that now."])
def test_a_valid_forced_call_comes_back_with_any_words_beside_it(tmp_path, force, stream, content):
    legacy = force == "legacy"
    deps, client = make(tmp_path, Scripted(content=content, calls=[("book_table", '{"people": 2}')],
                                           finish="function_call" if legacy else "tool_calls", legacy=legacy))
    r = post(client, stream, force)
    assert r.status_code == 200, r.text
    text, calls = calls_in(r, stream, legacy)
    assert calls and text == (content or "")
    if stream:
        assert r.text.rstrip().endswith("data: [DONE]")
        # The whole forced stream was read before the response began.
        assert last_trace(deps.settings)["forced_stream_buffered"] is True
        args = "".join(c["arguments"] if legacy else c[0]["function"].get("arguments", "") for c in calls)
        assert json.loads(args) == {"people": 2}                          # replayed exactly as sent


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "none", None])
def test_an_unforced_answer_in_prose_is_untouched(tmp_path, stream, choice):
    """Control: only a force is enforced."""
    extra = {"tools": [BOOK]} | ({"tool_choice": choice} if choice else {})
    deps, client = make(tmp_path, Scripted(content="Hello! How can I help?"))
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
                                                  "messages": [{"role": "user", "content": "Hello!"}], **extra})
    assert r.status_code == 200 and "forced_call_violation" not in last_trace(deps.settings)
    if stream:
        assert "forced_stream_buffered" not in last_trace(deps.settings)


@pytest.mark.parametrize("stream", [False, True])
def test_official_sdk_sees_a_server_error_not_prose(tmp_path, stream):
    deps, client = make(tmp_path, Scripted(content="Hello! 👋", finish="tool_calls"))
    sdk = openai.OpenAI(api_key="test", base_url="http://testserver/v1", http_client=client, max_retries=0)
    with pytest.raises(openai.APIStatusError) as caught:
        out = sdk.chat.completions.create(model="chord-1-poly", messages=[{"role": "user", "content": "Hello!"}],
                                          tools=[BOOK], tool_choice="required", stream=stream)
        if stream:
            list(out)
    assert caught.value.status_code == 502 and caught.value.body["code"] == FORCED_CALL_CODE


# the block on 206b75f: the allowed_tools subset, custom forcing, and
# malformed forcing objects.
def allowed(mode, *tools):
    return {"type": "allowed_tools", "allowed_tools": {"mode": mode, "tools": list(tools)}}


FN = lambda name: {"type": "function", "function": {"name": name}}
CUSTOM = {"type": "custom", "custom": {"name": "grammar"}}


def ask(client, stream, **extra):
    return client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
                                                     "messages": [{"role": "user", "content": "Hello!"}], **extra})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("called,status", [("book_table", 200), ("get_time", 502)])
def test_allowed_tools_required_holds_to_its_subset(tmp_path, stream, called, status):
    """ "one or more of the allowed tools" (pinned schema), not any declared one."""
    deps, client = make(tmp_path, Scripted(calls=[(called, "{}")], finish="tool_calls"))
    r = ask(client, stream, tools=[BOOK, TIME], tool_choice=allowed("required", FN("book_table")))
    assert r.status_code == status
    if status == 502:
        assert r.json()["error"]["code"] == FORCED_CALL_CODE
        assert last_trace(deps.settings)["forced_call_allowed"] == ["book_table"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("answer,status", [
    (dict(content="Hello!"), 200),                                            # no call is fine under auto
    (dict(calls=[("book_table", "{}")], finish="tool_calls"), 200),           # an allowed call
    (dict(calls=[("get_time", "{}")], finish="tool_calls"), 502),             # declared, but excluded
], ids=["prose", "allowed-call", "excluded-call"])
def test_allowed_tools_auto_is_a_subset_but_not_a_demand(tmp_path, stream, answer, status):
    """the addendum on #174: mode auto needs no call, but any call must be allowed."""
    deps, client = make(tmp_path, Scripted(**answer))
    r = ask(client, stream, tools=[BOOK, TIME], tool_choice=allowed("auto", FN("book_table")))
    assert r.status_code == status
    t = last_trace(deps.settings)
    if status == 502:
        assert t["forced_call_required"] is False and t["forced_call_allowed"] == ["book_table"]
    else:
        assert "forced_call_violation" not in t


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", [{"type": "custom", "custom": {"name": "grammar"}}, allowed("required", FN("book_table"), CUSTOM),
                                    allowed("auto", FN("book_table"), CUSTOM)],
                         ids=["custom", "allowed-required-with-custom", "allowed-auto-with-custom"])
def test_a_custom_force_is_refused_not_weakened(tmp_path, stream, choice):
    up = Scripted(content="Hello!")
    deps, client = make(tmp_path, up)
    r = ask(client, stream, tools=[BOOK, CUSTOM], tool_choice=choice)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_parameter" and err["param"] == "tool_choice"
    assert up.bodies == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("extra,param", [
    ({"tool_choice": {"type": "function", "function": "book_table"}}, "tool_choice"),
    ({"tool_choice": {"type": "function", "function": {}}}, "tool_choice"),
    ({"tool_choice": {"type": "function"}}, "tool_choice"),
    ({"tool_choice": {"type": "allowed_tools", "allowed_tools": "required"}}, "tool_choice"),
    ({"tool_choice": allowed("sometimes", FN("book_table"))}, "tool_choice"),
    ({"tool_choice": allowed("required")}, "tool_choice"),
    ({"tool_choice": allowed("required", "book_table")}, "tool_choice"),
    ({"tool_choice": {"type": "nope"}}, "tool_choice"),
    ({"tool_choice": "sometimes"}, "tool_choice"),
    ({"tool_choice": 5}, "tool_choice"),
    ({"function_call": {"arguments": "{}"}}, "function_call"),
    ({"function_call": "required"}, "function_call"),
    ({"function_call": 5}, "function_call"),
])
def test_a_malformed_force_is_a_400_not_a_crash(tmp_path, stream, extra, param):
    up = Scripted(content="Hello!")
    deps, client = make(tmp_path, up)
    r = ask(client, stream, tools=[BOOK], functions=[BOOK["function"]], **extra)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["code"] == "invalid_value" and err["param"] == param
    assert up.bodies == []


# ae36d9e: a malformed declared tool crashed the answer check after the
# model ran. Declarations are checked whole before anything is forwarded, and a
# forced name must be one the request declares.
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("extra,param", [
    ({"tools": [{"type": "function", "function": "book_table"}], "tool_choice": "required"}, "tools"),
    ({"tools": [{"type": "function", "function": None}], "tool_choice": "required"}, "tools"),
    ({"tools": [{"type": "function", "function": {}}]}, "tools"),
    ({"tools": [{"type": "nope"}]}, "tools"),
    ({"tools": ["book_table"]}, "tools"),
    ({"tools": {"book_table": {}}}, "tools"),
    ({"tools": [{"type": "custom", "custom": "grammar"}]}, "tools"),
    ({"functions": [{"description": "no name"}], "function_call": "auto"}, "functions"),
    ({"functions": "book_table"}, "functions"),
    ({"tools": [BOOK], "tool_choice": FN("get_time")}, "tool_choice"),
    ({"tools": [BOOK], "tool_choice": allowed("auto", FN("get_time"))}, "tool_choice"),
    ({"functions": [BOOK["function"]], "function_call": {"name": "get_time"}}, "function_call"),
    ({"tool_choice": FN("book_table")}, "tool_choice"),
])
def test_a_malformed_or_undeclared_declaration_is_a_400_before_the_model(tmp_path, stream, extra, param):
    up = Scripted(content="Hello!")
    deps, client = make(tmp_path, up)
    r = ask(client, stream, **extra)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["code"] == "invalid_value" and err["param"] == param
    assert up.bodies == []


# 64380fc: the answer is untrusted upstream output. A malformed call or
# delta under a valid force is the contract 502, never an internal crash, and a
# stream is still refused before any SSE byte.
class Raw(FakeUpstream):
    """Her model's answer exactly as given: a message, or raw stream deltas."""

    def __init__(self, message=None, deltas=()):
        super().__init__()
        self.message, self.deltas = message, list(deltas)

    async def complete(self, body):
        data, dep = await super().complete(body)
        data["choices"][0]["message"], data["choices"][0]["finish_reason"] = self.message, "tool_calls"
        return data, dep

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for delta in self.deltas:
            yield {"choices": [{"index": 0, "delta": delta}]}, {}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, {}


MESSAGES = {
    "function is a string": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": "book_table"}]},
    "function is null": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": None}]},
    "name is a number": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": 5, "arguments": "{}"}}]},
    "arguments are an object": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "book_table", "arguments": {}}}]},
    "tool_calls is an object": {"role": "assistant", "tool_calls": {"0": "book_table"}},
    "a call is a string": {"role": "assistant", "tool_calls": ["book_table"]},
    "one good, one malformed": {"role": "assistant", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "book_table", "arguments": "{}"}},
        {"id": "b", "type": "function", "function": "book_table"}]},
}
GOOD = {"index": 0, "id": "a", "type": "function", "function": {"name": "book_table", "arguments": "{}"}}
DELTAS = {
    "function is a string": [{"tool_calls": [{"index": 0, "id": "x", "type": "function", "function": "book_table"}]}],
    "name fragment is a number": [{"tool_calls": [{"index": 0, "function": {"name": 5, "arguments": "{}"}}]}],
    "arguments fragment is an object": [{"tool_calls": [{"index": 0, "function": {"name": "book_table", "arguments": {}}}]}],
    "index is a string": [{"tool_calls": [{"index": "0", "function": {"name": "book_table", "arguments": "{}"}}]}],
    "index is a bool": [{"tool_calls": [{"index": True, "function": {"name": "book_table", "arguments": "{}"}}]}],
    "tool_calls is an object": [{"tool_calls": {"index": 0}}],
    "a call is a string": [{"tool_calls": ["book_table"]}],
    "good call then a malformed fragment": [{"tool_calls": [GOOD]}, {"tool_calls": [{"index": 0, "function": ["{}"]}]}],
}


@pytest.mark.parametrize("shape", list(MESSAGES))
def test_a_malformed_upstream_call_is_the_contract_502_not_a_crash(tmp_path, shape):
    deps, client = make(tmp_path, Raw(message=MESSAGES[shape]))
    r = post(client, False, "required")
    assert r.status_code == 502 and r.json()["error"]["code"] == FORCED_CALL_CODE
    assert last_trace(deps.settings)["forced_call_violation"] == "a malformed call"


@pytest.mark.parametrize("shape", list(DELTAS))
def test_a_malformed_streamed_call_is_refused_before_any_byte(tmp_path, shape):
    deps, client = make(tmp_path, Raw(deltas=DELTAS[shape]))
    r = post(client, True, "required")
    assert r.status_code == 502 and "data:" not in r.text
    assert r.json()["error"]["code"] == FORCED_CALL_CODE
    assert last_trace(deps.settings)["forced_call_violation"] == "a malformed call"


@pytest.mark.parametrize("stream", [False, True])
def test_required_with_nothing_declared_is_a_400_before_the_model(tmp_path, stream):
    up = Scripted(content="Hello!")
    deps, client = make(tmp_path, up)
    r = ask(client, stream, tool_choice="required")
    assert r.status_code == 400 and r.json()["error"]["param"] == "tool_choice"
    assert up.bodies == []


# the live S03 rerun on 5d9f33a: allowed_tools came back as a bare 500,
# because neither backend accepts it (vLLM 400, LiteLLM 500; the control).
# It now goes upstream as the allowed functions with the mode as tool_choice.
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["required", "auto"])
def test_allowed_tools_goes_upstream_in_the_shape_the_backend_accepts(tmp_path, stream, mode):
    up = Scripted(calls=[("book_table", "{}")], finish="tool_calls")
    deps, client = make(tmp_path, up)
    r = ask(client, stream, tools=[BOOK, TIME], tool_choice=allowed(mode, FN("book_table")))
    assert r.status_code == 200, r.text
    sent = up.bodies[-1]
    assert sent["tools"] == [BOOK] and sent["tool_choice"] == mode        # never the allowed_tools object
    t = last_trace(deps.settings)
    assert t["router"] == "skipped_client_constraint"                     # routing read the original


@pytest.mark.parametrize("stream", [False, True])
def test_the_answer_is_still_checked_against_the_clients_allowed_set(tmp_path, stream):
    """Even if the model names a function it was not sent, the check refuses it."""
    up = Scripted(calls=[("get_time", "{}")], finish="tool_calls")
    deps, client = make(tmp_path, up)
    r = ask(client, stream, tools=[BOOK, TIME], tool_choice=allowed("required", FN("book_table")))
    assert up.bodies[-1]["tools"] == [BOOK]
    assert r.status_code == 502 and r.json()["error"]["code"] == FORCED_CALL_CODE


# --- #216: the backend doesn't force; a missing forced call is repaired through a JSON schema ---------
# Live 2026-09-17: "Hello!" under `required` came back with no call 11 of 11 against vLLM directly; carried
# as a json_schema the same ask gave a valid call 12 of 12, choosing the right function of two.

class ProseThenSchema(Scripted):
    """Her model as it is live: prose on the tool path, and a schema-shaped object when constrained."""

    def __init__(self, reply, **kw):
        super().__init__(content="Hello! How can I help?", finish="stop", **kw)
        self.reply = reply

    async def complete(self, body):
        if (body.get("response_format") or {}).get("type") == "json_schema":
            data, dep = await FakeUpstream.complete(self, body)
            data["choices"][0]["message"] = {"role": "assistant", "content": self.reply}
            return data, dep
        return await super().complete(body)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("force", list(FORCES))
def test_a_missing_forced_call_is_repaired_through_a_schema(tmp_path, force, stream):
    up = ProseThenSchema('{"name": "book_table", "arguments": {"people": 2}}', legacy=force == "legacy")
    deps, client = make(tmp_path, up)
    r = post(client, stream, force)
    assert r.status_code == 200, r.text
    content, got = calls_in(r, stream, force == "legacy")
    assert content == "" and "Hello" not in r.text                      # the prose she streamed first is void
    flat = json.dumps(got)
    assert '"book_table"' in flat and '{\\"people\\": 2}' in flat and flat.count('"name"') == 1   # exactly the repaired call
    repair = up.bodies[-1]
    assert not ({"tools", "tool_choice", "functions", "function_call"} & repair.keys())
    schema = repair["response_format"]["json_schema"]["schema"]
    names = {a["properties"]["name"]["const"] for a in (schema.get("anyOf") or [schema])}
    assert names == ({"book_table", "get_time"} if force == "required" else {"book_table"})   # a named force offers only its function
    assert last_trace(deps.settings)["forced_call_repaired"] is True


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply", ["Hello again!", '{"name": "delete_everything", "arguments": {}}', '{"name": "book_table"}', "{"])
def test_a_repair_that_is_not_a_declared_call_is_still_a_502(tmp_path, stream, reply):
    deps, client = make(tmp_path, ProseThenSchema(reply))
    r = post(client, stream, "required")
    assert r.status_code == 502 and r.json()["error"]["code"] == FORCED_CALL_CODE
    assert last_trace(deps.settings)["forced_call_repaired"] is False


@pytest.mark.parametrize("stream", [False, True])
def test_a_call_the_model_made_itself_is_never_repaired(tmp_path, stream):
    up = Scripted(calls=[("get_time", "{}")], finish="tool_calls")
    deps, client = make(tmp_path, up)
    assert post(client, stream, "required").status_code == 200
    assert len(up.bodies) == 1 and "forced_call_repaired" not in last_trace(deps.settings)


@pytest.mark.parametrize("schema, expected", [
    ({"type": "object", "properties": {"city": {"type": "string"}}}, {"city": "Paris", "extra": 1}),
    ({"type": "object", "properties": {"city": {"type": "string"}}, "additionalProperties": False}, {"city": "Paris"}),
])
def test_a_repaired_calls_arguments_follow_the_callers_own_schema(tmp_path, schema, expected):
    """#250: extras are legal unless the caller forbids them, so a permissive schema must
    keep them and only `additionalProperties: false` strips. Pinned so nobody makes it a blanket strip."""
    from chord.forced_call import Plan, to_message
    plan = Plan(functions=((("get_weather"), "", schema),), legacy=False)
    dropped: list[str] = []
    message, finish = to_message('{"name": "get_weather", "arguments": {"city": "Paris", "extra": 1}}', plan, dropped)
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == expected
    assert dropped == ([] if expected.get("extra") else ["extra"])


def test_a_repaired_call_carries_no_logprobs_from_the_discarded_pass(tmp_path):
    """The repaired choice kept the FIRST pass's logprobs -- numbers describing
    prose tokens the caller never receives (review 2026-09-22). The schema
    pass reports none, so the honest value is null."""
    class WithLogprobs(ProseThenSchema):
        async def complete(self, body):
            data, dep = await super().complete(body)
            if (body.get("response_format") or {}).get("type") != "json_schema":
                data["choices"][0]["logprobs"] = {"content": [{"token": "Hello", "logprob": -0.1}]}
            return data, dep

    up = WithLogprobs('{"name": "book_table", "arguments": {"people": 2}}')
    deps, client = make(tmp_path, up)
    r = post(client, False, "required")
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["message"]["tool_calls"]              # the repair happened
    assert choice["logprobs"] is None                   # ...without the discarded pass's numbers


# review 2026-09-24 B16: the repair pass nests the client's parameters under
# properties.arguments of a new root, so a pydantic-style "$ref": "#/$defs/X" pointed
# at a root that no longer had $defs and the constrained schema could not resolve.
PYDANTIC = {  # what pydantic v2's model_json_schema() emits for a nested model
    "$defs": {"Address": {"properties": {"city": {"title": "City", "type": "string"}},
                          "required": ["city"], "title": "Address", "type": "object"}},
    "properties": {"name": {"title": "Name", "type": "string"}, "home": {"$ref": "#/$defs/Address"},
                   "friends": {"type": "array", "items": {"$ref": "#"}}},
    "required": ["name", "home"], "title": "Person", "type": "object",
}
LEGACY_DEFS = {"definitions": {"Unit": {"enum": ["c", "f"], "type": "string"}},
               "properties": {"unit": {"$ref": "#/definitions/Unit"}}, "required": ["unit"], "type": "object"}


@pytest.mark.parametrize("tools", [
    [{"type": "function", "function": {"name": "add_person", "parameters": PYDANTIC}}],
    [{"type": "function", "function": {"name": "add_person", "parameters": PYDANTIC}},
     {"type": "function", "function": {"name": "weather", "parameters": LEGACY_DEFS}},
     {"type": "function", "function": {"name": "other", "parameters": {**LEGACY_DEFS, "definitions": {"Unit": {"type": "integer"}}}}}],
])
def test_the_repair_schema_resolves_the_clients_refs(tools):
    import jsonschema
    from referencing.exceptions import Unresolvable
    from chord import forced_call
    p = forced_call.plan({"tools": tools, "tool_choice": "required"})
    assert p is not None
    schema = forced_call.apply({"messages": [{"role": "user", "content": "hi"}]}, p)["response_format"]["json_schema"]["schema"]
    good = {"name": "add_person", "arguments": {"name": "Ava", "home": {"city": "Kyoto"},
                                                "friends": [{"name": "Ava", "home": {"city": "Oslo"}}]}}
    bad = {"name": "add_person", "arguments": {"name": "Ava", "home": {"city": 3}}}
    try:
        jsonschema.validate(good, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
        if len(tools) > 1:   # two clients' "Unit" definitions stay each its own
            jsonschema.validate({"name": "weather", "arguments": {"unit": "c"}}, schema)
            jsonschema.validate({"name": "other", "arguments": {"unit": 3}}, schema)
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({"name": "other", "arguments": {"unit": "c"}}, schema)
    except Unresolvable as exc:
        pytest.fail(f"a client $ref dangles in the repair schema: {exc}")
    assert tools[0]["function"]["parameters"] == PYDANTIC   # the caller's own schema is not mutated


def test_a_property_named_like_a_data_keyword_still_has_its_ref_relocated():
    from chord import forced_call
    params = {"$defs": {"X": {"type": "integer"}}, "type": "object",
              "properties": {"default": {"$ref": "#/$defs/X"}, "enum": {"const": {"$ref": "#/not-a-ref"}}}}
    p = forced_call.plan({"tools": [{"type": "function", "function": {"name": "f", "parameters": params}}],
                          "tool_choice": "required"})
    assert p is not None
    schema = forced_call.apply({"messages": []}, p)["response_format"]["json_schema"]["schema"]
    args = schema["properties"]["arguments"]
    assert schema["$defs"] == {"X": {"type": "integer"}} and "$defs" not in args
    assert args["properties"]["default"] == {"$ref": "#/$defs/X"}
    assert args["properties"]["enum"]["const"] == {"$ref": "#/not-a-ref"}   # instance data, untouched


# Copilot on #333: definition names are JSON Pointer tokens ("/" is "~1", "~" is "~0"),
# and a ref may name a whole definitions map ("#/$defs"), not only one entry in it.
ESCAPED = {"$defs": {"foo/bar": {"type": "integer"}, "a~b": {"type": "string"}},
           "definitions": {"Unit": {"enum": ["c", "f"]}},
           "properties": {"n": {"$ref": "#/$defs/foo~1bar"}, "s": {"$ref": "#/$defs/a~0b"},
                          "all": {"$ref": "#/definitions"}},
           "required": ["n", "s"], "type": "object"}


def _resolve(root, pointer):
    """Follow a local JSON Pointer the way a validator does."""
    import jsonschema
    resolver = jsonschema.validators.validator_for(root)(root)._resolver
    return resolver.lookup(pointer).contents


@pytest.mark.parametrize("many", [False, True])
def test_the_repair_schema_decodes_pointer_escapes_and_whole_map_refs(many):
    import jsonschema
    from referencing.exceptions import Unresolvable
    from chord import forced_call
    tools = [{"type": "function", "function": {"name": "f", "parameters": ESCAPED}}]
    if many:
        tools.append({"type": "function", "function": {"name": "g", "parameters": ESCAPED}})
    p = forced_call.plan({"tools": tools, "tool_choice": "required"})
    assert p is not None
    schema = forced_call.apply({"messages": []}, p)["response_format"]["json_schema"]["schema"]
    args = schema["anyOf"][0]["properties"]["arguments"] if many else schema["properties"]["arguments"]
    try:
        jsonschema.validate({"name": "f", "arguments": {"n": 1, "s": "x"}}, schema)
        for bad in ({"n": "1", "s": "x"}, {"n": 1, "s": 2}):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({"name": "f", "arguments": bad}, schema)
        whole = _resolve(schema, args["properties"]["all"]["$ref"])
    except Unresolvable as exc:
        pytest.fail(f"a client $ref dangles in the repair schema: {exc}")
    assert whole == {"Unit": {"enum": ["c", "f"]}}   # the map the client named, whole
    assert tools[0]["function"]["parameters"] == ESCAPED
