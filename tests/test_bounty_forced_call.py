"""Bug-bounty probe (testing the #216/#230 forced-call repair, 2026-09-17).

The repair makes a SECOND upstream call when the first reply lacks the required
tool call. Attacks here:

  1. usage accounting: two upstream calls, and the reply reports one.
  2. the repair pass carries the caller's sampling params (temperature, seed,
     max_completion_tokens) onto a call whose answer must be an exact JSON object.
  3. a function with no declared parameters accepts any arguments object.

Every test asserts its control first: the unforced path, and the forced path that
needs no repair, must behave before any claim is made about the repair.
"""
import json


from test_skeleton import FakeUpstream, make

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "current weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
NO_PARAMS = {"type": "function", "function": {"name": "ping", "description": "ping"}}


class RefusingUpstream(FakeUpstream):
    """Answers prose first (the vLLM behaviour #216 exists for), then obeys a
    json_schema. Records every body so the two passes can be told apart."""

    def __init__(self, repaired='{"name": "get_weather", "arguments": {"city": "Paris"}}', **kw):
        super().__init__(**kw)
        self.repaired = repaired

    async def complete(self, body):
        self.bodies.append(body)
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return ({"choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": self.repaired}}],
                     "usage": {"prompt_tokens": 40, "completion_tokens": 11, "total_tokens": 51}},
                    {"model-group": "example/chat"})
        return ({"choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": "Hello! How can I help?"}}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}},
                {"model-group": "example/chat"})


def _post(client, **extra):
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "Hello!"}], **extra}
    return client.post("/v1/chat/completions", json=body)


def _passes(up):
    """(first pass body, repair pass body) by response_format, not by order."""
    schema = [b for b in up.bodies if (b.get("response_format") or {}).get("type") == "json_schema"]
    plain = [b for b in up.bodies if b not in schema]
    return plain, schema


def test_control_no_force_makes_one_call_and_reports_its_usage(tmp_path):
    up = RefusingUpstream()
    _, client = make(tmp_path, upstream=up)
    r = _post(client)
    assert r.status_code == 200, r.text
    plain, schema = _passes(up)
    assert len(plain) == 1 and not schema, "control: an unforced turn never reaches the repair"
    assert r.json()["usage"]["completion_tokens"] == 7


def test_control_a_forced_call_the_model_makes_itself_is_never_repaired(tmp_path):
    class Obliging(FakeUpstream):
        async def complete(self, body):
            self.bodies.append(body)
            return ({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "call_1", "type": "function",
                                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]}}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 9, "total_tokens": 109}},
                    {"model-group": "example/chat"})
    up = Obliging()
    _, client = make(tmp_path, upstream=up)
    r = _post(client, tools=[WEATHER], tool_choice="required")
    assert r.status_code == 200, r.text
    plain, schema = _passes(up)
    assert not schema, "control: a real call must not trigger a repair pass"
    assert r.json()["usage"]["completion_tokens"] == 9


def test_repair_reports_only_the_second_pass_usage(tmp_path):
    """The finding: two upstream calls, one of them billed to the caller."""
    up = RefusingUpstream()
    _, client = make(tmp_path, upstream=up)
    r = _post(client, tools=[WEATHER], tool_choice="required")
    assert r.status_code == 200, r.text
    plain, schema = _passes(up)
    assert len(plain) == 1 and len(schema) == 1, "the repair really did make a second call"

    spent = {"prompt_tokens": 100 + 40, "completion_tokens": 7 + 11}
    reported = r.json()["usage"]
    assert reported["prompt_tokens"] == spent["prompt_tokens"] and reported["completion_tokens"] == spent["completion_tokens"], (
        f"two upstream passes consumed {spent}, the reply reports {reported}: the first "
        "pass's tokens are dropped, so usage under-reports what the turn actually cost")


def test_repair_pass_does_not_inherit_sampling_params_meant_for_prose(tmp_path):
    """max_completion_tokens sized for an answer can truncate the JSON object the
    repair must return; a seed/temperature chosen for prose is not chosen for it."""
    up = RefusingUpstream()
    _, client = make(tmp_path, upstream=up)
    r = _post(client, tools=[WEATHER], tool_choice="required",
              max_completion_tokens=8, temperature=1.9, seed=42)
    assert r.status_code == 200, r.text
    plain, schema = _passes(up)
    assert schema, "the repair pass ran"
    leaked = {k: schema[0][k] for k in ("max_completion_tokens", "temperature", "seed") if k in schema[0]}
    assert not leaked, (
        f"the repair pass carries the caller's prose sampling params {leaked}; "
        "max_completion_tokens in particular can truncate the constrained JSON object")


def test_a_function_with_no_parameters_does_not_accept_arbitrary_arguments(tmp_path):
    """_open() widens a missing `parameters` to a bare {'type': 'object'}, so the
    constrained reply may carry arguments the caller never declared."""
    up = RefusingUpstream(repaired='{"name": "ping", "arguments": {"unexpected": "field"}}')
    _, client = make(tmp_path, upstream=up)
    r = _post(client, tools=[NO_PARAMS], tool_choice="required")
    assert r.status_code == 200, r.text
    args = json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {}, (
        f"'ping' declares no parameters, but the repaired call carries {args}: "
        "to_message accepts any object as arguments and never checks them against "
        "the declared schema")


class TruncatingUpstream(RefusingUpstream):
    """Honours max_completion_tokens the way a backend does: the constrained reply
    is cut short. This is what the inherited param BUYS the caller."""

    async def complete(self, body):
        if (body.get("response_format") or {}).get("type") == "json_schema":
            limit = body.get("max_completion_tokens")
            if limit is not None and limit < 12:
                self.bodies.append(body)
                return ({"choices": [{"index": 0, "finish_reason": "length",
                                      "message": {"role": "assistant", "content": '{"name": "get_weather", "argum'}}],
                         "usage": {"prompt_tokens": 40, "completion_tokens": limit, "total_tokens": 40 + limit}},
                        {"model-group": "example/chat"})
        return await super().complete(body)


def test_control_repair_succeeds_when_the_budget_is_not_tiny(tmp_path):
    up = TruncatingUpstream()
    _, client = make(tmp_path, upstream=up)
    r = _post(client, tools=[WEATHER], tool_choice="required", max_completion_tokens=64)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_a_small_max_completion_tokens_makes_the_repair_fail_into_the_502_it_prevents(tmp_path):
    """The consequence of inheriting the caller's budget: the repair's JSON object is
    truncated, to_message rejects it, and the S03 check returns the exact 502 that
    #216/#230 exist to remove."""
    up = TruncatingUpstream()
    _, client = make(tmp_path, upstream=up)
    r = _post(client, tools=[WEATHER], tool_choice="required", max_completion_tokens=8)
    assert r.status_code == 200, (
        f"the repair inherited max_completion_tokens=8, truncated its JSON object and "
        f"failed into {r.status_code} {r.json().get('error', {}).get('code')!r} -- the "
        "very contract violation the repair was added to prevent")
