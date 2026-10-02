"""The forced-call repair sends a backend thinking switch only to the backend that has one.

`chat_template_kwargs.enable_thinking` is a Qwen chat-template field. Under the portable
default (`passthrough`) the repair must not invent it; a caller's own value still passes.
"""
from chord import forced_call

PARAMS = {"tools": [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}}}}],
    "tool_choice": {"type": "function", "function": {"name": "lookup"}}}
BODY = {"model": "m", "messages": [{"role": "user", "content": "find it"}]}


def _plan():
    p = forced_call.plan(PARAMS)
    assert p is not None
    return p


def test_passthrough_adds_no_backend_thinking_field():
    out = forced_call.apply(dict(BODY), _plan(), "passthrough")
    assert "chat_template_kwargs" not in out


def test_passthrough_keeps_a_callers_own_template_kwargs_untouched():
    out = forced_call.apply({**BODY, "chat_template_kwargs": {"x": 1}}, _plan(), "passthrough")
    assert out["chat_template_kwargs"] == {"x": 1}


def test_qwen_mode_turns_thinking_off_for_the_schema_pass():
    out = forced_call.apply({**BODY, "chat_template_kwargs": {"x": 1}}, _plan(), "qwen_chat_template")
    assert out["chat_template_kwargs"] == {"x": 1, "enable_thinking": False}
