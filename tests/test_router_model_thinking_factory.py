"""Thinking is off for the router MODEL, not only at the call sites graph.py wraps.

#345 bound enable_thinking=False at the four graph.py router calls, and its AST
guard only parses graph.py for the literal `settings.router_model`. The search
specialist calls the same model by its registry name through `ctx.model(...)`
(specialists/search.py), so it kept thinking on: the 9B median with thinking is
6,739 ms against a 10 s query budget (qa/performance/BASELINE-2026-09-23.md), and
a timed-out query falls back to the user's raw words. Review 2026-09-27, #6.

The switch now lives on the model client itself, built once per name in Deps, so
every caller of the router model gets it, including ones nobody has written yet.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from chord.config import Settings
from chord.dependencies import Deps

OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def _deps(tmp_path, **overrides) -> Deps:
    fields = dict(
        data_dir=tmp_path,
        router_model="example-router", router_base_url="http://router.test/v1",
        persona_model="persona-model", persona_base_url="http://persona.test/v1",
        router_thinking_mode="qwen_chat_template",
    )
    settings = Settings(**{**fields, **overrides})
    return Deps(settings)


def _payload(client) -> dict:
    return client._get_request_payload([HumanMessage("hi")])


def test_the_router_model_client_has_thinking_off(tmp_path):
    deps = _deps(tmp_path)
    assert _payload(deps.model("example-router")).get("extra_body") == OFF


def test_the_persona_model_client_is_untouched(tmp_path):
    """Persona thinking is the caller's to ask for (reasoning_effort)."""
    deps = _deps(tmp_path)
    assert "extra_body" not in _payload(deps.model("persona-model"))


def test_a_router_model_that_is_also_the_persona_keeps_its_thinking(tmp_path):
    deps = _deps(tmp_path, persona_model="example-router", persona_base_url="http://router.test/v1")
    assert "extra_body" not in _payload(deps.model("example-router"))
