"""The router: one model call that decides what the last user turn needs.

It never speaks. If its output can't be parsed, or the call fails or runs
past its time limit, the turn falls back to chat and the failure is traced. A
router that can't be read, or reached, must not be able to start work, and
must not be able to hold up a plain chat turn either.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from .registry import PROMPTS_DIR, Capability, prompt_version

ROUTER_PROMPT = PROMPTS_DIR / "router.md"
HISTORY_TURNS = 8


@dataclass
class RouteDecision:
    route: str = "chat"
    intent: str = ""
    constraints: list[str] = field(default_factory=list)
    latitude: str = ""
    question: str | None = None
    parse_error: str | None = None


def _capability_lines(capabilities: dict[str, Capability]) -> str:
    return "\n".join(f"- {c.id}: {c.description}" for c in capabilities.values()) or "- (none)"


def _transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages[-HISTORY_TURNS:]:
        if m.get("role") in ("user", "assistant"):
            lines.append(f"{m['role']}: {m.get('text', '')}")
    return "\n".join(lines)


def parse(raw: str, known: set[str]) -> RouteDecision:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return RouteDecision(parse_error="no JSON object in router output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return RouteDecision(parse_error=f"invalid JSON: {exc}")
    if not isinstance(data, dict):
        return RouteDecision(parse_error="router output is not a JSON object")
    route = data.get("route")
    if not isinstance(route, str) or route not in {"chat", "clarify", *known}:
        return RouteDecision(parse_error=f"unknown route {route!r}")
    constraints = data.get("constraints") or []
    if not isinstance(constraints, list):
        constraints = [str(constraints)]
    return RouteDecision(
        route=route,
        intent=str(data.get("intent") or ""),
        constraints=[str(c) for c in constraints],
        latitude=str(data.get("latitude") or ""),
        question=str(data.get("question") or "") or None,
    )


async def route(
    messages: list[dict], capabilities: dict[str, Capability], llm: Runnable, timeout: float = 10.0,
    *, json_mode: bool = True,
) -> tuple[RouteDecision, dict]:
    """Returns the decision and the provenance fields to trace."""
    system = ROUTER_PROMPT.read_text().format(capabilities=_capability_lines(capabilities))
    provenance = {"router_prompt_version": prompt_version(ROUTER_PROMPT)}
    # JSON mode (#83): on the b2 input the router model stopped before the closing
    # brace 12 of 12 times at temperature 0, so the clarification was lost and
    # the turn fell back to chat. With response_format json_object it gave 12 of 12
    # valid objects, and routed 14 other asks identically to free text.
    # Test doubles without .bind are used as they are. The llm is typed
    # Runnable rather than BaseChatModel because graph.router_client binds
    # it (returning _ChatModelBinding, which is-a Runnable but not strictly
    # a BaseChatModel). What we need here is .ainvoke; Runnable has it.
    if json_mode and hasattr(llm, "bind"):
        responder = llm.bind(response_format={"type": "json_object"})
    else:
        responder = llm
    try:
        # One bounded wait around the whole call, retries included. On
        # 2026-09-11 the router backend went down and every turn hung until
        # LiteLLM gave up at 180 s. A bare exception also gave a bare 500.
        reply = await asyncio.wait_for(responder.ainvoke([SystemMessage(system), HumanMessage(_transcript(messages))]), timeout)
    except Exception as exc:  # cancellation (BaseException) still propagates
        kind = "timed out" if isinstance(exc, TimeoutError) else f"failed ({type(exc).__name__})"
        provenance["router_call_error"] = f"router call {kind}"[:300]
        return RouteDecision(parse_error=provenance["router_call_error"]), provenance
    raw = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
    decision = parse(raw, set(capabilities))
    if decision.parse_error:
        provenance["router_parse_error"] = decision.parse_error
        provenance["router_raw"] = raw[:2000]
    return decision, provenance


# The lanes a classifier backend may return. There is no clarify lane.
CLASSIFIER_ROUTES = frozenset({"chat", "image", "search", "audio", "video"})


async def classify(
    messages: list[dict], url: str, timeout: float, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[str | None, dict]:
    """The lane from the configured classifier backend, or None when it can't be had.

    It reads the same text the router model does. A classifier is trained on
    _transcript's exact format, so any change here is a change to its input.
    None never means chat. The caller routes that turn with the router model,
    and the reason is traced.
    """
    provenance: dict = {}
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            reply = await asyncio.wait_for(client.post(url, json={"text": _transcript(messages)}), timeout)
        reply.raise_for_status()
        data = reply.json()
    except Exception as exc:  # cancellation (BaseException) still propagates
        timed_out = isinstance(exc, (TimeoutError, httpx.TimeoutException))
        provenance["classifier_error"] = ("classifier call timed out" if timed_out
                                          else f"classifier call failed ({type(exc).__name__})")[:300]
        return None, provenance
    route = data.get("route") if isinstance(data, dict) else None
    # Type first: a list or dict route is unhashable, and the membership test would
    # raise out of here instead of falling back (Copilot on #327).
    if not isinstance(route, str) or route not in CLASSIFIER_ROUTES:
        provenance["classifier_error"] = f"unknown classifier route {route!r}"[:300]
        return None, provenance
    provenance["classifier_route"] = route
    probabilities = data.get("probabilities")
    if isinstance(probabilities, dict) and isinstance(probabilities.get(route), (int, float)):
        provenance["classifier_confidence"] = round(float(probabilities[route]), 4)
    if isinstance(data.get("engine_sha256"), str):
        provenance["classifier_engine_sha256"] = data["engine_sha256"][:64]
    return route, provenance
