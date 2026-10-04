"""Chat Completions validation, tool policy, and response orchestration."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator
from typing import NamedTuple, cast
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from ulid import ULID

from . import forced_call
from . import graph as graph_mod
from . import audio
from . import artifact_links, manifest, normalize, progress, stored_chat, voice_clip, web_search
from .api_errors import unreachable as _unreachable
from .api_errors import upstream_error_body as _upstream_error_body
from .api_errors import upstream_http_status as _upstream_http_status
from .artifacts import ArtifactStore
from .chat_contract import CHAT_SPEC_PARAMS
from .chat_wire import _conform_choices as _conform_choices
from .chat_wire import _drop_reasoning as _drop_reasoning
from .responses_store import ResponseStore
from .contract import Outcome
from .http_transport import ClosingStreamingResponse
from .trace import Trace
from .upstream import UpstreamError

REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
def error(status: int, message: str, code: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}},
        status_code=status,
    )


# Declared no-ops (manifest `declared_noop_params`) have no effect, but their
# VALUES are still validated against the pinned spec: accepting "forever" for
# prompt_cache_retention and forwarding it upstream was a 200 OpenAI would
# 400 (#125). test_every_declared_noop_has_a_validator keeps this complete.
NOOP_VALIDATORS = {
    "user": lambda v: isinstance(v, str),
    "safety_identifier": lambda v: v is None or (isinstance(v, str) and len(v) <= 64),
    "prompt_cache_key": lambda v: v is None or isinstance(v, str),
    "prompt_cache_options": lambda v: isinstance(v, dict)
        and (("ttl" not in v) or v["ttl"] == "30m")
        and (("mode" not in v) or v["mode"] in ("implicit", "explicit")),
    "prompt_cache_retention": lambda v: v is None or (isinstance(v, str) and v in ("in_memory", "24h")),
}


# A forced tool call (S03, red team 2026-09-15). Routing it to chat was not
# enough: live, the model answered "Hello!" with prose under tool_choice
# "required", even called directly (vLLM 0.28, qwen3_xml parser). Whatever the
# backend does, prose alone never reaches the client as a satisfied force.
FORCED_CALL_CODE = "upstream_tool_call_contract_violation"
FORCED_CALL_MESSAGE = "The model did not return the tool call this request requires. Retry the request."


def _named(entry) -> str | None:
    """The function name in a {"type": "function", "function": {"name": ...}}."""
    if isinstance(entry, dict) and entry.get("type") == "function" and isinstance(entry.get("function"), dict):
        name = entry["function"].get("name")
        return name if isinstance(name, str) else None
    return None


def _declared_schema_error(body: dict) -> JSONResponse | None:
    """Every declared tool's `parameters` must be valid JSON Schema, checked HERE.

    the requirement, 2026-09-19: run it at request time and return 400 on a malformed
    schema, so a bad client schema surfaces as a bad client schema instead of a confusing
    failure on the repair path -- where it would arrive as "we could not evaluate" and
    could only be answered by refusing a bind on evidence nobody obtained.

    Compiled once per request here; the repair path compiles again on the rare occasion
    it runs, which is cheap and keeps this function free of hidden state."""
    for name, parameters in normalize.declared_functions(body).items():
        if not parameters:
            continue                     # a tool declaring nothing accepts nothing; legal
        try:
            normalize.check_declared_schema(parameters)
        except normalize.SchemaUnusable as exc:
            return error(400, f"the schema declared for {name!r} is not valid JSON Schema: {exc}",
                         "invalid_value", "tools")
    return None


def _tools_error(body: dict) -> JSONResponse | None:
    """The pinned shapes of tools and legacy functions, checked whole before
    anything reads inside them (#174: a string `function` crashed the
    answer check after the model had run)."""
    tools = body.get("tools")
    if tools is not None:
        bad = error(400, "tools must be an array of {type: function, function: {name}} or {type: custom, custom: {name}}",
                    "invalid_value", "tools")
        if not isinstance(tools, list):
            return bad
        for t in tools:
            if not isinstance(t, dict):
                return bad
            kind = t.get("type")
            if kind == "function":
                if _named(t) is None:
                    return bad
            elif kind == "custom":
                if not (isinstance(t.get("custom"), dict) and isinstance(t["custom"].get("name"), str)):
                    return bad
            else:
                return bad
    functions = body.get("functions")
    if functions is not None:
        if not (isinstance(functions, list) and all(isinstance(f, dict) and isinstance(f.get("name"), str) for f in functions)):
            return error(400, "functions must be an array of {name}", "invalid_value", "functions")
    return None


def _tool_choice_error(body: dict) -> JSONResponse | None:
    """The pinned shapes of tool_choice and function_call, checked whole before
    anything reads inside them (#174: malformed objects crashed). Run
    after _tools_error, so declared names can be read safely."""
    if (bad := _tools_error(body)) is not None:
        return bad
    # Shapes are sound, so the schemas inside them can be compiled (the requirement).
    if (bad := _declared_schema_error(body)) is not None:
        return bad
    declared = _declared_functions(body)

    def undeclared(name: str) -> JSONResponse:
        return error(400, f"{name!r} is not a declared function", "invalid_value", "tool_choice")

    choice = body.get("tool_choice")
    if choice is not None:
        bad = error(400, "tool_choice must be none, auto, required, a named function, or allowed_tools",
                    "invalid_value", "tool_choice")
        if isinstance(choice, str):
            if choice not in ("none", "auto", "required"):
                return bad
            if choice == "required" and not declared:
                return error(400, "tool_choice required needs at least one declared function", "invalid_value", "tool_choice")
        elif not isinstance(choice, dict):
            return bad
        elif choice.get("type") == "custom":
            # A real schema form we can't yet check the answer to: refused,
            # never silently weakened to "unforced".
            return error(400, "forcing a custom tool isn't supported yet", "unsupported_parameter", "tool_choice")
        elif choice.get("type") == "function":
            name = _named(choice)
            if name is None:
                return bad
            if name not in declared:
                return undeclared(name)
        elif choice.get("type") == "allowed_tools":
            allowed = choice.get("allowed_tools")
            if not isinstance(allowed, dict) or allowed.get("mode") not in ("auto", "required"):
                return bad
            tools = allowed.get("tools")
            if not isinstance(tools, list) or not tools or not all(isinstance(t, dict) for t in tools):
                return bad
            for t in tools:
                if t.get("type") == "custom":
                    # Refused in both modes: the answer check can't yet
                    # recognise a custom call, so an allowed one would 502
                    # (ae36d9e).
                    return error(400, "custom tools in allowed_tools aren't supported yet", "unsupported_parameter", "tool_choice")
                name = _named(t)
                if name is None:
                    return bad
                if name not in declared:
                    return undeclared(name)
        else:
            return bad
    call = body.get("function_call")
    if call is not None:
        if isinstance(call, str):
            if call not in ("none", "auto"):
                return error(400, "function_call must be none, auto, or a named function", "invalid_value", "function_call")
        elif not (isinstance(call, dict) and isinstance(call.get("name"), str)):
            return error(400, "function_call must be none, auto, or a named function", "invalid_value", "function_call")
        elif call["name"] not in declared:
            return error(400, f"{call['name']!r} is not a declared function", "invalid_value", "function_call")
    return None


class CallConstraint(NamedTuple):
    """What the client's tool_choice/function_call demands of the answer."""
    param: str                    # "tool_choice" or "function_call"
    required: bool                # a call must come back
    allowed: frozenset | None     # the names a call may use; None: any declared


def _forced_call(body: dict) -> CallConstraint | None:
    """The constraint the answer is checked against, or None. Only call on a
    body _tool_choice_error has passed."""
    choice = body.get("tool_choice")
    if choice == "required":
        return CallConstraint("tool_choice", True, None)
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            return CallConstraint("tool_choice", True, frozenset({_named(choice)}))
        if choice.get("type") == "allowed_tools":
            # A hard subset in both modes; "required" also demands a call
            # (pinned schema; the addendum on #174).
            tools = choice["allowed_tools"]["tools"]
            return CallConstraint("tool_choice", choice["allowed_tools"]["mode"] == "required",
                                  frozenset(n for n in map(_named, tools) if n is not None))
    call = body.get("function_call")
    if isinstance(call, dict):
        return CallConstraint("function_call", True, frozenset({call["name"]}))
    return None


def _declared_functions(body: dict) -> set[str]:
    """Declared function names; reads nothing it hasn't type-checked."""
    tools, functions = body.get("tools"), body.get("functions")
    names = {_named(t) for t in tools} if isinstance(tools, list) else set()
    if isinstance(functions, list):
        names |= {f.get("name") for f in functions if isinstance(f, dict)}
    return {n for n in names if isinstance(n, str)}


def _forced_call_violation(forced: CallConstraint, calls: list[tuple], declared: set[str]) -> str | None:
    """Why the returned calls don't satisfy the constraint, or None. `calls` is
    (name, arguments) per assembled call; content beside a valid call is fine."""
    allowed = forced.allowed
    if not calls:
        return "no tool call" if forced.required else None
    if MALFORMED in calls or any(not isinstance(n, str) or not isinstance(a, str) for n, a in calls):
        return "a malformed call"
    for called, arguments in calls:
        if called not in declared:
            return f"call to an undeclared function {called!r}"
        if allowed is not None and called not in allowed:
            return f"call to {called!r}, not the forced {sorted(allowed)}"
        try:
            json.loads(arguments if isinstance(arguments, str) else "")
        except ValueError:
            return f"call to {called!r} with arguments that are not JSON"
    return None


# The answer is untrusted upstream output: every shape is checked before it is
# read, and anything malformed becomes MALFORMED, which fails the constraint.
# Walking it must never raise (64380fc: a string `function` crashed).
MALFORMED = (None, None)


def _call_parts(fn) -> tuple:
    if not isinstance(fn, dict):
        return MALFORMED
    name, arguments = fn.get("name"), fn.get("arguments")
    return (name, arguments) if isinstance(name, str) and isinstance(arguments, str) else MALFORMED


def _message_calls(message) -> list[tuple]:
    if not isinstance(message, dict):
        return [MALFORMED]
    tool_calls, legacy = message.get("tool_calls"), message.get("function_call")
    calls = []
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            return [MALFORMED]
        calls += [_call_parts(c.get("function")) if isinstance(c, dict) else MALFORMED for c in tool_calls]
    if legacy is not None:
        calls.append(_call_parts(legacy))
    return calls


def _stream_calls(events: list) -> list[tuple]:
    """The calls a buffered stream assembled, by tool_call index, plus a legacy
    function_call assembled from its fragments. Any malformed piece adds
    MALFORMED rather than raising."""
    parts: dict[int, dict] = {}
    legacy: dict | None = None
    malformed = False

    def add(slot: dict, fn) -> bool:
        if not isinstance(fn, dict):
            return False
        for key in ("name", "arguments"):
            piece = fn.get(key)
            if piece is None:
                continue
            if not isinstance(piece, str):
                return False
            slot[key] += piece
        return True

    for mode, data in events:
        if mode != "custom" or not isinstance(data, dict) or "chunk" not in data:
            continue
        chunk = data["chunk"]
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if choices is None:
            continue
        if not isinstance(choices, list):
            malformed = True
            continue
        for choice in choices:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if delta is None:
                continue
            if not isinstance(delta, dict):
                malformed = True
                continue
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    malformed = True
                    tool_calls = []
                for call in tool_calls:
                    index = call.get("index", 0) if isinstance(call, dict) else None
                    if not isinstance(index, int) or isinstance(index, bool):
                        malformed = True
                        continue
                    slot = parts.setdefault(index, {"name": "", "arguments": ""})
                    if "function" in call and not add(slot, call["function"]):
                        malformed = True
            fc = delta.get("function_call")
            if fc is not None:
                legacy = legacy or {"name": "", "arguments": ""}
                if not add(legacy, fc):
                    malformed = True
    calls = [(p["name"], p["arguments"]) for _, p in sorted(parts.items())]
    if legacy is not None:
        calls.append((legacy["name"], legacy["arguments"]))
    return calls + ([MALFORMED] if malformed else [])


def _stream_calls_shaped(events: list) -> list[tuple]:
    """`_stream_calls` with each call's OWN shape kept: (shape, name, arguments).

    The guard only needs names, so the assembler throws the shape away. Telemetry needs
    it: a response carrying both `tool_calls` and a legacy `function_call` is `mixed`,
    and that is a real observable the schema exists to record. Stamping one shape across
    every call -- or worse, inferring it from what the REQUEST declared -- makes `mixed`
    unrecoverable at exactly the boundary it was designed for (2026-09-19).

    Derived from what came back. The request has no vote."""
    legacy_seen = False
    for mode, data in events:
        if mode != "custom" or not isinstance(data, dict) or "chunk" not in data:
            continue
        chunk = data["chunk"]
        for choice in (chunk.get("choices") if isinstance(chunk, dict) else None) or []:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if not isinstance(delta, dict):
                continue
            if delta.get("function_call"):
                legacy_seen = True
    calls = _stream_calls(events)
    # `_stream_calls` orders modern slots first, then the single legacy call,
    # then any MALFORMED sentinel appended for a malformed fragment. Count the
    # malformed positions so they are not stamped modern and the legacy call
    # is not stamped modern when both legacy and malformed co-occur
    # (review 2026-09-23). The shape is stamped by POSITION in `_stream_calls`'s
    # ordering, which is the only ordering the assembler keeps.
    n_malformed = sum(1 for _, args in calls if args is None)
    n_legacy = 1 if legacy_seen else 0
    n_modern = max(len(calls) - n_legacy - n_malformed, 0)
    shapes = [normalize.MODERN] * n_modern + [normalize.LEGACY] * n_legacy
    return [(shapes[i] if i < len(shapes) else normalize.MODERN, name, args)
            for i, (name, args) in enumerate(calls)]


def _message_calls_shaped(message) -> list[tuple]:
    """`_message_calls` with each call's own shape. Same reason as above."""
    if not isinstance(message, dict):
        return [(normalize.MODERN, None, None)]
    out = []
    for call in message.get("tool_calls") or []:
        name, args = _call_parts(call.get("function")) if isinstance(call, dict) else MALFORMED
        out.append((normalize.MODERN, name, args))
    if message.get("function_call") is not None:
        name, args = _call_parts(message["function_call"])
        out.append((normalize.LEGACY, name, args))
    return out


def _forced_call_error(forced, reason: str, trace, deps, headers) -> JSONResponse:
    trace.set(forced_call_violation=reason, forced_call_param=forced.param, forced_call_required=forced.required,
              forced_call_allowed=sorted(forced.allowed) if forced.allowed is not None else None)
    deps.traces.write(trace)
    return JSONResponse({"error": {"message": FORCED_CALL_MESSAGE, "type": "server_error",
                                   "param": forced.param, "code": FORCED_CALL_CODE}},
                        status_code=502, headers=headers)


# An UNDECLARED call (#289, red team 2026-09-19). The forced path already binds names
# (forced_call.py) but only fires when a call is REQUIRED; on the auto path nothing
# compared `function.name` to the tools the request declared, and a name the client
# never supplied reached it verbatim. Measured twice in 222 firings of one body, with
# two DIFFERENT invented names -- so membership in the request's own declared set is the
# only sound check: an alias map catches `weather` and waves `weather_getCurrent_2604`
# straight through.
#
# Same category as FORCED_CALL_CODE -- the backend claiming a call it was not entitled
# to make -- pointed the other way, so it reuses the code. It does NOT reuse the forced
# message or param, because nothing here was forced.
UNDECLARED_CALL_MESSAGE = ("The model returned a tool call naming a function this request did not "
                           "declare. Retry the request.")


def _undeclared_names(calls: list[tuple], declared: set[str]) -> list[str]:
    """Names in `calls` the request never declared, plus MALFORMED, in order, deduped.

    Ignoring MALFORMED because another path refused it was unsafe. A regression
    test disproved that premise on the non-stream auto path: a call whose
    `function` was the string "not-an-object" came back 200, while the same planted
    shape refused on the stream path. Two paths disagreeing in opposite directions is
    the tell that the property was being asserted per-path instead of about the
    response. A call with no readable name cannot be a member of any set, so it
    refuses here rather than relying on a guard that turns out not to run."""
    seen, out = set(), []
    for name, _arguments in calls:
        label = name if isinstance(name, str) else "<malformed>"
        if (label == "<malformed>" or label not in declared) and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def _record_normalization(trace, body: dict, shaped: list[tuple], *, transport: str,
                          sse_data_emitted: bool, content_delta_emitted: bool,
                          disposition: str) -> None:
    """Write the canonical row. The ONE place that assembles the observation.

    `shaped` is (shape, name, arguments) per ASSEMBLED call -- never per delta, and each
    carrying its OWN shape so a mixed response stays representable. An earlier version
    recorded `len(suppressed)`, which counted name fragments: two rows of telemetry for
    one returned call, and a count cannot substitute for an observation."""
    observation = normalize.observe(shaped, normalize.declared_functions(body))
    trace.set(**normalize.canonical_row(body, observation, transport=transport,
                                        sse_data_emitted=sse_data_emitted,
                                        content_delta_emitted=content_delta_emitted,
                                        disposition=disposition))


def _rename_call(message: dict, name: str, arguments: str) -> dict:
    """The message with its single call rebuilt under `name`. Never mutates the original:
    the emitted call stays intact as the trace's evidence of what the backend actually
    sent, which is the thing a repair must not overwrite."""
    fn = {"name": name, "arguments": arguments}
    if isinstance(message.get("function_call"), dict):
        return {**message, "function_call": fn}
    calls = message.get("tool_calls")
    assert isinstance(calls, list) and calls
    call = calls[0]
    return {**message, "tool_calls": [{**call, "function": fn}]}


def _bind_to_sole_tool(calls: list[tuple], body: dict, trace):
    """The repaired call, or None. The ONLY caller of the bind, and its licence gate.

    `calls` is (name, arguments) per assembled call, arguments AS EMITTED. The decision
    is taken on those raw arguments and nothing else: `_conform()` enforces the caller's
    schema by DROPPING keys it forbids, so reading conformed arguments would let the
    repair manufacture the evidence that licensed it, and every shape assertion would
    pass identically when the model meant another function entirely (the boundary).

    Conformance runs AFTER, on the bound tool's own schema, and its dropped keys are
    recorded on the turn as accounting -- never as grounds."""
    declared = normalize.declared_functions(body)
    observation = normalize.observe(
        [(normalize.MODERN, name, arguments) for name, arguments in calls], declared)
    try:
        bound = normalize.sole_tool_bind(observation, declared)
    except normalize.SchemaUnusable as exc:
        # The door refuses these before the model runs. This is the path that
        # still has a trace: a schema that only fails while weighing arguments
        # must not escape as an untraced 500.
        trace.set(schema_unusable=str(exc)[:300])
        return None
    if bound is None:
        return None
    raw = calls[0][1]
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:                       # unreachable: the bind needed valid JSON
        return None
    dropped: list = []
    args = forced_call._conform(args, declared[bound], dropped)
    trace.set(normalization_disposition="sole_tool_bind", bound_tool=bound,
              bind_dropped_keys=dropped or None)
    return bound, json.dumps(args)


# Any of these on a request means the caller spoke about tools, so a returned call is
# judged against what they declared. None of them at all is the no-tools case, the only
# one that takes the bounded text-only retry (both transports, review 2026-09-24 B14).
TOOL_PARAMS = ("tools", "functions", "tool_choice", "function_call", "parallel_tool_calls")


def _no_tool_parameter(body: dict) -> bool:
    return (normalize.tool_parameter_state(body) == normalize.ABSENT
            and not any(k in body for k in TOOL_PARAMS))


def _undeclared_call_error(names: list[str], trace, deps, headers) -> JSONResponse:
    trace.set(undeclared_tool_call_names=names)
    deps.traces.write(trace)
    return JSONResponse({"error": {"message": UNDECLARED_CALL_MESSAGE, "type": "server_error",
                                   "param": "tools", "code": FORCED_CALL_CODE}},
                        status_code=502, headers=headers)


def _without_calls(chunk: dict) -> dict:
    """The chunk with every tool-call field removed, prose and structure intact.

    Builds a new object rather than mutating: the original is still the evidence, and a
    suppressed call that has been edited in place cannot be recorded honestly."""
    choices = []
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            choices.append(choice)
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            delta = {k: v for k, v in delta.items() if k not in ("tool_calls", "function_call")}
            choice = {**choice, "delta": delta}
        choices.append(choice)
    return {**chunk, "choices": choices}


def _decisive(event) -> set:
    """The delta kinds a primed event carries, for the no-tools decision."""
    mode, data = event
    if mode != "custom" or not isinstance(data, dict) or "chunk" not in data:
        return set()
    return normalize.delta_kinds(data["chunk"])


async def _prime_stream(events, request: Request, stop: asyncio.Event, whole: bool = False,
                        until_decisive: bool = False):
    """Keep pre-header errors as HTTP errors while observing a vanished client.

    The request body is already consumed. Only this watcher reads ASGI receive
    until priming completes; StreamingResponse takes over afterward.
    None means the client disconnected, rather than an empty completed stream.
    `whole` reads the entire stream before any header: a forced tool call is
    only known valid once it is complete (S03; the design).

    `until_decisive` reads until the first chunk that carries CONTENT or a CALL,
    which is what the no-tools case needs and all it needs. When the request
    declared no tool parameter every returned call is invalid whatever its
    spelling, so the decision wants no name and no arguments -- only which of the
    two arrived first. Prose first means the stream is committed and a later call
    is suppressed with an explicit terminal error; a call first means nothing has
    reached the client yet and the refusal can still be an ordinary HTTP error.
    This is why an ordinary chat stream pays no buffering: only a request that
    declares tools quarantines whole calls (the amendment, dfa85af).
    """
    async def collect():
        pending = []
        async for event in events:
            pending.append(event)
            if whole:
                continue
            if until_decisive:
                if _decisive(event):
                    break
            elif event[0] == "custom":
                break
        return pending

    async def watch_disconnect():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    startup = asyncio.create_task(collect())
    watcher = asyncio.create_task(watch_disconnect())
    handed_off = False
    try:
        done, _ = await asyncio.wait({startup, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            watcher.result()  # unexpected receive errors are not disconnects
            return None
        pending = await startup  # preserve an upstream error before headers
        handed_off = True
        return pending
    finally:
        watcher.cancel()
        if not handed_off:
            stop.set()
            startup.cancel()
        await asyncio.gather(startup, watcher, return_exceptions=True)
        if not handed_off:
            await events.aclose()


async def _cancelled_by_disconnect(work: asyncio.Task, request: Request) -> bool:
    """Await `work`; True when the client left first and the work was cancelled.

    The stream path watches for a vanished client in _prime_stream; the non-stream
    path awaited the graph directly, so a turn whose client had gone kept running --
    an abandoned image turn held the GPU and the render CLI for up to
    IMAGE_DEADLINE_S (review 2026-09-27, #8). The body is already consumed, so the
    next ASGI message is the disconnect. Whatever happens to this coroutine, the
    work does not outlive it."""
    async def watch_disconnect():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    watcher = asyncio.create_task(watch_disconnect())
    try:
        done, _ = await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return False
        watcher.result()  # unexpected receive errors are not disconnects
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        return True
    finally:
        watcher.cancel()
        if not work.done():
            work.cancel()
        # Await BOTH. Awaiting only the watcher let a cancelled handler return while
        # the work was still in its own async cleanup (#354): the guarantee
        # above holds only once the work has actually finished.
        await asyncio.gather(watcher, work, return_exceptions=True)


def _bad_image_fields(m: dict) -> str | None:
    """Shape-check the image fields a client may replay on an assistant message
    (`image`, `images`) before anything normalises them."""
    if "image" in m and m["image"] is not None:
        img = m["image"]
        if not isinstance(img, dict) or not isinstance(img.get("url"), str):
            return "message.image must be an object with a string url"
    if "images" in m and m["images"] is not None:
        imgs = m["images"]
        if not isinstance(imgs, list) or not all(
            isinstance(x, dict) and isinstance(x.get("image_url"), dict) and isinstance(x["image_url"].get("url"), str)
            for x in imgs
        ):
            return "message.images must be a list of {image_url: {url: string}}"
    return None


# The image_url policy (review 2026-09-22, decision #7): the persona BACKEND
# fetches image_url parts itself, from its own position on the VLAN, so an
# http URL a caller names is a fetch primitive aimed at the GPU host's network
# -- SSRF by proxy, with the answer observable through the model's reply.
# Chord is the only place a policy can live for its callers, and the certified
# vision evidence is a data: URI (evidence/vision-input/vision_check.py), so
# data:image is what is allowed by default and http(s) only by exact-host
# opt-in (Settings.image_url_allowed_hosts). Every other scheme (file:, ftp:,
# ...) is refused outright: what those mean is the backend fetcher's decision,
# not the caller's. The refusal message never enumerates the allowlist.
IMAGE_URL_REFUSAL = ("image_url must be a data:image/... URI, or an http(s) URL whose exact host "
                     "this deployment allows")


def _image_url_policy_error(part: dict, allowed_hosts: frozenset) -> str | None:
    """Why this image_url part violates the policy, or None. Only called on a
    part whose type is image_url; a non-string url is left to the backend, as
    before -- this guard is about WHERE a string url points, not its shape."""
    img = part.get("image_url")
    url = img.get("url") if isinstance(img, dict) else None
    if not isinstance(url, str) or not url:
        return None
    if url.lower().startswith("data:"):
        return None if url.lower().startswith("data:image/") else IMAGE_URL_REFUSAL
    try:
        # urlparse raises on a malformed bracketed authority ("http://[::1"),
        # and so does .hostname on a bracketed non-address: both were a 500
        # here, where the policy promises a named 400 (Copilot on #339).
        parsed = urlparse(url)
        parsed_host = parsed.hostname
    except ValueError:
        return IMAGE_URL_REFUSAL
    if parsed.scheme.lower() in ("http", "https"):
        # Kill the parser-divergence class outright: chord's urlparse and the
        # BACKEND's fetcher are two implementations, and an authority carrying
        # userinfo or a backslash is exactly where they disagree about which
        # host a URL names -- a probe showed evil\@allowed.example reading
        # as evil.example to urllib3.util.parse_url while urlparse says
        # allowed.example. Neither form has a legitimate use in an image URL,
        # so both refuse regardless of allowlist tier (review follow-up).
        lowered = parsed.netloc.lower()
        if "@" in parsed.netloc or "\\" in parsed.netloc or "%40" in lowered or "%5c" in lowered:
            return IMAGE_URL_REFUSAL
        try:
            port = parsed.port          # raises on a malformed port: refuse, never forward
        except ValueError:
            return IMAGE_URL_REFUSAL
        host = (parsed_host or "").lower()
        netloc = parsed.netloc.lower()  # keeps any userinfo, so "x@host" never matches "host"
        for entry in allowed_hosts:
            if ":" in entry:
                if entry == netloc:
                    return None
            elif entry == host and port is None:
                return None
        return IMAGE_URL_REFUSAL
    return IMAGE_URL_REFUSAL


def _image_url_forwarded_hosts(messages: list) -> list[str]:
    """The http(s) hosts this turn's allowed image_url parts name, for the
    trace: the alpha visibility that tunes the allowlist from evidence instead
    of guesses. Hosts only -- a URL can carry a path and query that are
    content, and content does not belong in traces."""
    hosts = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            img = part.get("image_url")
            url = img.get("url") if isinstance(img, dict) else None
            if isinstance(url, str):
                try:
                    parsed = urlparse(url)
                except ValueError:
                    continue
                if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
                    hosts.append(parsed.netloc.lower())
    return sorted(set(hosts))


N_MAX = 8   # the spec allows 128; each choice here is a whole turn, so the ceiling is ours and named


def _single_choice(n) -> bool:
    """`n` that asks for what we return anyway: one choice. A bool is not a count."""
    return n is None or (type(n) is int and n == 1)


def _last_user_text(messages: list[dict]) -> str:
    """The trailing user message as a single string, for the shadow scorer.

    Walks messages from the end and returns the first one with role=user as
    plain text. String content is taken verbatim; array content is
    concatenated across text parts; everything else (image_url, audio,
    refusal, etc.) is skipped, which means an image-only turn produces an
    empty string and the shadow scorer sees nothing to flag -- the image
    gate is a Phase-3 surface, not Phase 1."""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and isinstance(p.get("text"), str)]
            return "\n".join(parts)
        return ""
    return ""


def _validate(body, settings=None) -> tuple[str | None, JSONResponse | None]:
    """Every shape check happens here, before anything calls .get or hashes a
    value, so malformed JSON is a structured 400, never a 500.

    `settings` carries the deployment's image_url allowlist; None (legacy and
    direct unit callers) means the strictest policy: data:image only."""
    if not isinstance(body, dict):
        return None, error(400, "body must be a JSON object", "invalid_body")
    model = body.get("model")
    persona_id = graph_mod.persona_for(model)
    if persona_id is None:
        return None, error(404, f"model {model!r} not served; use one of /v1/models", "model_not_found", "model")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, error(400, "messages must be a non-empty array", "invalid_messages", "messages")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            return None, error(400, "each message must be an object with a string role", "invalid_messages", f"messages[{i}]")
        # The pinned schema types `name` as a string. The instruction fold turns
        # it into a visible sender label, so a malformed one must be refused
        # here, never coerced into a valid-looking identity (#150).
        if "name" in m and not isinstance(m["name"], str):
            return None, error(400, "message name must be a string", "invalid_messages", f"messages[{i}].name")
        bad = _bad_image_fields(m)
        if not bad and m.get("audio") is not None and not (
            isinstance(m["audio"], dict) and isinstance(m["audio"].get("id"), str)
            and isinstance(m["audio"].get("transcript", ""), str)
        ):
            bad = "message.audio must be an object with a string id (and a string transcript, if any)"
        if bad:
            return None, error(400, bad, "invalid_messages", f"messages[{i}]")
        content = m.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            return None, error(400, "content must be a string, an array of parts, or null", "invalid_messages", f"messages[{i}].content")
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                return None, error(400, "each content part must be an object with a string type", "invalid_messages", f"messages[{i}].content")
            if part["type"] not in manifest.allowed_parts():
                return None, error(400, f"content part {part['type']!r} is not supported yet", "unsupported_content_part", f"messages[{i}].content")
            if part["type"] == "image_url":
                # WHERE the url points is a deployment policy, checked for every
                # role: an assistant-message part is forwarded and fetched just
                # the same (see IMAGE_URL_REFUSAL).
                img = part.get("image_url")
                if not isinstance(img, dict) or not isinstance(img.get("url"), str) or not img["url"]:
                    # The policy below judges only this shape; every other shape
                    # used to pass it unjudged, so the bare-string form of a
                    # metadata URL reached the backend (review 2026-09-24 A7).
                    return None, error(400, "image_url must be an object with a string url", "invalid_messages",
                                       f"messages[{i}].content")
                allowed = settings.image_url_allowed_hosts if settings is not None else frozenset()
                if (bad := _image_url_policy_error(part, allowed)) is not None:
                    return None, error(400, bad, "unsupported_value", f"messages[{i}].content")
            if part["type"] == "input_audio":
                # OpenAI accepts input_audio from the user only; anywhere else
                # it would reach the text model untranscribed.
                bad = audio.check_part(part) if m["role"] == "user" else "input_audio is only accepted in user messages"
                if bad:
                    return None, error(400, bad, "invalid_input_audio", f"messages[{i}].content")
    if body.get("stream") is not None and not isinstance(body.get("stream"), bool):
        # "false" is truthy: bool() of it STREAMED a request that asked not to
        # be streamed. Completions has refused this by name since before the
        # review; chat now answers identically (review 2026-09-22).
        return None, error(400, "stream must be a boolean", "invalid_value", "stream")
    options = body.get("stream_options")
    if options is not None and (not isinstance(options, dict) or any(
            v is not None and not isinstance(v, bool) for v in options.values())):
        # The n>1 door read `.get` on it after the turns had run and been stored,
        # so a string here was a 500 with a completion left behind (review
        # 2026-09-24 A5). Shape only: which keys and whether stream must be true
        # stay as they were for chat.
        return None, error(400, "stream_options must be an object of booleans", "invalid_value", "stream_options")
    modalities = body.get("modalities")
    if modalities is not None and (not isinstance(modalities, list) or not all(isinstance(x, str) for x in modalities)):
        return None, error(400, "modalities must be an array of strings", "invalid_modalities", "modalities")
    effort = body.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or effort not in REASONING_EFFORTS):
        # The gateway accepts any value silently; the model behind it rejects it.
        # We say so up front instead of inheriting the silence.
        return None, error(400, f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}", "invalid_reasoning_effort", "reasoning_effort")
    for param in ("max_tokens", "max_completion_tokens"):
        # Checked here, under the name the caller sent: LiteLLM renames
        # max_completion_tokens to max_tokens, so the model's refusal of 0
        # named a field the caller never sent (S-cf-072 live, 2026-09-15).
        value = body.get(param)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            return None, error(400, f"{param} must be an integer of at least 1", "invalid_value", param)
    for param in manifest.load().get("declared_noop_params", []):
        if param in body and not NOOP_VALIDATORS[param](body[param]):
            return None, error(400, f"invalid value for {param!r}", "invalid_value", param)
    if (bad := stored_chat.metadata_error(body.get("metadata"))) is not None:
        return None, error(400, bad, "invalid_value", "metadata")
    tier = body.get("service_tier")
    if tier is not None and (not isinstance(tier, str) or tier not in manifest.load()["service_tier_routes"]):
        return None, error(400, f"service_tier must be one of {sorted(manifest.load()['service_tier_routes'])}", "invalid_service_tier", "service_tier")
    wanted_outputs = set(modalities or [])
    if "audio" in wanted_outputs and not manifest.load()["output"]["audio"]:
        return None, error(400, "audio output is not supported yet", "unsupported_modality", "modalities")
    if (bad := _tool_choice_error(body)) is not None:
        return None, bad
    if body.get("verbosity") not in (None, "low", "medium", "high"):
        return None, error(400, "verbosity must be low, medium or high", "invalid_value", "verbosity")
    if "web_search_options" in body:
        try:
            web_search.parse_options(body["web_search_options"])
        except web_search.WebSearchOptionsError as exc:
            return None, error(400, str(exc), "invalid_web_search_options", exc.param)
        # A search answer is cited prose from our specialist: it can be neither
        # the client's forced call nor its JSON. Refused, never substituted
        # (S03/S05, red team 2026-09-15; #156).
        constraint = graph_mod.client_constraint(body)
        if constraint == "response_format":
            return None, error(400, "web_search_options can't be combined with a JSON response_format yet",
                               "unsupported_parameter", "response_format")
        if constraint == "forced_tool":
            param = "function_call" if isinstance(body.get("function_call"), dict) else "tool_choice"
            return None, error(400, f"web_search_options can't be combined with a forced {param} yet",
                               "unsupported_parameter", param)
    if "store" in body and body["store"] is not None and not isinstance(body["store"], bool):
        return None, error(400, "invalid value for 'store'", "invalid_value", "store")
    n = body.get("n")
    if not _single_choice(n):
        # Several choices (S13g, S-cf-038). Each is one full turn of its own, so every guard a
        # single reply gets still applies to each. Bounded: n turns cost n prefills here.
        if type(n) is not int or not 1 <= n <= N_MAX:
            return None, error(400, f"n must be an integer between 1 and {N_MAX}", "invalid_value", "n")
        temperature = body.get("temperature")
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and temperature == 0:
            return None, error(400, "n above 1 needs sampling: temperature must be above 0", "invalid_value", "n")
        if body.get("web_search_options") is not None:
            return None, error(400, "n above 1 is not supported with web_search_options yet", "unsupported_parameter", "n")
        if "audio" in (body.get("modalities") or []):
            return None, error(400, "n above 1 is not supported with audio output yet", "unsupported_parameter", "n")
    for param in manifest.load()["refused_params"]:
        # An explicit null is not a request for the capability, it is the absence of one,
        # and the spec declares `moderation` as anyOf[ModerationParam, null]. SDKs
        # serialise unset optionals as null routinely, so refusing it 400s a client for
        # asking us to do nothing. Found by firing the variants rather than by reading:
        # the frozen S-cf-033 bar only covers the object form, so nothing here would
        # have caught it (2026-09-18).
        if param == "moderation" and body.get("moderation") is None:
            continue
        # Same shape as moderation: an explicit null is the absence of Predicted
        # Outputs, which SDKs send for an unset optional. A real value asks for
        # the speedup and the prediction token counts, which we do not produce.
        if param == "prediction" and body.get("prediction") is None:
            continue
        if param in body:
            return None, error(400, f"{param!r} asks for a capability this model does not have", "unsupported_parameter", param)
    # Unknown fields are refused, as OpenAI refuses them, never forwarded unread.
    # Safe on the evidence: 273 chat requests 09-11..09-14 carried only spec
    # params and chat_template_kwargs (trace `params`, read 2026-09-14).
    extensions = set(manifest.load().get("extension_params", []))
    unknown = sorted(k for k in body if k not in CHAT_SPEC_PARAMS and k not in extensions)
    if unknown:
        return None, error(400, f"Unrecognized request argument supplied: {unknown[0]}", "unsupported_parameter", unknown[0])
    return persona_id, None


def _split(body: dict) -> tuple[dict, list[dict], int]:
    """Client params (forwarded untouched), the client's messages exactly as
    sent (system/developer included, in order), and how many of them carry
    instructions. The caller owns the assistant's identity; our base layer
    only goes in front of it."""
    # `n` only gets this far as 1 or null (validated): a no-op, not forwarded.
    # `service_tier` selects our persona model (service_tier_routes); it is ours
    # to honour, so it is not forwarded to the backend.
    # `verbosity` is ours too (S14): the backend ignores it, so the graph honours it.
    # `store` and its `metadata` are ours too (stored_chat.py): never forwarded.
    local_params = {"model", "messages", "stream", "n", "audio", "service_tier", "verbosity", "store", "metadata", "prediction", "moderation"}
    local_params.update(manifest.load()["declared_noop_params"])
    params = {k: v for k, v in body.items() if k not in local_params}
    if isinstance(params.get("modalities"), list) and "audio" in params["modalities"]:
        # We speak the model's finished words (audio.Speaker); the model only writes them.
        params["modalities"] = [x for x in params["modalities"] if x != "audio"] or ["text"]
    messages = _strip_inline_images(body["messages"])
    instructions = sum(1 for m in messages if m.get("role") in ("system", "developer"))
    return params, messages, instructions


def _turn_artifacts(state: dict) -> list[dict]:
    """What the turn made, for our trace and for delivery. Never on the wire as
    a field: the public body is the pinned spec, nothing added (2026-09-16)."""
    result = state.get("result")
    return [a.model_dump() for a in (result.artifacts if result else [])]


INLINE_IMAGE = re.compile(r"!\[([^\]]*)\]\(data:image/[^)]+\)")


def _markdown(urls: list[str]) -> str:
    """A generated image, delivered inside the spec: a markdown image in the
    reply text. Chat Completions has no image output field; `content` is text,
    and a markdown image renders wherever markdown does and is a plain link
    everywhere else."""
    return "".join(f"\n\n![image]({url})" for url in urls)


def _marker_for(url: str) -> str:
    """A compact reference to the artifact: the sha256 of its bytes, which is
    how the artifact store identifies it. Visual recall can find it again."""
    try:
        digest = hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()[:16]
    except (ValueError, IndexError):
        digest = "unknown"
    return f"[image you sent earlier, sha256 {digest}]"


def _strip_inline_images(messages: list[dict]) -> list[dict]:
    """Clients send the whole conversation back, including the images we
    returned: in content as markdown, in `images`, or in the singular `image`
    field the Playground uses. Only ASSISTANT messages are touched (our own
    generated images and progress lines); anything a user sends reaches the
    model unchanged.
    Each of our images becomes a compact sha reference in the content, so an
    earlier picture doesn't drag megabytes of base64 into later turns."""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        m = dict(m)
        content = m.get("content")
        if isinstance(content, str):
            content = voice_clip.strip(progress.strip_leading(content))
        markers = []
        if isinstance(content, str) and "](data:image/" in content:
            content = INLINE_IMAGE.sub(lambda g: _marker_for(g.group(0).split("(", 1)[1][:-1]), content)
        urls = [im.get("image_url", {}).get("url", "") for im in (m.pop("images", None) or []) if isinstance(im, dict)]
        single = m.pop("image", None)
        if isinstance(single, dict):
            urls.append(single.get("url", ""))
        for url in dict.fromkeys(u for u in urls if u.startswith("data:image/")):
            marker = _marker_for(url)
            if not (isinstance(content, str) and marker in content):
                markers.append(marker)
        if markers:
            content = ((content or "") + "\n\n" + " ".join(markers)).strip() if not isinstance(content, list) else content + [{"type": "text", "text": " ".join(markers)}]
        m["content"] = content
        out.append(m)
    return out


def _image_urls(artifacts: list[dict], store: ArtifactStore, settings) -> list[str]:
    """One URL per generated image. With PUBLIC_ARTIFACT_BASE set, a signed,
    expiring link to /v1/artifacts/{id} that a browser can load without our
    API key. Without it (the container has no public route of its own), the
    registered bytes as a data: URI."""
    out = []
    for a in artifacts:
        if a.get("type") != "image":
            continue
        found = store.locate(a["id"])
        if not found:
            continue
        path, mime = found
        base = settings.public_artifact_base
        if base and settings.artifact_signer_key:
            expires = int(time.time()) + settings.artifact_url_ttl_s
            sig = artifact_links.signature(settings.artifact_signer_key, a["id"], expires)
            out.append(f"{base.rstrip('/')}/v1/artifacts/{a['id']}?expires={expires}&sig={sig}")
        else:
            out.append(f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode())
    return out


def _search_query(state: dict) -> str | None:
    """The query a completed web search ran, for a door that reports it (Responses)."""
    result = state.get("result")
    if result is not None and result.provenance.get("kind") == "search":
        return result.provenance.get("query") or ""
    return None


def _image_blobs(artifacts: list[dict], store: ArtifactStore) -> list[tuple[str, bytes]]:
    """The turn's generated images as (mime, bytes), for a door that delivers
    them as structured items instead of markdown (Responses)."""
    out = []
    for a in artifacts:
        if a.get("type") == "image" and (found := store.locate(a["id"])):
            path, mime = found
            out.append((mime, path.read_bytes()))
    return out


def _audio_error(exc) -> JSONResponse:
    """A voice message nobody could hear is an error, not a reply from the assistant:
    the caller must be able to tell a failure from an answer."""
    if exc.backend:
        return JSONResponse({"error": {"message": "the voice message could not be transcribed right now",
                                       "type": "server_error", "param": None, "code": "transcription_unavailable"}},
                            status_code=502)
    return error(400, "the newest voice message could not be heard", "audio_unintelligible", "messages")


def register(app: FastAPI, deps, stored: ResponseStore) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except (ValueError, RecursionError):
            return error(400, "body is not JSON", "invalid_json")
        if isinstance(body, dict) and not _single_choice(body.get("n")):
            return await run_chat_n(body, request)
        return await run_chat(body, request)

    async def run_chat_n(body: dict, request: Request):
        """`n` above 1 (S13g): n independent single-choice turns, run together and returned as
        one completion. Nothing is routed to a specialist (several pictures or searches for one
        ask is not what `n` means), and `store` keeps the merged object once."""
        _, err = _validate(body, deps.settings)
        if err:
            return err
        n = body["n"]
        one = {k: v for k, v in body.items() if k not in ("n", "store", "stream", "stream_options")}
        replies = await asyncio.gather(*(run_chat(dict(one), request, allowed_routes=frozenset()) for _ in range(n)))
        for r in replies:
            if r.status_code != 200:
                return r
        payloads = [json.loads(bytes(r.body)) for r in replies]
        merged = dict(payloads[0])
        merged["choices"] = [{**p["choices"][0], "index": i} for i, p in enumerate(payloads)]
        if "metadata" in body or body.get("store") is True:
            # The sub-turns deliberately do not see `store`; the outer turn owns
            # the one stored object and therefore owns its metadata echo too.
            merged["metadata"] = body.get("metadata") or {}
        usages = [p.get("usage") for p in payloads]
        if all(isinstance(u, dict) for u in usages):
            done = sum(u.get("completion_tokens", 0) for u in usages)
            prompt = usages[0].get("prompt_tokens", 0)     # the prompt is one prompt, counted once
            merged["usage"] = {"prompt_tokens": prompt, "completion_tokens": done, "total_tokens": prompt + done}
        headers = {k: replies[0].headers[k] for k in ("x-chord-trace-id", "x-request-id") if k in replies[0].headers}
        if body.get("store") is True:
            stored.put_chat(merged, body.get("metadata") or {}, body["messages"])
        if not body.get("stream"):
            return JSONResponse(merged, headers=headers)

        # Streamed: the n turns are whole before anything is sent (each is guarded as a whole
        # reply), so every choice arrives as one delta and its finish, then usage when asked.
        # Spec-shaped chunks; the words of a choice do not trickle in.
        include_usage = ((body.get("stream_options") or {}).get("include_usage")) is True
        base = {"id": merged["id"], "object": "chat.completion.chunk", "created": merged["created"], "model": merged["model"]}
        if "system_fingerprint" in merged:
            base["system_fingerprint"] = merged["system_fingerprint"]

        def frame(choices: list, usage=None) -> str:
            chunk = {**base, "choices": choices}
            if include_usage:
                chunk["usage"] = usage
            return f"data: {json.dumps(chunk)}\n\n"

        async def sse() -> AsyncIterator[str]:
            for choice in merged["choices"]:
                message = choice["message"]
                delta = {k: v for k, v in message.items() if k in ("role", "content", "refusal", "tool_calls", "function_call") and v is not None}
                if "tool_calls" in delta:
                    delta["tool_calls"] = [{"index": j, **call} for j, call in enumerate(delta["tool_calls"])]
                yield frame([{"index": choice["index"], "delta": delta, "logprobs": choice.get("logprobs"), "finish_reason": None}])
                yield frame([{"index": choice["index"], "delta": {}, "logprobs": None, "finish_reason": choice["finish_reason"]}])
            if include_usage and merged.get("usage"):
                yield frame([], merged["usage"])
            yield "data: [DONE]\n\n"

        return ClosingStreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    async def run_chat(body, request: Request, *, allowed_routes: frozenset | None = None, sink: dict | None = None,
                       watch_client: bool = True):
        """The Chat Completions core, shared by the Responses API (responses.py).
        `allowed_routes`: the specialist routes this request offers (Responses
        offers them as built-in tools); None means the chat door's own rules.
        `sink`: when given, generated images are handed over in it as
        (mime, bytes) instead of markdown in content, and the trace rides in it.
        `watch_client`: cancel a non-stream turn whose client disconnects (#8). A
        background Response passes False: its client already has the queued answer,
        so the request reads as disconnected and the turn must outlive it."""
        persona_id, err = _validate(body, deps.settings)
        if err:
            return err
        if "web_search_options" in body and "search" not in deps.capabilities:
            return error(503, "web search is not configured", "capability_unavailable", "web_search_options")
        assert persona_id is not None
        from .progress_replay import count_ingress
        ingress_progress = count_ingress(body["messages"])
        params, messages, client_instructions = _split(body)
        stream = bool(body.get("stream"))
        trace = Trace(persona_id=persona_id, model_id_requested=body["model"])
        trace.set(client_instruction_messages=client_instructions, stream=stream, params=sorted(params))
        trace.set(ingress_progress=ingress_progress)
        # The alpha evidence for the image_url allowlist: which hosts allowed
        # http(s) parts actually named, on every turn that carried one.
        if (forwarded_hosts := _image_url_forwarded_hosts(messages)):
            trace.set(image_url_forwarded_hosts=forwarded_hosts)
        # Who the caller says the end user is (#124): attribution now, per-user
        # moderation and limits with #121. Opaque ids, validated above.
        trace.set(safety_identifier=body.get("safety_identifier"), end_user=body.get("user"))
        headers = {"x-chord-trace-id": trace.trace_id, "x-request-id": trace.trace_id}

        async def hear():
            with trace.timed("stt"):
                return await deps.transcriber.transcribe_messages(messages, trace)

        voice = deps.settings.voice_for(persona_id)
        wants_audio = "audio" in (body.get("modalities") or [])
        if wants_audio:
            bad = audio.check_audio_request(body, stream, voice)
            if bad:
                return error(400, bad, "invalid_audio_request", "audio")
        messages = deps.speaker.replayed(messages)

        async def spoken(text: str) -> dict:
            """Message fields for the spoken reply."""
            if not (wants_audio and isinstance(text, str) and text.strip()):
                return {}
            with trace.timed("tts"):
                said = await deps.speaker.speak(text, voice, body["audio"]["format"], trace)
            if said is None:
                return {}  # the reply still goes out as text; message.audio is absent (traced)
            message_audio, descriptor = said
            trace.artifacts.append(descriptor)
            return {"audio": message_audio}

        def audio_failure(exc: audio.AudioError) -> JSONResponse:
            # An STT failure ends the turn; never proceed as though the user said nothing.
            trace.set(stt_failed=str(exc), result_status=Outcome.failed.value)
            deps.traces.write(trace)
            r = _audio_error(exc)
            r.headers.update(headers)
            return r

        stop = asyncio.Event()
        graph = graph_mod.build(
            settings=deps.settings, capabilities=deps.capabilities, artifacts=deps.artifacts,
            upstream=deps.upstream, model=deps.model, trace=trace, stop=stop,
            image_backend=deps.image_backend,
        )
        route = manifest.load()["service_tier_routes"].get(body.get("service_tier") or "")
        persona_model = deps.settings.tier_model(route["slot"]) if route else None
        # Every accepted request: the service owns verbosity, omitted included.
        trace.set(verbosity_requested=body.get("verbosity"), verbosity_effective=body.get("verbosity") or "medium",
                  verbosity_handled_by="service")
        state_in: graph_mod.TurnState = {
            "params": params,
            "messages": messages,
            "stream": stream,
            "verbosity": body.get("verbosity"),
            "persona_model": persona_model,
            "audio_output": wants_audio,
            "allowed_routes": allowed_routes,
            "count_only": bool(sink and sink.get("count_only")),
        }
        if sink is not None:
            sink["trace"] = trace
            trace.set(**sink.get("trace_fields", {}))   # the calling door's own identity (Responses)
        tier_echo = route["echo"] if route else None
        template_kwargs = body.get("chat_template_kwargs")
        thinking = (isinstance(template_kwargs, dict) and template_kwargs.get("enable_thinking") is True) \
            or bool(sink and sink.get("keep_reasoning"))   # Responses: reasoning as a spec reasoning item
        stream_options = body.get("stream_options")
        include_usage = isinstance(stream_options, dict) and stream_options.get("include_usage") is True
        created = int(time.time())
        cid = f"chatcmpl-{trace.trace_id}"
        forced = _forced_call(body)
        # review 2026-09-24 B15: a stored stream keeps its usage whether or not the client
        # asked for it, so the backend is asked for it here. What the client receives is
        # unchanged: `frame` strips the usage it did not ask for.
        usage_for_store = stream and sink is None and body.get("store") is True and not include_usage
        if usage_for_store:
            state_in["params"] = {**params, "stream_options": {**(stream_options or {}), "include_usage": True}}

        if not stream:
            try:
                state_in["messages"] = await hear()
            except audio.AudioError as exc:
                return audio_failure(exc)
            async def invoke(turn: graph_mod.TurnState):
                """One draw: the final state, or the error response that ends the turn."""
                work = asyncio.create_task(graph.ainvoke(turn))
                if watch_client and await _cancelled_by_disconnect(work, request):
                    trace.set(client_disconnected=True)
                    deps.traces.write(trace)
                    return None, Response(status_code=499)  # socket is gone; no fallback work
                try:
                    return await work, None
                except UpstreamError as exc:
                    trace.set(upstream_error=str(exc), upstream_error_body=exc.body[:8000])
                    deps.traces.write(trace)
                    return None, JSONResponse(_upstream_error_body(exc, body), status_code=_upstream_http_status(exc),
                                              headers=headers)
                except httpx.HTTPError as exc:
                    return None, _unreachable(deps, trace, exc, headers)

            state, failed = await invoke(state_in)
            if state is None:
                assert failed is not None
                return failed
            declared = _declared_functions(body)
            if forced:
                reason = _forced_call_violation(forced, _message_calls(state.get("message") or {}), declared)
                if reason:
                    # A pre-header refusal traverses the boundary like any other response,
                    # so it carries the canonical row too. Without this, "every response"
                    # silently means "every response that got far enough to succeed", and
                    # the refusals -- the population we most want to count -- are the ones
                    # missing from the denominator (2026-09-19).
                    _record_normalization(trace, body,
                                          _message_calls_shaped(state.get("message") or {}),
                                          transport="nonstream", sse_data_emitted=False,
                                          content_delta_emitted=False, disposition="unrepaired")
                    return _forced_call_error(forced, reason, trace, deps, headers)
            # NOT `if declared:` -- that skips validation exactly when the declared set is
            # empty, which is the inverse of correct: nothing is a member of an empty set,
            # so a returned call with no declared tools is the clearest violation there is.
            # a review identified this shape as a falsifier before the test existed; the implementation had written
            # it (#289 review).
            calls = _message_calls(state.get("message") or {})
            first_draw: list = []      # the pre-retry observation, when a retry happens
            retried = bool(calls) and _no_tool_parameter(body)
            if retried:
                # THE BOUNDED TEXT-ONLY RETRY, as the stream path takes it (review
                # 2026-09-24 B14). No tool parameter at all, so every call is invalid, and
                # nothing has been sent: a second draw is still free. Refusing here while the
                # stream retried made one request a 502 unstreamed and usually a 200
                # streamed. Same mechanism, not a second one: draw 1's finished work is
                # carried and only the persona pass is re-drawn (graph entry edge), ONE
                # retry, and recovery is positive -- prose present and no call.
                first_draw = _message_calls_shaped(state.get("message") or {})
                trace.set(undeclared_nonstream_prebyte=True, prebyte_retry_attempted=True)
                retry_state = cast(graph_mod.TurnState,
                                   {**state_in, **graph_mod.retry_carry([("values", state)]),
                                    "retry_draw": True})
                state, failed = await invoke(retry_state)
                if state is None:
                    assert failed is not None
                    return failed
                calls = _message_calls(state.get("message") or {})
                content = (state.get("message") or {}).get("content")
                recovered = not calls and isinstance(content, str) and content != ""
                trace.set(prebyte_retry_recovered=recovered)
                if not recovered:
                    _record_normalization(trace, body,
                                          first_draw + _message_calls_shaped(state.get("message") or {}),
                                          transport="nonstream", sse_data_emitted=False,
                                          content_delta_emitted=False, disposition="unrepaired")
                    return _undeclared_call_error(_undeclared_names(calls, declared) or ["<no-tools>"],
                                                  trace, deps, headers)
            undeclared = _undeclared_names(calls, declared)
            repaired = _bind_to_sole_tool(calls, body, trace) if undeclared else None
            # THE DENOMINATOR. Written on every response through this door, clean ones
            # included -- recording only repairs and refusals gives numerators with no
            # base population, which is exactly why 2/222 could not price a neighbouring
            # branch, and it cannot be fixed later because the clean rows were never
            # written (the amendment; this remained blocker 3).
            _record_normalization(trace, body,
                                  first_draw if retried else _message_calls_shaped(state.get("message") or {}),
                                  transport="nonstream",
                                  sse_data_emitted=False, content_delta_emitted=False,
                                  disposition=("sole_tool_bind" if repaired else
                                               "unrepaired" if undeclared else
                                               "prebyte_retry" if retried else "none"))
            if undeclared and repaired is None:
                return _undeclared_call_error(undeclared, trace, deps, headers)
            message = dict(state.get("message", {"role": "assistant", "content": state.get("text", "")}))
            if repaired:
                # One declared tool, one returned call, and the model's OWN arguments
                # already satisfy that tool's schema. Rename rather than refuse: the
                # caller sees a compatible call to the only function they offered, and
                # never learns a smaller model needed help (option D, decision).
                message = _rename_call(message, *repaired)
            content = message.get("content")
            message.update(await spoken(content if isinstance(content, str) else ""))
            if sink is not None:
                sink["images"] = _image_blobs(_turn_artifacts(state), deps.artifacts)
                sink["search_query"] = _search_query(state)
            elif urls := _image_urls(_turn_artifacts(state), deps.artifacts, deps.settings):
                message["content"] = (message.get("content") or "") + _markdown(urls)
            # Phase-1 shadow moderation (#121): score the last user message and
            # the assistant reply, write the verdicts to the trace row, do NOT
            # modify the wire. The streaming path does not score here -- output
            # moderation on a fragmented stream is a Phase-3 decision once we
            # have FP/FN data. Both scoring calls run in parallel; the stub
            # adds sub-millisecond latency and a real guard's cost is bounded
            # by whichever side is slower.
            from .moderation import score as _shadow_score
            input_text = _last_user_text(body.get("messages") or [])
            output_text = content if isinstance(content, str) else ""
            in_verdict, out_verdict = await asyncio.gather(
                _shadow_score(input_text),
                _shadow_score(output_text),
            )
            trace.set(moderation_scored=True, moderation_input=in_verdict.as_trace_field(),
                      moderation_output=out_verdict.as_trace_field(),
                      moderation_guard=in_verdict.guard_model,
                      moderation_policy=in_verdict.policy_name)
            deps.traces.write(trace)
            payload = {
                "id": cid, "object": "chat.completion", "created": created, "model": body["model"],
                "choices": [{"index": 0, "message": message, "logprobs": state.get("logprobs"),
                             "finish_reason": state.get("finish_reason") or "stop"}],
            }
            if "metadata" in body or body.get("store") is True:
                # The store canonicalises null to the empty map. Return that same
                # object now, including when the caller omitted metadata, so
                # create and retrieve cannot disagree about bytes.
                payload["metadata"] = body.get("metadata") or {}
            if state.get("usage"):
                payload["usage"] = state["usage"]
            if tier_echo:
                payload["service_tier"] = tier_echo  # the tier actually used (pinned spec)
            if state.get("system_fingerprint") is not None:
                # Any string, "" included: absent and empty are different answers.
                payload["system_fingerprint"] = state["system_fingerprint"]
            if not thinking:
                payload = _drop_reasoning(payload, stream=False)
            payload = _conform_choices(payload, stream=False)
            if sink is None and body.get("store") is True:
                stored.put_chat(payload, body.get("metadata") or {}, body["messages"])
            return JSONResponse(payload, headers=headers)

        async def heard_then_events():
            # Transcription is the first step of the startup _prime_stream
            # races against a disconnect, so a client that leaves mid-STT
            # cancels it (#33).
            state_in["messages"] = await hear()
            inner = _events(graph, state_in, stop)
            try:
                async for event in inner:
                    yield event
            finally:
                await inner.aclose()

        events = heard_then_events()
        # Run until the model's first chunk before answering, so an upstream
        # refusal is still a real HTTP error (errors before the
        # response starts). `values` events that arrive first are buffered.
        try:
            # Buffer the whole stream when a name will have to be validated before any
            # byte goes out: forced (S03), or ANY request that declares tools (#289).
            # Validation after emission is an apology -- the invalid name is already on
            # the wire by the time it is known to be invalid. `_nested()` cannot
            # be the choke point: it sees each delta, a name may be fragmented across
            # several, and it does not delay emission.
            #
            # This imposes buffering latency on tool-enabled auto streams. That is a real
            # cost and it is recorded rather than hidden: `tool_stream_buffered` on the
            # trace marks every request that paid it.
            declared = _declared_functions(body)
            # NOT `bool(declared)`: an empty declared set must still be validated, because
            # nothing is a member of an empty set (F1). The trigger is whether the
            # request carries ANY tool parameter at all -- the shortcut `bool(declared)`
            # made "no tools" mean "skip buffering AND skip validation", the same shortcut
            # twice in series, on the one path I was not looking at.
            tool_params = any(k in body for k in TOOL_PARAMS)
            # NO TOOL PARAMETER AT ALL is its own case, and it is the one the six open
            # matrix cells named: nothing triggered buffering, so an invalid call could
            # cross the response boundary after prose. It does NOT need buffering to fix.
            # Every returned call is invalid here whatever its spelling -- nothing is a
            # member of an empty set -- so the decision wants no name and no arguments,
            # only which arrived first. Priming to the first decisive delta keeps ordinary
            # chat streams at their old latency: whole-call quarantine is paid by requests
            # that declare tools, not by everyone (option D; the amendment at dfa85af).
            no_tools = normalize.tool_parameter_state(body) == normalize.ABSENT
            # Bound here, not inside the `tool_params` branch below. Left unbound, the
            # no-tools path raised NameError from inside `handle()`, the broad stream-error
            # handler caught it, and the response LOOKED like a correct suppression --
            # 200, no `[DONE]`. Only the trace showed it. A failure that imitates the
            # success it replaced is the worst shape there is, and the reason for
            # one canonical trace row per response.
            repaired_call = None
            first_draw: list = []      # the pre-retry observation, when a retry happens
            pending = await _prime_stream(events, request, stop,
                                          whole=bool(forced) or tool_params,
                                          until_decisive=no_tools and not tool_params)
            if pending is None:
                trace.set(client_disconnected=True)
                deps.traces.write(trace)
                return Response(status_code=499)  # socket is gone; no fallback work
            if forced:
                # The whole forced stream is in hand: replay it, or refuse
                # before a single byte goes out (S03). Forced streams arrive
                # late by design.
                trace.set(forced_stream_buffered=True)
                # The graph repaired a missing forced call (#216, forced_call.py): what the
                # model streamed before its marker is void, and only the repaired call goes out.
                marks = [i for i, (mode, data) in enumerate(pending)
                         if mode == "custom" and isinstance(data, dict) and data.get("forced_call_repair")]
                if marks:
                    pending = [e for e in pending[:marks[-1]] if not (e[0] == "custom" and isinstance(e[1], dict) and "chunk" in e[1])] \
                        + pending[marks[-1] + 1:]
                reason = _forced_call_violation(forced, _stream_calls(pending), declared)
                if reason:
                    _record_normalization(trace, body, _stream_calls_shaped(pending),
                                          transport="stream", sse_data_emitted=False,
                                          content_delta_emitted=False, disposition="unrepaired")
                    return _forced_call_error(forced, reason, trace, deps, headers)
            if tool_params:
                # Assembled, not per-delta: a fragmented name is only a name once the
                # buffered stream is whole. Unconditional in `declared` -- see above.
                trace.set(tool_stream_buffered=True)
                stream_calls = _stream_calls(pending)
                undeclared = _undeclared_names(stream_calls, declared)
                repaired_call = _bind_to_sole_tool(stream_calls, body, trace) if undeclared else None
                if undeclared and repaired_call is None:
                    _record_normalization(trace, body, _stream_calls_shaped(pending),
                                          transport="stream", sse_data_emitted=False,
                                          content_delta_emitted=False, disposition="unrepaired")
                    return _undeclared_call_error(undeclared, trace, deps, headers)
            elif no_tools and any(normalize.CALL in _decisive(e) for e in pending):
                # A call arrived and NOTHING has reached the client: priming stops at the
                # first decisive delta, and a call being decisive means no content delta
                # preceded it. So this is still pre-header and the honest answer is an
                # ordinary HTTP error, after the bounded retry below -- the same retry and
                # the same 502 the non-stream path gives (review 2026-09-24 B14).
                trace.set(undeclared_stream_prebyte=True)

                # THE BOUNDED TEXT-ONLY RETRY (the amendment). Nothing has reached the
                # client, so a second draw is still free: the request declared no tools,
                # the backend produced a call anyway, and the condition is stochastic --
                # `weather` and `weather_getCurrent_2604` were 2 of 222 on one frozen
                # body, so most draws do not do this.
                #
                # ONE retry, not a loop. An unbounded retry against a stochastic fault
                # turns a rare defect into an unbounded latency bill on the exact turns
                # that are already going badly.
                #
                # The first stream is closed before the second opens: leaving it running
                # would keep an upstream generation alive that nothing will ever read
                # (`stop` is what actually halts the node -- see `_events`).
                await events.aclose()
                # CLEAR the original event; do NOT make a new one. `_events` sets `stop`
                # when its generator closes early, and that set flag is what halts an
                # in-flight node. But the GRAPH captured this exact event object at build
                # time, so handing the retry a fresh Event leaves the node still checking
                # the old, already-set one: the second draw produced nothing, and "no
                # call" and "no response" are the same shape to the check below -- it
                # reported RECOVERED on an empty stream. Two upstream draws, a bare finish
                # frame, and a green trace bit (the capture and a second probe
                # produced the false green, 2026-09-19).
                # The FIRST draw's observation, kept before `pending` is overwritten.
                # Without it a successful retry records `call_observation: none` and the
                # invalid call that caused the retry disappears -- every repair erasing
                # its own numerator, so the instrument would report repairs against no
                # observed defect.
                first_draw = _stream_calls_shaped(pending)
                stop.clear()
                trace.set(prebyte_retry_attempted=True)
                # Carry draw 1's finished work into draw 2 and enter at the
                # persona node (graph entry edge): the router is stochastic,
                # so re-running it could re-decide the turn -- a search that
                # came back chat would silently drop its own citations -- and
                # a specialist re-run burned a second GPU render and orphaned
                # the first one's artifact. Only the persona pass leaked the
                # call, so only the persona pass is re-drawn (batch 3).
                retry_state = cast(graph_mod.TurnState,
                                   {**state_in, **graph_mod.retry_carry(pending),
                                    "retry_draw": True})
                events = _events(graph, retry_state, stop)
                # WHOLE, not until-decisive. The retry is still PRE-BYTE: if draw two
                # emits prose and then a call, releasing the prose first would commit the
                # response and force an in-band error, when the amendment says a failed
                # bounded retry is an ordinary HTTP error. Buffering the rare path costs
                # nothing anyone feels -- it only runs on a turn already going wrong --
                # and it makes "a call anywhere in draw two" a pre-header failure.
                pending = await _prime_stream(events, request, stop, whole=True)
                if pending is None:
                    trace.set(client_disconnected=True)
                    deps.traces.write(trace)
                    return Response(status_code=499)
                # RECOVERY IS POSITIVE, not the absence of a call. A retry that returns
                # nothing has not produced a text-only answer, and accepting it would
                # hand the caller an empty success in place of an actionable error.
                # Recovery is POSITIVE and whole-stream: content present AND no call
                # anywhere in the buffered draw. "No call" alone is satisfied by an empty
                # stream, which is not a text-only answer -- it is nothing, dressed as a
                # success.
                kinds = set().union(*(_decisive(e) for e in pending)) if pending else set()
                recovered = normalize.CONTENT in kinds and normalize.CALL not in kinds
                trace.set(prebyte_retry_recovered=recovered)
                if not recovered:
                    _record_normalization(trace, body, first_draw + _stream_calls_shaped(pending),
                                          transport="stream", sse_data_emitted=False,
                                          content_delta_emitted=False,
                                          disposition="unrepaired")
                    return _undeclared_call_error(
                        _undeclared_names(_stream_calls(pending), declared) or ["<no-tools>"],
                        trace, deps, headers)

        except asyncio.CancelledError:
            trace.set(request_cancelled=True)
            deps.traces.write(trace)
            raise
        except audio.AudioError as exc:
            return audio_failure(exc)
        except UpstreamError as exc:
            trace.set(upstream_error=str(exc), upstream_error_body=exc.body[:8000])
            deps.traces.write(trace)
            return JSONResponse(_upstream_error_body(exc, body), status_code=_upstream_http_status(exc), headers=headers)
        except httpx.HTTPError as exc:
            return _unreachable(deps, trace, exc, headers)

        async def sse() -> AsyncIterator[str]:
            # The canonical row is written by `_emit`'s outer finally, AFTER every yield.
            # Written in the inner one it preceded the repaired call, the held finish, the
            # usage chunk and `[DONE]` -- so a call-only sole-tool bind recorded
            # `client_sse_data_emitted=False` and then emitted the repaired call. The row
            # must describe what crossed the boundary, and only a scope enclosing every
            # yield can know that (2026-09-19).
            async for out in _emit():
                yield out

        async def _emit() -> AsyncIterator[str]:
            suppressed: list = []
            emitted_data = False
            emitted_content = False
            try:
                held_usage, held_finish, final_state = None, None, {}
                completed = False
                # Calls suppressed on the no-tools path, after prose already went out. The
                # list is the evidence; its emptiness is what lets the ordinary ending run.
                # `emitted_data` covers any SSE payload; `emitted_content` is
                # specifically prose and carries the orphaned-promise concern.
                keep = [] if sink is None and body.get("store") is True else None

                def frame(chunk: dict) -> str:
                    if not thinking:
                        chunk = _drop_reasoning(chunk, stream=True)
                    chunk = _conform_choices(chunk, stream=True)
                    if tier_echo:
                        chunk = {**chunk, "service_tier": tier_echo}
                    if include_usage and "usage" not in chunk:
                        # With include_usage, every chunk but the usage one carries
                        # usage: null (S12, red team 2026-09-15).
                        chunk = {**chunk, "usage": None}
                    if usage_for_store:
                        chunk = {k: v for k, v in chunk.items() if k != "usage"}   # asked for by us, not the client (B15)
                    sent = {**chunk, 'id': cid, 'model': body['model'], 'created': created}
                    if keep is not None:
                        keep.append(sent)
                    return f"data: {json.dumps(sent)}\n\n"

                def handle(event) -> str | None:
                    nonlocal held_usage, held_finish, final_state, emitted_content
                    mode, data = event
                    if mode == "values":
                        final_state = data
                        return None
                    if "chunk" not in data:
                        return None
                    chunk = data["chunk"]
                    if repaired_call and normalize.CALL in normalize.delta_kinds(chunk):
                        # The buffered stream held an unauthorised call whose raw arguments
                        # satisfy the sole declared tool. Its fragments are dropped here and
                        # the repaired call is emitted whole below, so the client never sees
                        # the name the backend actually chose -- compatible ingress and
                        # egress, unaware of the fix inside (option D, decision).
                        chunk = _without_calls(chunk)
                        if not any(c.get("delta") or c.get("finish_reason")
                                   for c in (chunk.get("choices") or [])):
                            return None
                    if no_tools and normalize.CALL in normalize.delta_kinds(chunk):
                        # The request declared no tool parameter, so this call is unauthorised
                        # whatever it is named. Prose has already crossed the boundary -- that
                        # is why we are here rather than in the pre-byte refusal above -- and a
                        # delivered sentence cannot be recalled. So the call fields are dropped
                        # and any prose riding the same delta still goes out; the turn ends in
                        # an explicit error below rather than pretending it completed.
                        suppressed.append(chunk)
                        # Set here, inside the try, so the single `finally` write carries it.
                        # The earlier version set it AFTER that write and wrote again: two
                        # JSONL rows with one trace id, the first with result_status None.
                        # A trace that can appear twice destroys the denominator the amendment
                        # is built on -- you cannot count responses by counting rows.
                        trace.set(result_status=Outcome.failed.value)
                        chunk = _without_calls(chunk)
                        if not any(c.get("delta") or c.get("finish_reason")
                                   for c in (chunk.get("choices") or [])):
                            return None
                    if normalize.CONTENT in normalize.delta_kinds(chunk):
                        emitted_content = True
                    choices = chunk.get("choices") or []
                    # Hold a usage-only chunk so it stays last, as clients expect.
                    # LiteLLM sends it as choices=[{"delta": {}}], not choices=[].
                    if chunk.get("usage") and all(not c.get("delta") and not c.get("finish_reason") for c in choices):
                        held_usage = chunk
                        trace.set(stream_usage=chunk["usage"])  # kept for us, sent only if asked
                        return None
                    # Hold the model's finish: a generated image goes out after
                    # the assistant's last words and before it. A finish chunk can carry words
                    # of its own, so those go out now and only the finish is held.
                    if any(c.get("finish_reason") for c in choices):
                        # The content half carries the tokens, so it keeps logprobs.
                        # The finish half must not: both used to be copies of the same
                        # choice, and a client concatenating logprobs.content counted
                        # those tokens twice. usage on this mixed chunk is held for the
                        # trailing usage frame, and only when the client asked for it.
                        words = [{k: v for k, v in c.items() if k != "finish_reason"} for c in choices if c.get("delta")]
                        held_finish = {k: v for k, v in chunk.items() if k != "usage"}
                        # Drop logprobs from the finish half only when the content half
                        # already carried them. A finish chunk whose delta is empty is
                        # the only copy.
                        held_finish["choices"] = [
                            {k: v for k, v in c.items() if k != "logprobs" or not c.get("delta")} | {"delta": {}}
                            for c in choices
                        ]
                        if chunk.get("usage") and held_usage is None:
                            held_usage = chunk   # sent only if asked, stored either way (B15)
                        words_chunk = {k: v for k, v in chunk.items() if k != "usage"}
                        words_chunk["choices"] = words
                        return frame(words_chunk) if words else None
                    return frame(chunk)

                try:
                    for event in pending:
                        out = handle(event)
                        if out:
                            emitted_data = True
                            yield out
                    async for event in events:
                        out = handle(event)
                        if out:
                            emitted_data = True
                            yield out
                    completed = True
                except Exception as exc:  # after headers: an OpenAI stream error event, never a reply
                    trace.set(stream_error=repr(exc)[:500], result_status=Outcome.failed.value)
                    completed = True
                    failure = {"error": {"message": "the response failed while streaming", "type": "server_error",
                                         "param": None, "code": "stream_failed"}}
                    # The error object IS client-visible data. Failing before any ordinary
                    # frame used to leave the row saying nothing crossed while this went
                    # out -- the same accounting gap as the repaired call, on the path
                    # least likely to be exercised.
                    emitted_data = True
                    yield f"data: {json.dumps(failure)}\n\n"
                    return
                finally:
                    # Runs on normal completion and when the client leaves (the
                    # response closes this generator). Closing the graph's stream
                    # tears down the upstream request, so generation stops.
                    if not completed:
                        trace.set(client_disconnected=True)
                    await events.aclose()
                if repaired_call:
                    # Whole, after the prose and before the finish -- a client assembles
                    # fragments, and one complete call is the simplest honest thing to send.
                    name, arguments = repaired_call
                    emitted_data = True
                    yield frame({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {
                        "tool_calls": [{"index": 0, "id": f"call_{str(ULID()).lower()}", "type": "function",
                                        "function": {"name": name, "arguments": arguments}}]}}]})
                if suppressed:
                    # The 2026-09-19 decision: an explicit in-band error object, then stop.
                    # No `[DONE]`, no terminal `finish_reason: stop`.
                    #
                    # The ordering is load-bearing, not tidiness. `openai-python` 3.13.0 and
                    # `openai-node` 6.45.0 both check `[DONE]` FIRST and return; an error frame
                    # after it is unreachable and the stream ends looking complete. Both raise
                    # on an in-band `error` object, and neither touches `choices` before doing
                    # so, so a frame without one is fine for them (both SDKs were checked at those
                    # versions; LiteLLM, the Vercel SDK and hand-rolled parsers are UNVERIFIED).
                    #
                    # `FORCED_CALL_CODE` is reused deliberately. It is the published category
                    # for this event; two regression files already assert it:
                    # the backend claimed a call it was not entitled to make. Same code, new
                    # transport (an example was mistaken for a rename in review).
                    emitted_data = True
                    yield f"data: {json.dumps({'error': {'message': UNDECLARED_CALL_MESSAGE, 'type': 'server_error', 'param': 'tools', 'code': FORCED_CALL_CODE}})}\n\n"
                    return
                if sink is not None:
                    sink["images"] = _image_blobs(_turn_artifacts(final_state), deps.artifacts)
                    sink["search_query"] = _search_query(final_state)
                    urls = []
                else:
                    urls = _image_urls(_turn_artifacts(final_state), deps.artifacts, deps.settings)
                if urls:
                    # Image markdown is CONTENT the client can see. The field describes
                    # client-visible content, not merely upstream chunks -- and this is
                    # prose we generated, which is exactly the kind the orphaned-promise
                    # concern is about.
                    emitted_data = True
                    emitted_content = True
                    yield frame({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": _markdown(urls)}}]})
                finish = held_finish or {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                emitted_data = True
                yield frame(finish)
                if held_usage and include_usage:
                    # The spec's usage chunk has choices: [], not LiteLLM's empty
                    # pseudo-choice, and comes after the finish chunk. Unrequested,
                    # it is not sent at all (S12; #172).
                    yield frame({**held_usage, "choices": []})
                if keep:
                    completion = stored_chat.completion_from_chunks(keep)
                    assert completion is not None
                    if held_usage and held_usage.get("usage") and "usage" not in completion:
                        # Not sent (the client did not ask), still the completion's usage,
                        # as the non-stream store keeps it (review 2026-09-24 B15).
                        completion["usage"] = _conform_choices({"usage": held_usage["usage"]}, stream=True)["usage"]
                    stored.put_chat(completion, body.get("metadata") or {}, body["messages"])
                yield "data: [DONE]\n\n"
            finally:
                # THE DENOMINATOR, after every yield. Clean rows included: recording only
                # repairs and refusals gives numerators with no base population, and the
                # clean rows cannot be reconstructed later because they were never written.
                _record_normalization(
                    trace, body,
                    (first_draw if trace.fields.get("prebyte_retry_recovered")
                     else _stream_calls_shaped(pending))
                    + _stream_calls_shaped([("custom", {"chunk": c}) for c in suppressed]),
                    transport="stream",
                    sse_data_emitted=emitted_data, content_delta_emitted=emitted_content,
                    # `drop` is the intermediate ACTION; `extension_error` is the terminal
                    # OUTCOME the design chose. The enum keeps both so they stay distinguishable
                    # -- reserve `drop` for a turn normalized by omission alone, without
                    # the public error contract.
                    disposition=("sole_tool_bind" if repaired_call else
                                 "extension_error" if suppressed else
                                 "prebyte_retry" if trace.fields.get("prebyte_retry_attempted")
                                 else "none"))
                deps.traces.write(trace)


        return ClosingStreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    app.state.run_chat = run_chat


class _Done:
    pass


async def _events(graph, state_in: graph_mod.TurnState, stop: asyncio.Event):
    """Run the graph in its own task and relay its events.

    When this generator is closed early (the client left), it sets `stop`,
    which the streaming node checks on every chunk, and then cancels the
    producer. `stop` is the guarantee. Cancelling the producer alone was tested
    and did NOT stop an in-flight node inside a live request: the model kept
    generating until server shutdown (tests/test_disconnect.py, red without
    the stop check).
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        try:
            async for event in graph.astream(state_in, stream_mode=["custom", "values"]):
                await queue.put(event)
            await queue.put(_Done)
        except BaseException as exc:  # relay failures, including cancellation
            await queue.put(exc)
            raise

    task = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if item is _Done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()  # the node notices on its next chunk and closes the upstream
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task
