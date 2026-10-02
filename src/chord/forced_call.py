"""A forced tool call the backend can't force (#216).

`tool_choice: "required"`, a named function, `allowed_tools` with mode
`required`, and a forced legacy `function_call` all mean: the reply IS a call.
vLLM 0.28 does not enforce any of them for this model's tool parser: measured
2026-09-17 against the backend directly, "Hello!" under `required` came back
with no call 11 of 11 times, thinking on or off, and a prompt
workaround 5 of 5. Chord returned a 502 to the caller.

The backend does honour a JSON schema (S05, certified). So the force is carried
as one: the reply is constrained to `{"name": <a declared function>,
"arguments": <that function's parameters>}`, and returned to the caller as the
tool call it is. Measured the same day through the gateway: 12 of 12, including
choosing the right function of two. The S03 check in chat_api.py still reads the
result; a reply that isn't a valid call is still a 502, never prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ulid import ULID

TOOL_PARAMS = ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call")
# The repair pass emits one small constrained JSON object. It must not inherit the sampling and
# budget the caller chose for PROSE: a caller who set max_completion_tokens 8 and forced a tool
# got a truncated object and the 502 this pass exists to prevent, and temperature 1.9 is nobody's
# choice for a schema emission (bug bounty 2026-09-17). Dropped, not overridden, so the
# upstream default applies where we have no opinion; `stop` goes too, since a stop string can cut
# the object mid-emission.
PROSE_PARAMS = ("temperature", "top_p", "seed", "max_tokens", "max_completion_tokens", "stop",
                "frequency_penalty", "presence_penalty", "logit_bias", "logprobs", "top_logprobs",
                "stream", "stream_options", "verbosity", "prediction", "n")
# No budget of our own either: a cap generous enough today truncates the one tool whose argument
# is long, which is C3 again at a rarer threshold. The schema bounds the SHAPE; the backend's own
# limit bounds the length.
LINE = ("This turn must be answered by calling exactly one of the functions below, even if the user only made small talk:"
        " pick the closest one and fill its arguments with sensible values. Reply with one JSON object"
        ' {"name": ..., "arguments": {...}} and nothing else.\nFunctions:\n')


@dataclass(frozen=True)
class Plan:
    functions: tuple[tuple[str, str, dict], ...]   # (name, description, parameters)
    legacy: bool                                    # forced function_call: the reply is message.function_call


def _open(parameters) -> dict:
    """A function that declares no parameters accepts NO arguments. Widening a missing
    `parameters` to a bare {"type": "object"} let the constrained pass emit any field, and a
    strict client got arguments its tool never declared (bug bounty 2026-09-17)."""
    if isinstance(parameters, dict) and parameters:
        return parameters
    return {"type": "object", "properties": {}, "additionalProperties": False}


def plan(params: dict) -> Plan | None:
    """The functions this turn must call one of, or None when nothing is forced
    (or the force names something a schema can't carry: a custom tool)."""
    legacy_choice = params.get("function_call")
    if isinstance(legacy_choice, dict) and isinstance(legacy_choice.get("name"), str):
        found = [f for f in params.get("functions") or [] if isinstance(f, dict) and f.get("name") == legacy_choice["name"]]
        return Plan(tuple((f["name"], f.get("description") or "", _open(f.get("parameters"))) for f in found[:1]), True) if found else None
    declared = [t["function"] for t in params.get("tools") or []
                if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict)]
    choice = params.get("tool_choice")
    if choice == "required":
        allowed = None
    elif isinstance(choice, dict) and choice.get("type") == "function":
        allowed = {(choice.get("function") or {}).get("name")}
    elif isinstance(choice, dict) and choice.get("type") == "allowed_tools" and (choice.get("allowed_tools") or {}).get("mode") == "required":
        allowed = {(t.get("function") or {}).get("name") for t in choice["allowed_tools"].get("tools") or [] if isinstance(t, dict)}
    else:
        return None
    chosen = [f for f in declared if isinstance(f.get("name"), str) and (allowed is None or f["name"] in allowed)]
    if not chosen:
        return None
    return Plan(tuple((f["name"], f.get("description") or "", _open(f.get("parameters"))) for f in chosen), False)


_DEFS = ("$defs", "definitions")
_DATA = frozenset({"const", "enum", "default", "examples"})   # instance values, never schemas
_NAMED = frozenset({*_DEFS, "properties", "patternProperties", "dependentSchemas"})   # name -> schema maps


def _relocate(parameters: dict, at: str, prefix: str, hoisted: dict) -> dict:
    """The client's parameters, moved from its own root to `at` inside the repair schema.

    Nesting it under properties.arguments left every local `$ref` pointing at a root that
    was no longer its own: pydantic's `"$ref": "#/$defs/Address"` dangled (review
    2026-09-24 B16). Its `$defs`/`definitions` are hoisted to the new root under `prefix`
    (so two functions' same-named definitions stay apart), refs into them are renamed to
    match, and any other local pointer (`"#"`, `"#/properties/x"`) is re-based on `at`.
    A plain-name anchor (`"#foo"`) is location-free and left alone. The caller's dict is
    never mutated."""
    whole: set[str] = set()   # definition maps some ref names as a whole

    def ref(value: str) -> str:
        for key in _DEFS:
            if value == f"#/{key}":
                # The map itself (Copilot on #333): it stays where the client put it, as
                # well as being hoisted, since the hoisted copy's names carry `prefix`.
                whole.add(key)
                return at + value[1:]
            if value.startswith(f"#/{key}/"):
                # A definition name is one JSON Pointer token: decode it, prefix the
                # NAME, and re-escape, so the ref and the hoisted key agree whatever
                # the name holds ("/" is "~1", "~" is "~0"; Copilot on #333).
                token, _, rest = value[len(key) + 3:].partition("/")
                name = prefix + token.replace("~1", "/").replace("~0", "~")
                return f"#/{key}/" + name.replace("~", "~0").replace("/", "~1") + (f"/{rest}" if rest else "")
        return at + value[1:]

    def walk(node: Any) -> Any:
        """A schema: keywords, where const/enum/default/examples hold instance data."""
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        return {k: ref(v) if k == "$ref" and isinstance(v, str) and (v == "#" or v.startswith("#/"))
                else v if k in _DATA
                else {name: walk(sub) for name, sub in v.items()} if k in _NAMED and isinstance(v, dict)
                else walk(v) for k, v in node.items()}

    moved: dict = walk(parameters)
    for key in _DEFS:
        defs = moved.get(key) if key in whole else moved.pop(key, None)
        if isinstance(defs, dict):
            hoisted.setdefault(key, {}).update({prefix + name: d for name, d in defs.items()})
    return moved


def apply(body: dict, p: Plan, thinking_mode: str = "passthrough") -> dict:
    """The upstream body for a forced turn: no tools, a schema instead, and, for a
    backend that has a thinking switch, thinking off (the answer is a small JSON
    object; reasoning tokens only delay it). Under `passthrough` no backend field is
    invented; the caller's own `chat_template_kwargs` is left as sent."""
    one = len(p.functions) == 1
    hoisted: dict = {}
    alts = [{"type": "object", "properties": {"name": {"const": n}, "arguments": _relocate(
                params, "#/properties/arguments" if one else f"#/anyOf/{i}/properties/arguments",
                "" if one else f"fn{i}_", hoisted)},
             "required": ["name", "arguments"], "additionalProperties": False}
            for i, (n, _, params) in enumerate(p.functions)]
    listing = "\n".join(f"- {n}: {d} parameters {json.dumps(params)}".replace(":  parameters", ": parameters") for n, d, params in p.functions)
    messages = list(body.get("messages") or [])
    line = LINE + listing
    if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
        messages[0] = {**messages[0], "content": messages[0]["content"].rstrip() + "\n\n" + line}
    else:
        messages.insert(0, {"role": "system", "content": line})
    out = {k: v for k, v in body.items() if k not in TOOL_PARAMS and k not in PROSE_PARAMS and k != "response_format"}
    out["messages"] = messages
    out["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "tool_call", "strict": True, "schema": {**(alts[0] if one else {"anyOf": alts}), **hoisted}}}
    if thinking_mode == "qwen_chat_template":
        out["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), "enable_thinking": False}
    return out


def _conform(args: dict, parameters: dict, dropped: list) -> dict:
    """Enforce the caller's OWN schema on the repaired arguments, and only that.

    Asking the backend for `additionalProperties: false` is not the same as enforcing it: the
    repaired reply carried a field the tool never declared (bug bounty 2026-09-17).

    This is JSON Schema, not a preference, and the asymmetry is deliberate:
      - a permissive schema (no `additionalProperties: false`) PASSES EXTRAS THROUGH, because
        extra properties are legal unless the caller forbids them;
      - a schema that sets `additionalProperties: false` said "I do not accept extra fields", so
        handing it exactly its declared fields is honouring that, not hiding anything.
    Do not "fix" this into a blanket strip: it would break every caller whose schema legally allows
    extras. Dropped keys are returned so the turn records them, never silently."""
    # `additionalProperties: false` with no `properties` key means no property is
    # declared, so every key is additional. Requiring `properties` to already be a
    # dict skipped that schema and passed the extra keys through unrecorded.
    if parameters.get("additionalProperties") is False:
        props = parameters.get("properties")
        if not isinstance(props, dict):
            props = {}
        dropped.extend(sorted(k for k in args if k not in props))
        return {k: v for k, v in args.items() if k in props}
    return args


def to_message(content, p: Plan, dropped: list | None = None) -> tuple[dict, str] | None:
    """(assistant message, finish_reason) for a constrained reply, or None when it
    isn't one of the planned calls (the S03 check then refuses the turn)."""
    try:
        obj = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        return None
    names = {n for n, _, _ in p.functions}
    if not (isinstance(obj, dict) and obj.get("name") in names and isinstance(obj.get("arguments"), dict)):
        return None
    parameters = next(params for n, _, params in p.functions if n == obj["name"])
    args = _conform(obj["arguments"], parameters, dropped if dropped is not None else [])
    fn = {"name": obj["name"], "arguments": json.dumps(args)}
    if p.legacy:
        return {"role": "assistant", "content": None, "function_call": fn}, "function_call"
    call = {"id": f"call_{str(ULID()).lower()}", "type": "function", "function": fn}
    return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
