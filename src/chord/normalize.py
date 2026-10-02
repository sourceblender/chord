"""Observation of a returned tool call, relative to what the request declared.

This module OBSERVES and does not decide. Every value here is a fact about the request
or the response; validity is a later verdict derived from them. That split is the whole
point, and it is the one the matrix had to learn twice: an axis named by its outcome
cannot express the case where the outcome is the question.

So `declared_match` is not a value here, and neither is `valid`. Syntax is read off the
returned call. Membership is computed against the request's actual declared set. An
unparseable name yields `not_evaluable` -- never `no_match` -- because the relation could
not be asked, and recording "no match" for a question nobody could pose is a verdict
wearing an observation's clothes.

The vocabulary is the live trace schema: each production row maps onto a
matrix cell without translation (design dfa85af).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# MODULE SCOPE, deliberately. A deferred `import jsonschema` inside the function would
# hide this dependency from any import check that proves a production install carries
# what the package needs, because such checks only see module-scope imports.
import jsonschema
from jsonschema.exceptions import SchemaError
import referencing
from referencing.jsonschema import DRAFT202012

# Declaration lives in `tools` / `functions` ONLY. `tool_choice` and friends constrain a
# call, they do not declare a function -- a request carrying `tool_choice` with no `tools`
# has declared nothing, and anything returned against it is unauthorised. Keeping the two
# axes apart is what `chat_api._tools_error` already relies on.
DECLARING_PARAMS = ("tools", "functions")

ABSENT, EMPTY, NONEMPTY = "absent", "empty", "nonempty"
WELL_FORMED, MALFORMED = "well_formed", "malformed"
MATCH, NO_MATCH, NOT_EVALUABLE = "match", "no_match", "not_evaluable"
ALL_MATCH, SOME_NO_MATCH, NONE_MATCH = "all_match", "some_no_match", "none_match"
MODERN, LEGACY, MIXED = "modern", "legacy", "mixed"


def tool_parameter_state(body: dict) -> str:
    """`absent` | `empty` | `nonempty`, on the DECLARING parameters only.

    Three states, not a boolean. `bool(declared)` as a trigger is the F1 defect -- it made
    "nothing declared" mean "skip", when nothing can be a member of an empty set and a
    returned call is therefore the clearest violation available. Collapsing the axis in the
    INSTRUMENT is worse than in the guard: a guard with a bad trigger fails a test, an
    instrument that cannot see a distinction emits confident wrong numbers forever."""
    present = [body.get(k) for k in DECLARING_PARAMS if k in body]
    if not present:
        return ABSENT
    return NONEMPTY if declared_functions(body) else EMPTY


def declared_functions(body: dict) -> dict[str, dict]:
    """{name: parameters} the request actually declared. Reads nothing it hasn't type-checked.

    Mirrors `chat_api._declared_functions` on names and additionally carries each function's
    parameters, because membership answers *which* tool and the bind needs *its* schema."""
    out: dict[str, dict] = {}
    # Bound once, then narrowed: re-calling .get() inside the isinstance guard
    # leaves the value expression Unknown to the checker (batch E).
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name, parameters = fn.get("name"), fn.get("parameters")
        if isinstance(name, str):
            out[name] = parameters if isinstance(parameters, dict) else {}
    for fn in body.get("functions") or []:
        if not isinstance(fn, dict):
            continue
        name, parameters = fn.get("name"), fn.get("parameters")
        if isinstance(name, str):
            out[name] = parameters if isinstance(parameters, dict) else {}
    return out


@dataclass(frozen=True)
class CallObservation:
    index: int
    call_shape: str                  # MODERN | LEGACY
    syntax: str                      # WELL_FORMED | MALFORMED
    function_name: str | None        # kept for grouping; its spelling carries no verdict
    membership: str                  # MATCH | NO_MATCH | NOT_EVALUABLE
    raw_arguments: object = None     # AS EMITTED. Never conformed, never re-serialised:
                                     # the bind's only licence is what the model actually
                                     # sent, and a repaired copy cannot testify for it.


@dataclass(frozen=True)
class BatchObservation:
    """One tagged fact about call presence, so dependent fields cannot disagree.

    Three separately written `none` values are three chances to contradict each other, and
    a row saying "no call" beside "membership: match" is nonsense no query can repair.
    Presence is decided once, here, and everything else derives from `calls`."""
    call_shape: str | None = None            # None exactly when there is no call
    calls: tuple[CallObservation, ...] = field(default_factory=tuple)

    @property
    def present(self) -> bool:
        return bool(self.calls)

    @property
    def batch_membership(self) -> str | None:
        """Derived, never an independent opinion.

        A scalar `match`/`no_match` cannot describe one authorised sibling beside one
        unauthorised one -- and that mixed batch is precisely the case whole-batch atomicity
        exists to prevent, so a field that cannot express it blinds the instrument to the
        defect it was built for."""
        if not self.calls:
            return None
        results = {c.membership for c in self.calls}
        if NOT_EVALUABLE in results:
            return NOT_EVALUABLE
        if results == {MATCH}:
            return ALL_MATCH
        if results == {NO_MATCH}:
            return NONE_MATCH
        return SOME_NO_MATCH

    def __post_init__(self):
        # Writer-side contradiction rejection. The amendment requires impossible rows to be
        # refused BEFORE persistence, and requires tests that deliberately attempt them.
        if self.calls and self.call_shape is None:
            raise ValueError("calls present with no call_shape")
        if not self.calls and self.call_shape is not None:
            raise ValueError("call_shape set with no calls; absence is one fact, not three")
        if len({c.index for c in self.calls}) != len(self.calls):
            raise ValueError("duplicate call index; per-sibling identity must survive")


def observe(calls, declared: dict[str, dict]) -> BatchObservation:
    """Build the observation from assembled calls.

    `calls` is a sequence of (shape, name, arguments) as returned. A name that is not a
    string could not be compared to anything, so its membership is NOT_EVALUABLE and its
    syntax is MALFORMED -- two different facts about the same call, kept apart because a
    well-formed call can still name nothing, and a malformed one forecloses the question."""
    observed = []
    for i, (shape, name, arguments) in enumerate(calls):
        parseable = isinstance(name, str)
        syntax = WELL_FORMED if parseable and isinstance(arguments, (str, dict)) else MALFORMED
        if not parseable:
            membership = NOT_EVALUABLE
        else:
            # Against an absent or empty declared set every name is NO_MATCH by construction:
            # nothing is a member of nothing. The name is still recorded for grouping.
            membership = MATCH if name in declared else NO_MATCH
        observed.append(CallObservation(index=i, call_shape=shape, syntax=syntax,
                                        function_name=name if parseable else None,
                                        membership=membership, raw_arguments=arguments))
    if not observed:
        return BatchObservation()
    shapes = {c.call_shape for c in observed}
    return BatchObservation(call_shape=shapes.pop() if len(shapes) == 1 else MIXED,
                            calls=tuple(observed))


# ---------------------------------------------------------------- the repair decision

DROP, PREBYTE_RETRY, SOLE_TOOL_BIND = "drop", "prebyte_retry", "sole_tool_bind"
CONSTRAINED_RETRY, UNREPAIRED, NONE = "constrained_retry", "unrepaired", "none"


class SchemaUnusable(Exception):
    """The caller's own schema is not valid JSON Schema.

    Raised rather than swallowed: a schema we cannot compile has not said the arguments
    are wrong, and treating "could not ask" as "no" would license a bind on evidence that
    was never obtained. The door answers this with a 400 naming the tool -- a bad client
    schema must not surface as a confusing repair-path failure (the requirement)."""


def _refuse_to_retrieve(uri: str):
    """The retriever that never retrieves.

    `jsonschema` FETCHES remote `$ref`s by default -- measured, not assumed: a schema
    served from a local HTTP server was retrieved, the request appeared in the server's
    log, and validation returned True while the library emitted its own
    "Automatically retrieving remote references can be a security vulnerability" warning
    (2026-09-19).

    Tool schemas are CLIENT-SUPPLIED, so that is an SSRF vector reachable by anyone who
    can declare a tool. Handing the registry a retriever that raises is what actually
    stops it; the previous version documented offline behaviour it did not have."""
    raise SchemaUnusable(
        f"this tool's schema references {uri!r}, which chord does not fetch: remote "
        f"reference resolution is refused for client-supplied schemas")


#: Empty, with retrieval refused. Every validator is built against it, so no `$ref` can
#: reach the network no matter what a caller declares.
_OFFLINE = referencing.Registry(retrieve=_refuse_to_retrieve)


def check_declared_schema(parameters: dict) -> None:
    """Raise SchemaUnusable if this is not valid JSON Schema. The door's request-time
    check, so a client learns at 400 rather than through a repair that cannot proceed.

    `check_schema` does not resolve `$ref`. A dangling or remote reference is still a
    syntactically valid schema, and it used to reach the repair path, where the
    failure escaped as a 500 and the turn was never traced."""
    _validator(parameters)
    _require_resolvable_refs(parameters)


def _refs(node, found: list[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            found.append(ref)
        for value in node.values():
            _refs(value, found)
    elif isinstance(node, list):
        for value in node:
            _refs(value, found)


def _require_resolvable_refs(parameters: dict) -> None:
    found: list[str] = []
    _refs(parameters, found)
    if not found:
        return
    resolver = _OFFLINE.resolver_with_root(DRAFT202012.create_resource(parameters))
    for ref in found:
        try:
            resolver.lookup(ref)
        except Exception as exc:
            raise SchemaUnusable(
                f"this tool's schema references {ref!r}, which chord does not resolve: {exc}"
            ) from exc


def _validator(parameters: dict):
    """A Draft 2020-12 validator for one declared tool, pinned and genuinely offline.

    Pinned rather than sniffed from `$schema`: a client-supplied document would otherwise
    choose which dialect judges it."""
    try:
        jsonschema.Draft202012Validator.check_schema(parameters)
    except SchemaError as e:
        raise SchemaUnusable(str(e)) from e
    return jsonschema.Draft202012Validator(parameters, registry=_OFFLINE)


def raw_arguments_validate(raw, parameters: dict) -> bool:
    """Do the MODEL'S OWN arguments satisfy this tool's schema, before any repair?

    `raw` is what the backend emitted: a JSON string on the wire, or an already-parsed
    object. Unparseable is False -- not an error, just no evidence.

    THE EVIDENCE BOUNDARY. `_conform()` enforces a schema by DROPPING keys it forbids, so
    it turns arguments that fail into arguments that pass. If the bind read conformed
    arguments, membership, schema-validity and repair-success would all be outputs of the
    repair: every shape assertion true, and true identically when the model meant another
    function. Nothing in the test set could disagree.

        `_conform()` may execute after the decision; none of its manufactured outputs
        may serve as evidence for the decision that invoked it.   -- 2026-09-19
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return False
    if not isinstance(raw, dict):
        return False
    try:
        return _validator(parameters).is_valid(raw)
    except SchemaUnusable:
        raise
    except Exception as e:                       # noqa: BLE001
        # A `$ref` we will not resolve, most likely. `jsonschema` does not fetch remotely
        # by default -- verified: an unresolvable `http://` ref raises RESOLUTION rather
        # than attempting retrieval -- and client-supplied schemas make remote resolution
        # an SSRF vector, so we keep it that way. But the error escapes as an unexpected
        # failure unless caught here, and "we could not evaluate" must never quietly
        # become "invalid": refusing a bind on evidence nobody obtained reads as caution
        # and is the same mistake as licensing one on evidence nobody obtained.
        raise SchemaUnusable(
            f"this tool's schema could not be evaluated without resolving a reference "
            f"we do not fetch: {e}") from e


def sole_tool_bind(observation: BatchObservation, declared: dict[str, dict]) -> str | None:
    """The declared tool an unauthorised call may be bound to, or None.

    Returns a NAME, never a repaired call: the decision and the repair stay separate so
    the repair cannot supply its own licence. `_conform()` runs afterwards, on the
    caller's schema, and its dropped keys are recorded -- as accounting, not as evidence.

    Four conditions, and each one alone is enough to refuse:

      * exactly ONE function declared. With several, picking one is guessing, and the
        amendment sends that to a bounded constrained retry instead. No fuzzy names, no
        aliases, no truncation heuristics, no convention maps -- the two undeclared names
        we have observed differ in KIND, so a repair that catches `weather` silently
        misroutes `weather_getCurrent_2604`;
      * exactly ONE returned call. A batch cannot be bound sibling-by-sibling without
        releasing a valid sibling before discovering an invalid one;
      * the call is WELL FORMED. A malformed call has no arguments to weigh;
      * the RAW arguments validate against that tool's schema AND are not empty. The
        second half is not pedantry, and my own #289 regression suite found it: the test
        tools declare `{"type": "object", "properties": {}}`, which forbids nothing, so
        `{}` validated trivially and EVERY undeclared call bound. A schema that accepts
        anything makes "the arguments fit this tool" an `assert True` -- the licence is
        free and the evidence is vacuous.
        Empty arguments are consistent with every function in existence, so they indicate
        no particular one. The bind's whole question is *did the model mean THIS tool*,
        and only arguments that could have failed can answer it. Same shape as the
        permissive-schema hazard one level up: a check that cannot fail has not passed.

    A call that already matches needs no bind and returns None -- this is the repair path,
    not the happy path."""
    if len(declared) != 1 or len(observation.calls) != 1:
        return None
    call = observation.calls[0]
    if call.syntax != WELL_FORMED or call.membership == MATCH:
        return None
    (name, parameters), = declared.items()
    if not _non_empty_object(call.raw_arguments):
        return None
    # `{}` and `{"type":"object","properties":{}}` validate every object. A check
    # that cannot fail has not passed, so it cannot license a rename.
    if not _schema_can_reject_an_object(parameters):
        return None
    return name if raw_arguments_validate(call.raw_arguments, parameters) else None


def _schema_can_reject_an_object(parameters: dict) -> bool:
    """Whether some object is invalid against `parameters`.

    An empty schema and a schema with no required fields and no property
    constraints accept every object. Arguments that validate against those are
    not evidence the model meant this tool."""
    if not parameters:
        return False
    probes = (
        {"__chord_probe__": 1},
        {"city": 1},
        {"city": "x", "extra": True},
        {"a": [], "b": {}},
    )
    return any(not raw_arguments_validate(json.dumps(probe), parameters) for probe in probes)


def _non_empty_object(raw) -> bool:
    """Did the model actually say something about its arguments?

    `{}` against a tool that declares no parameters is a perfect fit and no evidence at
    all: it fits every tool equally. Only a call that COULD have been wrong about this
    function can be right about it."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return False
    return isinstance(raw, dict) and bool(raw)


# ---------------------------------------------------------------- stream deltas

CONTENT, CALL = "content", "call"


def delta_kinds(chunk) -> set[str]:
    """What a single streamed chunk carries: CONTENT, CALL, both, or neither.

    Used to decide the no-tools case WITHOUT buffering the stream. When the request
    declared no tool parameter, every returned call is invalid regardless of its
    spelling, so the decision needs no name and no arguments -- it is decidable on the
    FIRST call delta. That is what lets ordinary chat streams keep their latency: only
    a request that declares tools pays for whole-call quarantine.

    Reads nothing it has not type-checked. A chunk whose shape it cannot parse carries
    neither kind, because a guess about an unreadable chunk is worth less than nothing."""
    kinds: set[str] = set()
    if not isinstance(chunk, dict):
        return kinds
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return kinds
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("content"), str) and delta["content"]:
            kinds.add(CONTENT)
        # `tool_calls` or a legacy `function_call` -- presence is the whole signal here.
        # An empty list is not a call: a backend that opens the field without filling it
        # has not claimed anything yet.
        if delta.get("tool_calls") or delta.get("function_call"):
            kinds.add(CALL)
    return kinds


# ---------------------------------------------------------------- the canonical row

def observation_record(observation: BatchObservation) -> dict | str:
    """The `call_observation` field: `none`, or the tagged present-shape.

    ONE tagged fact about call presence, so dependent fields cannot contradict each
    other. Three separately written `none`s are three chances to disagree, and a row
    saying "no call" beside "membership: match" is nonsense no query can repair.

    Per-call, because a scalar cannot describe one authorised sibling beside one
    unauthorised one -- and that mixed batch is exactly the case whole-batch atomicity
    exists to prevent, so a field unable to express it blinds the instrument to the
    defect it was built for. `batch_membership` is DERIVED from the list, never written
    as an independent opinion."""
    if not observation.present:
        return NONE
    return {
        "call_shape": observation.call_shape,
        "calls": [{"index": c.index, "syntax": c.syntax,
                   "function_name": c.function_name, "membership": c.membership}
                  for c in observation.calls],
        "batch_membership": observation.batch_membership,
    }


def canonical_row(body: dict, observation: BatchObservation, *, transport: str,
                  sse_data_emitted: bool, content_delta_emitted: bool,
                  disposition: str) -> dict:
    """Every field the amendment requires, for EVERY response that traverses the boundary.

    Including the clean ones. Recording only repairs and refusals produces numerators with
    no recoverable base population -- which is precisely the hole that made `2/222`
    unable to price a neighbouring branch, and it cannot be fixed retroactively because
    the clean rows were never written."""
    return {
        "transport": transport,
        "tool_parameter_state": tool_parameter_state(body),
        "call_observation": observation_record(observation),
        "client_sse_data_emitted": sse_data_emitted,
        "client_content_delta_emitted": content_delta_emitted,
        "normalization_disposition": disposition,
    }
