"""The observation layer records facts; verdicts are derived. Tests for both halves.

Three properties this file exists to hold, each of which has already cost us once:

1. `tool_parameter_state` is THREE states. `bool(declared)` was F1 -- it made "nothing
   declared" mean "skip validation" when nothing is a member of an empty set.
2. `batch_membership` is DERIVED from per-call results, never written independently, so a
   mixed batch stays visible. A scalar opinion cannot express one authorised sibling
   beside one unauthorised one, which is the exact defect batch atomicity prevents.
3. Contradictory rows are REFUSED at the writer, not filtered at read time. The amendment
   requires tests that deliberately attempt impossible states.

2026-09-19.
"""
import pytest

from chord import normalize as n

WEATHER = {"type": "object", "properties": {"city": {"type": "string"}},
           "required": ["city"], "additionalProperties": False}


def _body(**kw):
    return {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}], **kw}


def _tool(name="get_weather", params=None):
    return {"type": "function", "function": {"name": name, "parameters": params or WEATHER}}


# ---------------------------------------------------------------- declaration state

@pytest.mark.parametrize("body, expected", [
    (_body(), n.ABSENT),
    (_body(tool_choice="auto"), n.ABSENT),                 # constrains a call, declares nothing
    (_body(parallel_tool_calls=False), n.ABSENT),
    (_body(tools=[]), n.EMPTY),
    (_body(functions=[]), n.EMPTY),
    (_body(tools=["not-a-dict"]), n.EMPTY),                # present, yields no names
    (_body(tools=[_tool()]), n.NONEMPTY),
    (_body(functions=[{"name": "get_weather", "parameters": WEATHER}]), n.NONEMPTY),
])
def test_tool_parameter_state_is_three_states(body, expected):
    assert n.tool_parameter_state(body) == expected


def test_tool_choice_alone_declares_nothing():
    """The axis separation, asserted rather than assumed.

    A request carrying `tool_choice` and no `tools` has declared no function, so anything
    returned against it is unauthorised. Folding constraint parameters into the declaration
    axis would make that request look like it had declarations."""
    body = _body(tool_choice="required", parallel_tool_calls=True)
    assert n.tool_parameter_state(body) == n.ABSENT
    assert n.declared_functions(body) == {}


def test_declared_functions_carries_parameters_not_just_names():
    """Membership answers WHICH tool; the bind then needs that tool's own schema. A set of
    names cannot supply the second, and fetching it separately is how the two drift."""
    got = n.declared_functions(_body(tools=[_tool()]))
    assert got == {"get_weather": WEATHER}


def test_declared_functions_ignores_shapes_it_has_not_type_checked():
    body = _body(tools=[{"type": "function"}, {"function": {"name": 7}}, "x", _tool()],
                 functions=[{"no_name": 1}, {"name": "book_table"}])
    assert set(n.declared_functions(body)) == {"get_weather", "book_table"}
    assert n.declared_functions(body)["book_table"] == {}, "a missing schema is {}, never None"


# ---------------------------------------------------------------- per-call observation

def test_membership_against_an_empty_declared_set_is_no_match_for_every_name():
    """Nothing is a member of nothing -- including a name that WOULD have matched.

    This is the cell the matrix filter removed twice, on the reasoning that a valid-looking
    name with nothing declared is nonsense. It is the most real input there is."""
    obs = n.observe([(n.MODERN, "get_weather", "{}")], declared={})
    assert obs.calls[0].membership == n.NO_MATCH
    assert obs.calls[0].function_name == "get_weather", "the spelling is kept for grouping"
    assert obs.batch_membership == n.NONE_MATCH


def test_an_unparseable_name_is_not_evaluable_never_no_match():
    """The relation could not be asked. Recording `no_match` would be a verdict on a
    question nobody could pose, and it would be indistinguishable from a real refusal."""
    obs = n.observe([(n.MODERN, None, "{}")], declared={"get_weather": WEATHER})
    call = obs.calls[0]
    assert call.membership == n.NOT_EVALUABLE
    assert call.syntax == n.MALFORMED
    assert call.function_name is None
    assert obs.batch_membership == n.NOT_EVALUABLE


def test_syntax_and_membership_are_independent_facts():
    """A well-formed call can name nothing; a malformed one forecloses the question. Two
    axes, because collapsing them loses which of the two went wrong."""
    declared = {"get_weather": WEATHER}
    wellformed_nomatch = n.observe([(n.MODERN, "weather", "{}")], declared).calls[0]
    assert (wellformed_nomatch.syntax, wellformed_nomatch.membership) == (n.WELL_FORMED, n.NO_MATCH)

    malformed = n.observe([(n.MODERN, "get_weather", 7)], declared).calls[0]
    assert (malformed.syntax, malformed.membership) == (n.MALFORMED, n.MATCH)


# ---------------------------------------------------------------- the batch

def test_a_mixed_batch_stays_visible():
    """The case whole-batch atomicity exists for. If `batch_membership` collapsed to a
    scalar opinion, the authorised sibling would hide the unauthorised one and the
    instrument would be blind to the exact defect D is designed to prevent."""
    obs = n.observe([(n.MODERN, "get_weather", "{}"),
                     (n.MODERN, "weather_getCurrent_2604", "{}")],
                    declared={"get_weather": WEATHER})
    assert [c.membership for c in obs.calls] == [n.MATCH, n.NO_MATCH]
    assert obs.batch_membership == n.SOME_NO_MATCH
    assert [c.index for c in obs.calls] == [0, 1], "per-sibling identity must survive"


def test_batch_membership_is_derived_and_cannot_be_written():
    """It is a property, not a field. A derived value that can also be assigned is a value
    that will eventually disagree with what it derives from."""
    obs = n.observe([(n.MODERN, "get_weather", "{}")], declared={"get_weather": WEATHER})
    assert obs.batch_membership == n.ALL_MATCH
    with pytest.raises(AttributeError):
        obs.batch_membership = n.NONE_MATCH


def test_mixed_shapes_are_representable():
    """A response carrying both `tool_calls` and `function_call` is a real observable shape,
    not an exclusion -- and one the request never asked for."""
    obs = n.observe([(n.MODERN, "get_weather", "{}"), (n.LEGACY, "get_weather", "{}")],
                    declared={"get_weather": WEATHER})
    assert obs.call_shape == n.MIXED


def test_absence_is_one_fact():
    obs = n.observe([], declared={"get_weather": WEATHER})
    assert obs.present is False
    assert obs.call_shape is None
    assert obs.batch_membership is None
    assert obs.calls == ()


# ---------------------------------------------------------------- writer-side refusal

def test_the_writer_refuses_contradictory_states():
    """Deliberately attempted impossible rows, per the amendment: reject before persistence.

    Filtering these at read time is too late -- the bad rows are already written, and the
    first analysis has to guess which field to trust."""
    call = n.CallObservation(0, n.MODERN, n.WELL_FORMED, "get_weather", n.MATCH)

    with pytest.raises(ValueError, match="no call_shape"):
        n.BatchObservation(call_shape=None, calls=(call,))

    with pytest.raises(ValueError, match="absence is one fact"):
        n.BatchObservation(call_shape=n.MODERN, calls=())

    with pytest.raises(ValueError, match="duplicate call index"):
        n.BatchObservation(call_shape=n.MODERN, calls=(call, call))


def test_no_verdict_vocabulary_leaks_into_the_observation_layer():
    """Tripwire, not proof: fixture-only and verdict-shaped labels must not become telemetry
    VALUES. `match_shaped` is meaningful in the harness because the fixture knows which
    hypothetical declaration it chose a spelling from; production has no such reference.

    Scans string literals that are used as values, via the AST -- never raw source text.
    The first version of this grepped the source and tripped on the module docstring, which
    NAMES the banned words in order to forbid them. A text scan cannot tell a value from
    the prose explaining it, which is the same reason `git diff | grep -v '^[+-]#'` cannot
    certify a change as comment-only: docstrings are code and prose is not."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(n))
    docstrings = {id(ast.get_docstring(node, clean=False))
                  for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))}
    values = {node.value for node in ast.walk(tree)
              if isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node.value) not in docstrings}

    banned = {"match_shaped", "declared_match", "valid", "invalid"}
    leaked = sorted(values & banned)
    assert not leaked, (
        f"{leaked} used as a value. These are verdicts or fixture labels, not observations. "
        f"Record what came back; derive validity from syntax x membership x declaration "
        f"state -- an axis named by its outcome cannot express the cell where the outcome "
        f"is the question.")
