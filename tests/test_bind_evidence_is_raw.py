"""THE EVIDENCE BOUNDARY: `_conform()` must never supply the proof that licensed it.

Option D's fast path binds an undeclared call to the sole declared tool when the model's
arguments already validate against it. The whole design rests on one question -- *did the
model mean this function* -- and that question has exactly one instrument: whether what it
actually emitted fits the only tool on offer.

`_conform()` enforces the caller's schema by DROPPING keys the schema forbids. So it can
turn arguments that do not satisfy the schema into arguments that do. If the bind decision
reads conformed arguments, then membership, schema-validity and repair-success are all
outputs of the repair, every shape assertion passes trivially, and they pass IDENTICALLY
when the model meant something else entirely. Nothing in the test set could contradict it.

    `_conform()` may execute after the decision; none of its manufactured outputs may
    serve as evidence for the decision that invoked it.   -- 2026-09-19

This file exists BEFORE the implementation, deliberately. The first test proves the hazard
is real against the shipped `_conform` -- otherwise the guard below has no subject and is
decoration. Written per the canonical amendment at
`qa/redteam/undeclared-tool-name-normalization-amendment-2026-09-19.md` (dfa85af).

2026-09-19.
"""
import json

import jsonschema
import pytest

from chord import forced_call
from chord import normalize as nz


# A schema that FORBIDS extras. `_conform` drops the offending key here, which is correct
# behaviour for conformance and disqualifying as evidence -- the two are the same act.
STRICT = {"type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"],
          "additionalProperties": False}

# The same schema without the prohibition. `_conform` passes extras through, because extra
# properties are legal unless the caller forbids them.
#: Not valid JSON Schema: `required` must be an array. Compiling it must RAISE.
BROKEN = {"type": "object", "properties": {"city": {"type": "string"}}, "required": "city"}

PERMISSIVE = {"type": "object",
              "properties": {"city": {"type": "string"}},
              "required": ["city"]}


def _valid(args, schema) -> bool:
    try:
        jsonschema.validate(args, schema)
        return True
    except jsonschema.ValidationError:
        return False


def test_conform_manufactures_validity_so_it_cannot_license_a_bind():
    """THE HAZARD, proven against the shipped function, not asserted about it.

    If this ever goes green in the first arm -- raw arguments already valid -- the example
    has stopped reproducing the manufacturing step and every guard built on it is vacuous."""
    raw = {"city": "Paris", "units": "celsius"}

    assert not _valid(raw, STRICT), \
        "precondition: the raw arguments must FAIL the sole tool's schema, or there is " \
        "nothing for _conform to manufacture and this file has no subject"

    dropped: list = []
    conformed = forced_call._conform(raw, STRICT, dropped)

    assert _valid(conformed, STRICT), "_conform did not produce schema-valid arguments"
    assert dropped == ["units"], f"the drop must be recorded, got {dropped}"

    # The two answers disagree. That disagreement is the entire reason the decision has to
    # be taken on the left-hand side.
    assert _valid(raw, STRICT) is not _valid(conformed, STRICT), \
        "raw-invalid and conformed-valid must differ, or the boundary is unobservable"


def test_a_dropped_key_is_evidence_against_the_bind_not_noise():
    """A non-empty `dropped` list means the model emitted something the sole tool does not
    accept. That is a REASON TO DOUBT the target, recorded at the moment it is discovered.

    The amendment is explicit: a dropped key is evidence against using conformance as the
    binding justification, even when the final call is shape-valid."""
    dropped: list = []
    forced_call._conform({"city": "Paris", "units": "celsius"}, STRICT, dropped)
    assert dropped, "a dropped key must be observable to the caller, never silent"

    clean: list = []
    forced_call._conform({"city": "Paris"}, STRICT, clean)
    assert clean == [], "arguments the schema accepts whole must drop nothing"


def test_the_permissive_schema_hides_the_same_hazard_behind_a_pass():
    """The asymmetry that makes this easy to get wrong.

    Under a permissive schema `_conform` is a no-op, so raw and conformed agree and the
    boundary is invisible. A test suite built only on permissive schemas would show the
    guard working while never exercising it -- the configuration a suite omits is green."""
    raw = {"city": "Paris", "units": "celsius"}
    dropped: list = []
    conformed = forced_call._conform(raw, PERMISSIVE, dropped)

    assert conformed == raw and dropped == [], "permissive schemas must pass extras through"
    assert _valid(raw, PERMISSIVE) and _valid(conformed, PERMISSIVE), \
        "both valid here -- which is precisely why STRICT is the fixture that can fail"


# ------------------------------------------------------ the bind, now that it exists

def _obs(name, arguments, shape=nz.MODERN, declared=None):
    return nz.observe([(shape, name, arguments)], declared or {"get_weather": STRICT})


def test_the_bind_reads_raw_arguments_not_conformed_ones():
    """THE ACCEPTANCE CRITERION, and the reason this file was written before the feature.

    `{"city","units"}` fails the sole tool's schema and `_conform` would make it pass. The
    bind must read the left-hand answer, so an undeclared call whose RAW arguments the
    declared tool would reject gets no licence."""
    raw = json.dumps({"city": "Paris", "units": "celsius"})
    assert not _valid(json.loads(raw), STRICT), "precondition: raw must fail the schema"

    assert nz.sole_tool_bind(_obs("weather", raw), {"get_weather": STRICT}) is None, \
        "the bind was licensed by arguments the sole declared tool would have rejected"

    # ...and the conformed version WOULD have licensed it, which is the whole hazard.
    conformed = forced_call._conform(json.loads(raw), STRICT, [])
    assert _valid(conformed, STRICT)
    assert nz.sole_tool_bind(_obs("weather", json.dumps(conformed)),
                             {"get_weather": STRICT}) == "get_weather", \
        "control: with valid raw arguments the bind DOES fire, so the test above is not " \
        "passing because the bind never fires at all"


def test_the_bind_refuses_when_more_than_one_tool_is_declared():
    """With a choice, picking one is guessing. The amendment routes that to a bounded
    constrained retry -- never a fuzzy name match, because the two undeclared names we
    have observed differ in KIND and a repair catching one misroutes the other."""
    two = {"get_weather": STRICT, "book_table": PERMISSIVE}
    assert nz.sole_tool_bind(_obs("weather", '{"city": "Paris"}', declared=two), two) is None


def test_the_bind_refuses_a_batch():
    """Whole-batch atomicity: a sibling cannot be bound while another is unresolved,
    because binding it means releasing it."""
    obs = nz.observe([(nz.MODERN, "weather", '{"city": "Paris"}'),
                      (nz.MODERN, "weather_getCurrent_2604", '{"city": "Tokyo"}')],
                     {"get_weather": STRICT})
    assert nz.sole_tool_bind(obs, {"get_weather": STRICT}) is None


@pytest.mark.parametrize("name, arguments, why", [
    (None, '{"city": "Paris"}', "malformed name: no question to ask"),
    ("weather", 7, "malformed arguments: nothing to weigh"),
    ("weather", "{not json", "unparseable arguments are no evidence, not an error"),
    ("weather", '"a string"', "valid JSON that is not an object"),
    ("get_weather", '{"city": "Paris"}', "already a match: this is the repair path"),
])
def test_the_bind_refuses_without_evidence(name, arguments, why):
    assert nz.sole_tool_bind(_obs(name, arguments), {"get_weather": STRICT}) is None, why


def test_a_schema_that_accepts_every_object_cannot_license_a_bind():
    """`{"type":"object","properties":{}}` is not empty, and it still validates
    every object. That is the same vacant licence as a missing schema."""
    declared = {"get_time": {"type": "object", "properties": {}}}
    obs = nz.observe([(nz.MODERN, "getCurrentTime_99", '{"tz": "JST"}')], declared)
    assert nz.sole_tool_bind(obs, declared) is None


def test_a_missing_schema_cannot_license_a_bind():
    """`{}` validates every object. A tool that declared no parameters must not
    have an invented call renamed onto it."""
    declared = {"get_time": {}}
    obs = nz.observe([(nz.MODERN, "getCurrentTime_99", '{"tz": "JST"}')], declared)
    assert nz.sole_tool_bind(obs, declared) is None
    closed = {"get_time": {"type": "object", "properties": {}, "additionalProperties": False}}
    assert nz.sole_tool_bind(nz.observe([(nz.MODERN, "getCurrentTime_99", '{"tz": "JST"}')], closed), closed) is None


def test_additional_properties_false_with_no_properties_drops_every_key():
    dropped: list = []
    schema = {"type": "object", "additionalProperties": False}
    assert forced_call._conform({"sneaky": 1}, schema, dropped) == {}
    assert dropped == ["sneaky"]


def test_a_dangling_ref_is_refused_at_the_door():
    schema = {"type": "object", "properties": {"a": {"$ref": "http://127.0.0.1:9/nope.json"}}}
    with pytest.raises(nz.SchemaUnusable):
        nz.check_declared_schema(schema)
    local = {"type": "object", "$defs": {"x": {"type": "string"}}, "properties": {"a": {"$ref": "#/$defs/x"}}}
    nz.check_declared_schema(local)


def test_an_uncompilable_caller_schema_raises_rather_than_licensing():
    """A schema we cannot compile has not said the arguments are wrong.

    Treating "could not ask" as "no" would be the safe-looking direction and still wrong:
    the door answers this with a 400 naming the tool, so a bad client schema surfaces as
    a bad client schema rather than a confusing repair-path failure."""
    with pytest.raises(nz.SchemaUnusable):
        nz.sole_tool_bind(_obs("weather", '{"city": "Paris"}', declared={"x": BROKEN}),
                          {"x": BROKEN})


@pytest.fixture
def answering_server():
    """An HTTP server that WOULD answer, and a log of what it was asked for.

    The previous version of this test pointed at port 1 and asserted the call raised.
    Port 1 refuses the connection, so the raise proved a fetch had FAILED -- not that no
    fetch was attempted. A false green on a security condition, and measurably wrong:
    against a server that answers, `jsonschema` retrieved the schema, validation returned
    True, and the library printed its own "Automatically retrieving remote references can
    be a security vulnerability" warning (2026-09-19).

    Absence of a fetch can only be shown by something that would have recorded one."""
    import http.server
    import threading

    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = json.dumps({"type": "string"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/schema", hits
    finally:
        server.shutdown()
        server.server_close()


def test_a_remote_ref_is_refused_and_never_fetched(answering_server):
    """Client-supplied schemas make remote `$ref` resolution an SSRF vector reachable by
    anyone who can declare a tool. The registry enforces that requirement."""
    url, hits = answering_server
    schema = {"type": "object", "properties": {"city": {"$ref": url}},
              "required": ["city"], "additionalProperties": False}

    with pytest.raises(nz.SchemaUnusable):
        nz.raw_arguments_validate('{"city": "Paris"}', schema)

    assert hits == [], f"the schema was FETCHED: {hits}. A refusal after a request is not " \
                       f"a refusal to request."


def test_the_control_proves_the_server_would_have_answered(answering_server):
    """Without this, the cell above passes against a server that was never listening."""
    import urllib.request

    url, hits = answering_server
    with urllib.request.urlopen(url, timeout=5) as r:
        assert json.loads(r.read()) == {"type": "string"}
    assert hits == ["/schema"], "the fixture did not record a real request"
