"""S01 (red team pass 1, 2026-09-15): a `developer` message, or a `system`
message anywhere but first, reached the upstream chat template as a second
system message and came back as a raw 400 ("System message must be at the
beginning"). Acceptance (#149): the 400 goes away AND instruction authority
and relative order survive the fold into the single upstream system slot."""
import json

import pytest

from chord.graph import INSTRUCTION_ROLES, layered
from test_skeleton import FakeUpstream, make

BASE = "BASE LAYER"


def roles(msgs):
    return [m["role"] for m in msgs]


def only_one_system_and_first(sent):
    assert sent[0]["role"] == "system"
    assert not [m for m in sent[1:] if m["role"] in INSTRUCTION_ROLES]


# The three red-team repro shapes, sent through the real endpoint.
S01_CASES = {
    "S-cf-051 developer first": [
        {"role": "developer", "content": "Answer only in UPPERCASE letters."},
        {"role": "user", "content": "say hello"}],
    "S-cf-063 second system mid-conversation": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "From now on answer in French."},
        {"role": "user", "content": "how are you"}],
    "S-cf-064 developer after turns": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "developer", "content": "Reply in one word."},
        {"role": "user", "content": "weather?"}],
}


@pytest.mark.parametrize("case", list(S01_CASES))
def test_s01_shapes_reach_upstream_with_one_leading_system(tmp_path, case):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    msgs = S01_CASES[case]
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": msgs})
    assert r.status_code == 200, r.text
    sent = up.bodies[0]["messages"]
    only_one_system_and_first(sent)
    # Every instruction survives, in the system slot, in the client's order.
    system = sent[0]["content"]
    texts = [m["content"] for m in msgs if m["role"] in INSTRUCTION_ROLES]
    positions = [system.index(t) for t in texts]
    assert positions == sorted(positions)
    # The conversation itself is unchanged and in order.
    assert sent[1:] == [m for m in msgs if m["role"] not in INSTRUCTION_ROLES]


def test_developer_first_keeps_its_role_as_a_label():
    out = layered(BASE, [{"role": "developer", "content": "D"}, {"role": "user", "content": "u"}])
    assert out == [{"role": "system", "content": "BASE LAYER\n\n[Developer instruction]\nD"},
                   {"role": "user", "content": "u"}]


def test_leading_unnamed_system_stays_unlabelled_byte_for_byte():
    out = layered(BASE, [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}])
    assert out[0] == {"role": "system", "content": "BASE LAYER\n\nS"}


def test_later_instructions_keep_order_and_are_labelled_with_their_position():
    msgs = [{"role": "system", "content": "S0"},
            {"role": "user", "content": "u1"},
            {"role": "developer", "content": "D1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "system", "content": "S2"}]
    out = layered(BASE, msgs)
    assert out[0]["content"] == ("BASE LAYER\n\nS0"
                                 "\n\n[Developer instruction given after message 1 of the conversation]\nD1"
                                 "\n\n[System instruction given after message 3 of the conversation]\nS2")
    assert out[1:] == [msgs[1], msgs[3], msgs[4]]


def test_later_instruction_wins_by_coming_after_the_leading_ones():
    """Precedence: a later instruction appears after every leading one, the same
    relative order the client sent, so a later override still reads as later."""
    out = layered(BASE, [{"role": "system", "content": "Answer in English."},
                         {"role": "user", "content": "hi"},
                         {"role": "system", "content": "Answer in French."}])
    c = out[0]["content"]
    assert c.index("Answer in English.") < c.index("Answer in French.")


def test_parts_content_is_kept_as_parts_with_labels():
    parts = [{"type": "text", "text": "P"}]
    out = layered(BASE, [{"role": "user", "content": "u"}, {"role": "developer", "content": parts}])
    content = out[0]["content"]
    assert isinstance(content, list) and content[-1] == parts[0]
    assert "[Developer instruction given after message 1 of the conversation]" in "".join(p["text"] for p in content)
    assert out[1:] == [{"role": "user", "content": "u"}]


def test_note_still_goes_last_after_later_instructions():
    out = layered(BASE, [{"role": "user", "content": "u"}, {"role": "system", "content": "late"}], note="NOTE")
    assert out[0]["content"].endswith("late\n\nNOTE")
    only_one_system_and_first(out)


def test_no_instruction_messages_is_unchanged():
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    assert layered(BASE, msgs) == [{"role": "system", "content": BASE}, *msgs]


@pytest.mark.parametrize("seq", ["sud", "dsu", "usu", "uds", "sdusdu", "ududsd", "d", "s"])
def test_any_mix_sends_exactly_one_system_first(seq):
    kinds = {"s": "system", "d": "developer", "u": "user"}
    msgs = [{"role": kinds[k], "content": f"{k}{i}"} for i, k in enumerate(seq)]
    out = layered(BASE, msgs)
    only_one_system_and_first(out)
    assert [m["content"] for m in out[1:]] == [m["content"] for m in msgs if m["role"] == "user"]


def test_role_swapped_inputs_no_longer_collapse():
    """the discriminating pair (#150): these used to normalize byte-identically."""
    a = layered(BASE, [{"role": "developer", "content": "A"}, {"role": "user", "content": "u"},
                       {"role": "system", "content": "B"}])
    b = layered(BASE, [{"role": "system", "content": "A"}, {"role": "user", "content": "u"},
                       {"role": "developer", "content": "B"}])
    assert a != b
    assert a[0]["content"] == ("BASE LAYER\n\n[Developer instruction]\nA"
                               "\n\n[System instruction given after message 1 of the conversation]\nB")
    assert b[0]["content"] == ("BASE LAYER\n\nA"
                               "\n\n[Developer instruction given after message 1 of the conversation]\nB")


@pytest.mark.parametrize("m,label", [
    ({"role": "system", "name": "ops", "content": "x"}, '[System instruction from "ops"]'),
    ({"role": "developer", "name": "harness", "content": "x"}, '[Developer instruction from "harness"]'),
])
def test_names_are_kept_on_leading_instructions(m, label):
    out = layered(BASE, [m, {"role": "user", "content": "u"}])
    assert out[0]["content"] == f"BASE LAYER\n\n{label}\nx"
    assert "name" not in out[0]          # never inherited onto the slot by accident


def test_names_are_kept_on_later_instructions():
    out = layered(BASE, [{"role": "user", "content": "u"}, {"role": "developer", "name": "ops", "content": "x"}])
    assert out[0]["content"].endswith('[Developer instruction from "ops" given after message 1 of the conversation]\nx')


def test_every_role_and_name_assignment_normalizes_distinctly():
    """Injective over who gave each instruction: vary role and name at every
    instruction position of one conversation; no two inputs may collapse."""
    import itertools
    shape = ["i", "u", "i", "u", "i"]
    variants = [("system", None), ("developer", None), ("system", "ops"), ("developer", "ops")]
    seen = {}
    for combo in itertools.product(variants, repeat=shape.count("i")):
        it = iter(combo)
        msgs = []
        for k, kind in enumerate(shape):
            if kind == "u":
                msgs.append({"role": "user", "content": f"u{k}"})
            else:
                role, name = next(it)
                msgs.append({"role": role, "content": f"i{k}", **({"name": name} if name else {})})
        out = json.dumps(layered(BASE, msgs))
        assert out not in seen, (combo, seen.get(out))
        seen[out] = combo
    assert len(seen) == len(variants) ** 3


@pytest.mark.parametrize("name", [123, None, ["ops"], {"n": 1}, True])
def test_a_non_string_name_is_refused_not_coerced_into_a_sender(tmp_path, name):
    """The repro on 9b19e3c: name 123 came back 200 and upstream saw
    [System instruction from "123"]. The schema types name as a string."""
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [
        {"role": "system", "name": name, "content": "x"}, {"role": "user", "content": "u"}]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_messages" and err["param"] == "messages[0].name"
    assert up.bodies == []                       # nothing reached the model


def test_a_string_name_still_passes_through_the_endpoint(tmp_path):
    up = FakeUpstream()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [
        {"role": "system", "name": "ops", "content": "x"}, {"role": "user", "content": "u"}]})
    assert r.status_code == 200
    assert up.bodies[0]["messages"][0]["content"].endswith('[System instruction from "ops"]\nx')
