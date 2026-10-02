"""S13g (S-cf-038): `n` above 1 was refused. It is n independent single-choice turns, run together
and returned as one completion, so every guard a single reply gets applies to each choice."""
import itertools
import json

import pytest

from test_responses import MODEL, make, strict
from test_skeleton import FakeUpstream

ASK = [{"role": "user", "content": "Name one planet."}]


class Varied(FakeUpstream):
    """A different answer per call, as sampling gives."""
    def __init__(self):
        super().__init__(); self.turn = itertools.count()

    async def complete(self, body):
        data, dep = await super().complete(body)
        data["choices"][0]["message"]["content"] = ["Mars", "Venus", "Earth", "Saturn"][next(self.turn) % 4]
        data["usage"] = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
        return data, dep


def test_n_choices_come_back_indexed_in_one_strict_valid_completion(tmp_path):
    up = Varied()
    deps, client, sdk = make(tmp_path, up)
    c = sdk.chat.completions.create(model=MODEL, messages=ASK, n=3, temperature=1)
    assert [ch.index for ch in c.choices] == [0, 1, 2]
    assert sorted(ch.message.content for ch in c.choices) == ["Earth", "Mars", "Venus"]
    raw = client.post("/v1/chat/completions", json={"model": MODEL, "messages": ASK, "n": 2, "temperature": 1}).json()
    strict(raw, "chat")
    assert raw["usage"] == {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}   # one prompt, counted once
    assert all("n" not in b for b in up.bodies)                                               # each turn is a single-choice turn


def test_n_turns_are_never_routed_to_a_specialist(tmp_path, monkeypatch):
    from chord import specialists
    ran = []

    async def image(job, ctx):
        ran.append(job)
        raise AssertionError("n above 1 must not render n pictures")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)

    class Router:
        async def ainvoke(self, msgs):
            class R: content = '{"route": "image", "intent": "a mug"}'
            return R()

    deps, client, _ = make(tmp_path, Varied(), router=Router, router_enabled=True, experimental_routes=frozenset({"image"}))
    r = client.post("/v1/chat/completions", json={"model": MODEL, "messages": [{"role": "user", "content": "make me a mug picture"}],
                                                  "n": 2, "temperature": 1})
    assert r.status_code == 200 and ran == [] and len(r.json()["choices"]) == 2


@pytest.mark.parametrize("extra, why", [
    ({"temperature": 0}, "temperature must be above 0"),
    ({"web_search_options": {}}, "with web_search_options"),
    ({"modalities": ["text", "audio"], "audio": {"voice": "alloy", "format": "wav"}}, "audio output"),
])
def test_what_n_cannot_do_yet_is_refused_by_name(tmp_path, extra, why):
    _, client, _ = make(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": MODEL, "messages": ASK, "n": 2, **extra})
    assert r.status_code == 400 and r.json()["error"]["param"] == "n" and why in r.json()["error"]["message"], r.text
    strict(r.json(), "error")


def test_an_upstream_failure_in_any_choice_is_the_error_not_a_partial_completion(tmp_path):
    from chord.upstream import UpstreamError

    class SecondFails(Varied):
        async def complete(self, body):
            if next(self.turn) == 1:
                raise UpstreamError(500, '{"error": {"message": "boom"}}')
            return await FakeUpstream.complete(self, body)

    _, client, _ = make(tmp_path, SecondFails())
    r = client.post("/v1/chat/completions", json={"model": MODEL, "messages": ASK, "n": 3, "temperature": 1})
    assert r.status_code >= 500 and set(r.json()) == {"error"}


def test_store_keeps_the_merged_completion_once(tmp_path):
    deps, client, sdk = make(tmp_path, Varied())
    c = sdk.chat.completions.create(model=MODEL, messages=ASK, n=2, temperature=1, store=True, metadata={"k": "v"})
    listing = client.get("/v1/chat/completions").json()["data"]
    assert [x["id"] for x in listing] == [c.id] and len(listing[0]["choices"]) == 2 and listing[0]["metadata"] == {"k": "v"}


@pytest.mark.parametrize("include_usage", [False, True])
def test_n_streams_as_spec_shaped_chunks_one_delta_and_a_finish_per_choice(tmp_path, include_usage):
    from qa.conformance.schema import Spec, validate_payload
    deps, client, sdk = make(tmp_path, Varied())
    body = {"model": MODEL, "messages": ASK, "n": 3, "temperature": 1, "stream": True}
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    for ch in chunks:
        row = validate_payload(ch, kind="chat-stream", spec=Spec(), fields="strict")
        assert row["verdict"] == "pass", json.dumps(row["evidence"])[:600]
    text = {i: "".join(c["delta"].get("content") or "" for ch in chunks for c in ch["choices"] if c["index"] == i) for i in range(3)}
    assert sorted(text.values()) == ["Earth", "Mars", "Venus"]
    finishes = [(c["index"], c["finish_reason"]) for ch in chunks for c in ch["choices"] if c["finish_reason"]]
    assert finishes == [(0, "stop"), (1, "stop"), (2, "stop")]
    if include_usage:                                                    # S12's shape holds for n too
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 6
        assert all(ch["usage"] is None for ch in chunks[:-1])
    else:
        assert all("usage" not in ch for ch in chunks)
    got = list(sdk.chat.completions.create(model=MODEL, messages=ASK, n=2, temperature=1, stream=True))   # the official SDK reads it
    assert {c.index for ch in got for c in ch.choices} == {0, 1}


@pytest.mark.parametrize("options", ["yes", ["include_usage"], {"include_usage": "yes"}])
def test_a_malformed_stream_options_is_refused_before_any_turn_runs_or_is_stored(tmp_path, options):
    """Review 2026-09-24 A5: n>1 streamed read `(stream_options or {}).get(...)` after
    the n turns had run on the GPU and the merged completion had been stored, so a
    non-object stream_options answered 500 and left a stored completion behind."""
    up = Varied()
    deps, client, sdk = make(tmp_path, up)
    body = {"model": MODEL, "messages": ASK, "n": 2, "temperature": 1, "stream": True,
            "store": True, "stream_options": options}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "stream_options"
    assert up.bodies == []                                             # no turn ran
    assert client.get("/v1/chat/completions").json()["data"] == []    # nothing stored
