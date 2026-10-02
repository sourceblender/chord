"""S14(b) (red team pass 1, 2026-09-15; S-cf-020/021): verbosity was forwarded
to a backend that ignores it, so low and high came back the same (344 vs 360
tokens). The service now honours it (the ruling): validated, never
forwarded, applied as a labelled service line in the one system message, and
traced. Prose length is too stochastic to be this gate; the boundary is."""
import pytest

from chord.graph import VERBOSITY_LINES
from test_progress import last_trace
from test_skeleton import FakeUpstream, make

ASK = [{"role": "system", "content": "You are Ava."}, {"role": "user", "content": "Explain what a CPU is."}]


def post(client, stream, **extra):
    return client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream, "messages": ASK, **extra})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("level", ["low", "high"])
def test_low_and_high_become_a_service_line_and_are_never_forwarded(tmp_path, stream, level):
    up = FakeUpstream()
    deps, client = make(tmp_path, up)
    assert post(client, stream, verbosity=level).status_code == 200
    sent = up.bodies[-1]
    assert "verbosity" not in sent                                        # consumed, not forwarded
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]    # one system slot, order kept
    system = sent["messages"][0]["content"]
    assert system.endswith("You are Ava.\n\n" + VERBOSITY_LINES[level])  # after the harness's text, last
    assert system.count("[Service instruction") == 1
    assert sent["messages"][1] == ASK[1]                                  # never user text
    t = last_trace(deps.settings)
    assert (t["verbosity_requested"], t["verbosity_effective"], t["verbosity_handled_by"]) == (level, level, "service")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("extra", [{}, {"verbosity": None}, {"verbosity": "medium"}], ids=["omitted", "null", "medium"])
def test_medium_and_the_default_leave_the_system_layer_as_it_was(tmp_path, stream, extra):
    up = FakeUpstream()
    deps, client = make(tmp_path, up)
    assert post(client, stream, **extra).status_code == 200
    sent = up.bodies[-1]
    assert "verbosity" not in sent and "[Service instruction" not in sent["messages"][0]["content"]
    assert sent["messages"][0]["content"].endswith("You are Ava.")
    t = last_trace(deps.settings)                                          # traced even when omitted
    assert (t["verbosity_requested"], t["verbosity_effective"], t["verbosity_handled_by"]) == (
        extra.get("verbosity"), "medium", "service")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", ["loud", "", "LOW", 3, True, ["low"], {"level": "low"}])
def test_any_other_verbosity_is_a_400_before_the_model(tmp_path, stream, value):
    up = FakeUpstream()
    deps, client = make(tmp_path, up)
    r = post(client, stream, verbosity=value)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_value" and err["param"] == "verbosity"
    assert up.bodies == []


def test_the_service_line_goes_before_a_turn_note(tmp_path, monkeypatch):
    """A voiced specialist result adds our turn note last; the verbosity line
    sits just before it, both inside the one system message."""
    from fastapi.testclient import TestClient
    from chord import specialists
    from chord.config import Settings
    from chord.contract import Outcome, Result
    from chord.server import Deps, create_app, load_specialists
    from test_progress import FixedRouter
    from test_skeleton import PNG
    load_specialists()

    async def render(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")], summary="a cat on a chair")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", render)
    settings = Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset({"image"}))
    up = FakeUpstream()
    client = TestClient(create_app(Deps(settings, upstream=up, model=lambda n: FixedRouter())))
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "verbosity": "high",
                                                  "messages": [{"role": "user", "content": "draw a cat"}]})
    assert r.status_code == 200, r.text
    system = up.bodies[-1]["messages"][0]["content"]
    line = VERBOSITY_LINES["high"]
    assert line in system and not system.endswith(line)                   # a turn note follows it
    assert system.split(line, 1)[1].startswith("\n\n")                     # the note, in the same message
