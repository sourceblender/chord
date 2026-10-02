"""Job progress: a specialist reports what just happened. Since
2026-09-16 it is traced, never shown: no line in her reply, no status event
(the external interface is the spec)."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app, load_specialists

from test_skeleton import AvailableImageBackend, PNG, FakeUpstream

# Register the real specialists now, so a test's stand-in isn't overwritten
# when the first Deps() imports them.
load_specialists()

PREPARING = "_Preparing the image…_\n\n"
SUBMITTING = "_Sending the render request…_\n\n"


class FixedRouter:
    async def ainvoke(self, msgs):
        class R: content = '{"route": "image", "intent": "a mug"}'
        return R()


def image_settings(tmp_path):
    return Settings(data_dir=tmp_path, router_enabled=True, experimental_routes=frozenset({"image"}))


def stream_chunks(client, text="draw a mug"):
    with client.stream("POST", "/v1/chat/completions", json={"model": "chord-1-poly", "stream": True,
                                                              "messages": [{"role": "user", "content": text}]}) as r:
        return [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]


def content_of(chunks):
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))


def last_trace(settings):
    lines = [l for p in sorted(Path(settings.trace_dir).glob("*.jsonl")) for l in p.read_text().splitlines()]
    return json.loads(lines[-1])


def test_progress_is_traced_and_never_reaches_a_stream(tmp_path, monkeypatch):
    async def fake_image(job, ctx):
        ctx.progress("preparing")
        ctx.progress("submitting")
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a mug")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)
    settings = image_settings(tmp_path)
    chunks = stream_chunks(TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                                                     image_backend=AvailableImageBackend()))))

    assert content_of(chunks).startswith("hi there\n\n![image](") and "Preparing" not in content_of(chunks)
    assert not any("event" in c or "provider_specific_fields" in json.dumps(c) for c in chunks)
    assert [p["stage"] for p in last_trace(settings)["progress"]] == ["preparing", "submitting"]


def test_non_streaming_traces_progress_but_never_puts_it_in_her_reply(tmp_path, monkeypatch):
    async def fake_image(job, ctx):
        ctx.progress("preparing")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.failed, summary="render failed")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)
    settings = image_settings(tmp_path)
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                                        image_backend=AvailableImageBackend())))
    msg = client.post("/v1/chat/completions", json={"model": "chord-1-poly",
                                                    "messages": [{"role": "user", "content": "draw a mug"}]}).json()["choices"][0]["message"]
    assert msg["content"] == "hi there" and "outcome" not in msg
    assert [p["stage"] for p in last_trace(settings)["progress"]] == ["preparing"]


def test_replayed_progress_lines_are_stripped_from_her_history_only(tmp_path):
    up = FakeUpstream()
    client = TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=up, model=lambda n: None)))
    quoted = f"what does {PREPARING.strip()} mean?"
    client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [
        {"role": "user", "content": PREPARING + "draw a mug"},  # a user's text is never touched
        {"role": "assistant", "content": PREPARING + SUBMITTING + "Here it is."},
        {"role": "assistant", "content": "I said " + PREPARING},  # not at the start: hers, kept
        {"role": "user", "content": quoted}]})
    sent = [m["content"] for m in up.bodies[0]["messages"] if m["role"] != "system"]
    assert sent == [PREPARING + "draw a mug", "Here it is.", "I said " + PREPARING, quoted]


def test_every_turn_without_an_artifact_tells_her_nothing_was_made(tmp_path, monkeypatch):
    """the watering-can turn and the preflight fixture: the expert skipped its
    tool, and she described a picture that did not exist. Every outcome that
    isn't `completed` now carries the plain fact to her voice."""
    from chord.graph import NOTHING_MADE

    outcomes = {
        "failed": Outcome.failed, "needs_clarification": Outcome.needs_clarification, "cancelled": Outcome.cancelled,
    }
    for name, status in outcomes.items():
        async def fake_image(job, ctx, status=status):
            return Result(job_id=job.job_id, revision=job.revision, status=status, question="Which bench?", summary="nothing")

        monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)
        up = FakeUpstream()
        client = TestClient(create_app(Deps(image_settings(tmp_path), upstream=up, model=lambda n: FixedRouter(),
                                            image_backend=AvailableImageBackend())))
        client.post("/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a mug"}]})
        assert NOTHING_MADE in up.bodies[0]["messages"][0]["content"], name

    async def made(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a mug")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", made)
    up = FakeUpstream()
    TestClient(create_app(Deps(image_settings(tmp_path), upstream=up, model=lambda n: FixedRouter(),
                               image_backend=AvailableImageBackend()))).post(
        "/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a mug"}]})
    assert NOTHING_MADE not in up.bodies[0]["messages"][0]["content"]


def test_ingress_progress_counts_detect_content_and_reasoning_before_cleanup(tmp_path):
    from test_skeleton import make
    _, client = make(tmp_path)
    body = {"model": "chord-1-poly", "messages": [
        {"role": "assistant", "content": PREPARING,
         "reasoning_content": "Sending the render request…"},
        {"role": "assistant", "content": [{"type": "text", "text": "Sending the render request…"}],
         "reasoning_details": [{"type": "reasoning.text", "text": "Preparing the image…"}]},
        {"role": "user", "content": "received"},
    ]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    counts = last_trace(Settings(data_dir=tmp_path))["ingress_progress"]
    assert counts["messages"] == 3
    assert counts["matches"]["content"] == {"preparing": 1, "submitting": 1, "retrying": 0}
    assert counts["matches"]["reasoning_content"]["submitting"] == 1
    assert counts["matches"]["reasoning_details"]["preparing"] == 1
    assert "Preparing the image" not in json.dumps(counts)


def test_ingress_progress_ignores_status_metadata(tmp_path):
    from test_skeleton import make
    _, client = make(tmp_path)
    body = {"model": "chord-1-poly", "messages": [
        {"role": "assistant", "content": "An image.",
         "statusHistory": [{"description": "Preparing the image…", "done": True}]},
        {"role": "user", "content": "received"},
    ]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    counts = last_trace(Settings(data_dir=tmp_path))["ingress_progress"]
    assert counts["messages"] == 2
    assert all(n == 0 for fields in counts["matches"].values() for n in fields.values())


def test_completed_image_note_names_the_fake_tool_call_failure(tmp_path, monkeypatch):
    """the C2: image attached, but the caption said "Let me create that" and
    wrote a JSON action. The note must say it's done and forbid exactly that."""
    async def made(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a lighthouse")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", made)
    up = FakeUpstream()
    TestClient(create_app(Deps(image_settings(tmp_path), upstream=up, model=lambda n: FixedRouter(),
                               image_backend=AvailableImageBackend()))).post(
        "/v1/chat/completions", json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "draw a lighthouse"}]})
    note = up.bodies[0]["messages"][0]["content"]
    for phrase in ("already made and attached", "no tool", "JSON", '"action"', "don't say you're about to make it",
                   "You haven't seen the finished picture"):
        assert phrase in note, phrase
