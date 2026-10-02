"""One retention window bounds the data volume (review 2026-09-22).

Traces, artifacts and failed render attempts used to grow forever; the stores'
own retentions (30-day chats/responses, 24h videos, expires_at files) are
unchanged and not re-tested here.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from chord import retention
from chord.config import Settings
from chord.server import Deps, create_app
from chord.trace import Trace, TraceSink
from test_skeleton import FakeUpstream


def _aged(path, days: int) -> None:
    ts = time.time() - days * 86400
    os.utime(path, (ts, ts))


def _day(offset_days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=offset_days)).strftime("%Y-%m-%d")


def test_sweep_removes_only_what_is_past_the_window(tmp_path):
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / f"{_day(40)}.jsonl").write_text("{}\n")
    (tmp_path / "traces" / f"{_day(10)}.jsonl").write_text("{}\n")
    (tmp_path / "traces" / "not-a-date.jsonl").write_text("{}\n")   # undatable: kept
    (tmp_path / "artifacts").mkdir()
    old_artifact = tmp_path / "artifacts" / "old.png"
    old_artifact.write_bytes(b"x")
    _aged(old_artifact, 40)
    new_artifact = tmp_path / "artifacts" / "new.png"
    new_artifact.write_bytes(b"x")
    (tmp_path / "render-attempts").mkdir()
    old_attempt = tmp_path / "render-attempts" / "old"
    old_attempt.mkdir()
    (old_attempt / "partial.png").write_bytes(b"x")
    _aged(old_attempt, 40)
    new_attempt = tmp_path / "render-attempts" / "new"
    new_attempt.mkdir()

    removed = retention.sweep(tmp_path, 30)

    assert removed == {"traces": 1, "artifacts": 1, "render_attempts": 1, "videos": 0}
    assert not (tmp_path / "traces" / f"{_day(40)}.jsonl").exists()
    assert (tmp_path / "traces" / f"{_day(10)}.jsonl").exists()
    assert (tmp_path / "traces" / "not-a-date.jsonl").exists()
    assert not old_artifact.exists() and new_artifact.exists()
    assert not old_attempt.exists() and new_attempt.exists()


def test_zero_keeps_everything(tmp_path):
    (tmp_path / "traces").mkdir()
    ancient = tmp_path / "traces" / f"{_day(400)}.jsonl"
    ancient.write_text("{}\n")
    (tmp_path / "artifacts").mkdir()
    artifact = tmp_path / "artifacts" / "old.png"
    artifact.write_bytes(b"x")
    _aged(artifact, 400)

    assert retention.sweep(tmp_path, 0) == {"traces": 0, "artifacts": 0, "render_attempts": 0, "videos": 0}
    assert ancient.exists() and artifact.exists()


def test_the_sink_sweeps_once_per_utc_day(tmp_path, monkeypatch):
    calls: list[tuple] = []

    def fake_sweep(data_dir, days):
        calls.append((data_dir, days))
        return {"traces": 0, "artifacts": 0, "render_attempts": 0, "videos": 0}

    monkeypatch.setattr(retention, "sweep", fake_sweep)
    sink = TraceSink(tmp_path / "traces", data_dir=tmp_path, retention_days=30)
    sink.write(Trace(persona_id="generic", model_id_requested=None))
    for _ in range(100):                    # the sweep rides a daemon thread now
        if calls:
            break
        time.sleep(0.02)
    sink.write(Trace(persona_id="generic", model_id_requested=None))
    time.sleep(0.05)                        # any second sweep would have started
    assert calls == [(tmp_path, 30)]        # the day's first write sweeps; the second does not


def test_a_failing_sweep_is_logged_and_never_touches_the_write(tmp_path, monkeypatch, caplog):
    """Two contracts, and the second is the one that was hollow. That a sweep
    failure cannot fail the write is STRUCTURAL now -- the sweep runs on its
    own thread -- so a test asserting only "the write survived" passed with
    the error handling deleted (the batch-4 probe, reproduced: 6/6 green
    without it). What the handler is FOR is observability: a volume that
    quietly stopped sweeping is exactly the silent failure this module exists
    to prevent, so the failure must leave a warning -- asserted here by
    polling caplog, which the thread-boundary does not stop."""
    import logging

    def broken(data_dir, days):
        raise OSError("volume on fire")

    monkeypatch.setattr(retention, "sweep", broken)
    sink = TraceSink(tmp_path / "traces", data_dir=tmp_path, retention_days=30)
    with caplog.at_level(logging.WARNING, logger="chord.trace"):
        sink.write(Trace(persona_id="generic", model_id_requested=None))   # must not raise or block
        assert (tmp_path / "traces" / f"{_day(0)}.jsonl").exists()
        for _ in range(100):                        # the warning arrives on the sweep thread
            if "retention sweep failed" in caplog.text:
                break
            time.sleep(0.02)
        assert "retention sweep failed" in caplog.text, \
            "a failing sweep left no trace: the volume would silently stop being swept"


def test_a_live_request_sweeps_the_volume_end_to_end(tmp_path):
    (tmp_path / "artifacts").mkdir(parents=True)
    stale = tmp_path / "artifacts" / "stale.png"
    stale.write_bytes(b"x")
    _aged(stale, 40)
    settings = Settings(service_api_key="k", data_dir=tmp_path,
                        retention_days=30)
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None)))

    r = client.post("/v1/chat/completions",
                    json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer k"})
    assert r.status_code == 200, r.text
    for _ in range(100):                    # the sweep rides a daemon thread now
        if not stale.exists():
            break
        time.sleep(0.02)
    assert not stale.exists()               # the turn's trace write swept it


def test_the_sweep_runs_off_the_trace_write(tmp_path, monkeypatch):
    """The day's first write used to walk the whole volume inline: on a
    grown directory that stalls every concurrent stream on both servers,
    which share one loop (batch 4). The write must return while the sweep is
    still running."""
    import threading

    from chord import retention
    from chord.trace import Trace, TraceSink

    started = threading.Event()
    hold = threading.Event()

    def blocking(data_dir, days):
        started.set()
        hold.wait(timeout=5)
        return {"traces": 0, "artifacts": 0, "render_attempts": 0, "videos": 0}

    monkeypatch.setattr(retention, "sweep", blocking)
    sink = TraceSink(tmp_path / "traces", data_dir=tmp_path, retention_days=30)
    t0 = time.monotonic()
    sink.write(Trace(persona_id="generic", model_id_requested=None))
    elapsed = time.monotonic() - t0
    hold.set()
    assert elapsed < 1.0, f"the trace write waited {elapsed:.2f}s on the sweep"
    assert started.wait(timeout=2), "the sweep never ran"
