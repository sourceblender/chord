"""Small robustness fixes from the 2026-10-01 code review (bug 8)."""
import http.server
import os
import pathlib
import shlex
import sqlite3
import subprocess
import sys
import threading

from chord import chat_wire, router
from chord.videos import VideoStore

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_a_non_string_router_question_becomes_a_string():
    decision = router.parse('{"route": "clarify", "question": ["which one?"]}', set())
    assert isinstance(decision.question, str)


def test_an_empty_router_question_is_still_none():
    assert router.parse('{"route": "clarify", "question": ""}', set()).question is None


def test_a_non_dict_provider_block_is_dropped_not_crashed_on():
    payload = {"id": "x", "object": "chat.completion", "created": 1, "model": "m", "choices": [
        {"index": 0, "finish_reason": "stop",
         "message": {"role": "assistant", "content": "hi", "provider_specific_fields": "opaque"}}]}
    out = chat_wire._conform_choices(payload, False)
    message = out["choices"][0]["message"]
    assert message["content"] == "hi" and message["refusal"] is None
    assert "provider_specific_fields" not in message


def test_the_video_index_uses_wal_so_the_sweep_does_not_block_readers(tmp_path):
    live = VideoStore(tmp_path)
    assert live._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    sweep = sqlite3.connect(tmp_path / "index.sqlite3", isolation_level=None)
    sweep.execute("BEGIN EXCLUSIVE")
    sweep.execute("DELETE FROM videos")
    reader = sqlite3.connect(tmp_path / "index.sqlite3", timeout=0.05)
    assert reader.execute("SELECT COUNT(*) FROM videos").fetchone() == (0,)
    sweep.execute("ROLLBACK")


def _healthcheck_argv() -> list[str]:
    line = next(l for l in (ROOT / "Dockerfile").read_text().splitlines() if l.startswith("HEALTHCHECK"))
    argv = shlex.split(line.split(" CMD ", 1)[1])
    assert argv[0] == "python"
    return [sys.executable, *argv[1:]]


def test_the_healthcheck_probes_the_configured_public_port():
    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *_):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        env = {**os.environ, "PUBLIC_PORT": str(server.server_port)}
        done = subprocess.run(_healthcheck_argv(), env=env, capture_output=True, timeout=10)
        assert done.returncode == 0, done.stderr.decode()[-300:]
    finally:
        server.shutdown()
