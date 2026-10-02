"""A client that disconnects mid-stream must stop the upstream generation.

Runs a real uvicorn server (TestClient can't simulate a dropped socket) with an
upstream that streams slowly and records whether its stream was closed.
"""
import asyncio
import json
import socket
import threading
import time

import httpx
import uvicorn
import pytest

from chord.config import Settings
from chord.server import Deps, create_app
from test_skeleton import FakeUpstream


class SlowUpstream:
    def __init__(self):
        self.closed = threading.Event()
        self.chunks_sent = 0

    async def complete(self, body):
        raise AssertionError("not used")

    async def stream(self, body):
        try:
            yield None, {"model-api-base": "fake"}
            for i in range(10_000):
                self.chunks_sent += 1
                yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": f"w{i} "}, "finish_reason": None}]}, {}
                await asyncio.sleep(0.02)
        finally:
            self.closed.set()

    async def aclose(self):
        pass


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_client_disconnect_closes_the_upstream_stream(tmp_path):
    up = SlowUpstream()
    deps = Deps(Settings(data_dir=tmp_path), upstream=up, model=lambda n: None)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(deps), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        with httpx.Client(timeout=10) as client:
            with client.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                               json={"model": "chord-1-poly", "stream": True, "messages": [{"role": "user", "content": "go"}]}) as r:
                for i, _ in enumerate(r.iter_lines()):
                    if i > 5:
                        break  # drop the connection mid-stream
        assert up.closed.wait(timeout=5), "upstream stream still open after the client left"
        sent_at_close = up.chunks_sent
        time.sleep(0.3)
        assert up.chunks_sent == sent_at_close, "upstream kept generating after close"
    finally:
        server.should_exit = True
        t.join(timeout=5)


@pytest.mark.parametrize("stage", ["router", "voice"])
def test_disconnect_before_headers_cancels_work_and_writes_trace(tmp_path, stage):
    started, cancelled = threading.Event(), threading.Event()

    class StalledRouter:
        async def ainvoke(self, messages):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    class StalledVoice(FakeUpstream):
        async def stream(self, body):
            self.bodies.append(body)
            started.set()
            try:
                await asyncio.Event().wait()
                yield {}, {}  # async generator; deliberately never emits
            finally:
                cancelled.set()

    up = FakeUpstream() if stage == "router" else StalledVoice()
    settings = Settings(data_dir=tmp_path,
                        router_enabled=stage == "router", router_timeout_s=1.5)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(Deps(settings, upstream=up, model=lambda _: StalledRouter())),
                            host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    sock = None
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started
        body = json.dumps({"model": "chord-1-poly", "stream": True,
                           "messages": [{"role": "user", "content": "hello"}]}).encode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        sock.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                      f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
        assert started.wait(2), "fixture never entered the stalled stage"
        sock.settimeout(0.05)
        with pytest.raises(socket.timeout):
            sock.recv(1)  # no response headers, not merely no content tokens
        sock.close()
        sock = None
        assert cancelled.wait(0.7), "disconnect did not cancel the stalled stage before router timeout"
        for _ in range(100):
            records = [json.loads(line) for path in settings.trace_dir.glob("*.jsonl")
                       for line in path.read_text().splitlines()]
            if records:
                break
            time.sleep(0.01)
        assert len(records) == 1 and records[0]["client_disconnected"] is True
        if stage == "router":
            time.sleep(1.6)  # exceed router deadline: no delayed fallback may start
            assert up.bodies == []
        else:
            assert len(up.bodies) == 1
    finally:
        if sock:
            sock.close()
        server.should_exit = True
        thread.join(timeout=5)


def test_disconnect_during_transcription_cancels_stt_and_starts_nothing(tmp_path):
    """the #33 control: STT runs inside the startup _prime_stream races, so a
    client that leaves while its voice message is being transcribed cancels
    the transcription, and no router or voice call follows."""
    import base64

    started, cancelled = threading.Event(), threading.Event()

    class StalledSTT(FakeUpstream):
        async def transcribe(self, audio_bytes, fmt, mime, model):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    up = StalledSTT()
    settings = Settings(data_dir=tmp_path)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(Deps(settings, upstream=up, model=lambda _: None)),
                            host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    sock = None
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started
        clip = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
        body = json.dumps({"model": "chord-1-poly", "stream": True, "messages": [
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}}]}]}).encode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        sock.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                      f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
        assert started.wait(2), "never reached transcription"
        sock.settimeout(0.05)
        with pytest.raises(socket.timeout):
            sock.recv(1)  # still before headers
        sock.close()
        sock = None
        assert cancelled.wait(0.7), "client left, transcription still running"
        for _ in range(100):
            records = [json.loads(line) for path in settings.trace_dir.glob("*.jsonl")
                       for line in path.read_text().splitlines()]
            if records:
                break
            time.sleep(0.01)
        assert len(records) == 1 and records[0]["client_disconnected"] is True
        assert up.bodies == []  # no voice call after the client left
    finally:
        if sock:
            sock.close()
        server.should_exit = True
        thread.join(timeout=5)


class BlockingCompleteUpstream(SlowUpstream):
    """A non-stream turn whose model call would run for a long time."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def complete(self, body):
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "late"},
                             "finish_reason": "stop"}]}, {}


def test_client_disconnect_cancels_a_non_stream_turn(tmp_path):
    """Review 2026-09-27, #8: the only disconnect watcher was on the stream path, so a
    non-stream turn whose client had gone kept running -- an abandoned image turn held
    the GPU for up to IMAGE_DEADLINE_S (1500 s). The work must be cancelled."""
    up = BlockingCompleteUpstream()
    deps = Deps(Settings(data_dir=tmp_path), upstream=up, model=lambda n: None)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(deps), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        with httpx.Client(timeout=httpx.Timeout(1.5, connect=5)) as client:
            with pytest.raises(httpx.ReadTimeout):
                client.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                            json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "go"}]})
        assert up.started.is_set(), "the model call never started; the test proves nothing"
        assert up.cancelled.wait(timeout=5), "the non-stream turn kept running after the client left"
    finally:
        server.should_exit = True


def test_cancelling_the_watcher_does_not_orphan_the_work():
    """#354: ASGI can cancel the handler itself. The helper's finally
    cancelled the work but awaited only the watcher, so a work task with async
    cancellation cleanup was still running after the helper returned."""
    from chord.chat_api import _cancelled_by_disconnect

    class NeverDisconnects:
        async def receive(self):
            await asyncio.Event().wait()

    async def scenario():
        cleaned = asyncio.Event()

        async def work_body():
            try:
                await asyncio.sleep(30)
            finally:
                await asyncio.sleep(0.05)   # async cleanup, as a real render cancel has
                cleaned.set()

        work = asyncio.create_task(work_body())
        helper = asyncio.create_task(_cancelled_by_disconnect(work, NeverDisconnects()))
        await asyncio.sleep(0.05)
        helper.cancel()
        await asyncio.gather(helper, return_exceptions=True)
        return work.done(), cleaned.is_set()

    done, cleaned = asyncio.run(scenario())
    assert done and cleaned, "the work outlived the cancelled helper"
