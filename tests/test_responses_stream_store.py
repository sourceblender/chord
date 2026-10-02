"""A streamed Response's id must exist from the moment it is announced.

response.created hands the caller an id; the store write used to happen only
after the LAST event, so a client that disconnected mid-stream held a resp_X
that 404d forever -- and in conversation mode the turn's input was lost with
it (review 2026-09-22, #6). Runs a real uvicorn server because TestClient
cannot drop a socket mid-stream (the pattern test_disconnect.py established).
The mid-stream DELETE case is the batch-1 carry-forward: with the row stored
early, the FINAL write must be the tombstone-checked one, or the resurrection
returns through this door.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import uvicorn

from chord.config import Settings
from chord.server import Deps, create_app


class CountedStreamUpstream:
    """Streams a finite, slow reply so a test can act mid-stream."""

    def __init__(self, chunks: int = 60, delay: float = 0.02) -> None:
        self.chunks, self.delay = chunks, delay
        self.bodies: list = []

    async def complete(self, body):
        raise AssertionError("not used")

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        for i in range(self.chunks):
            yield {"object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {"content": f"w{i} "}, "finish_reason": None}]}, {}
            await asyncio.sleep(self.delay)
        yield {"object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}

    async def aclose(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server(tmp_path, up):
    deps = Deps(Settings(data_dir=tmp_path), upstream=up, model=lambda n: None)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(deps), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, thread, f"http://127.0.0.1:{port}"


def test_a_stream_disconnect_leaves_a_terminal_retrievable_row(tmp_path):
    up = CountedStreamUpstream()
    server, thread, base = _server(tmp_path, up)
    try:
        with httpx.Client(timeout=10) as client:
            rid = None
            with client.stream("POST", f"{base}/v1/responses",
                               json={"model": "chord-1-poly", "input": "go", "stream": True}) as r:
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    rid = rid or (event.get("response") or {}).get("id")
                    if event.get("type") == "response.in_progress" and rid:
                        break                       # drop the connection mid-stream
            assert rid, "response.created never arrived"
            time.sleep(0.5)
            got = client.get(f"{base}/v1/responses/{rid}")
            assert got.status_code == 200, got.status_code   # was 404: the id existed only on the wire
            body = got.json()
            # TERMINAL, not stuck (the required fix): in_progress here would
            # leave a poller polling forever and the chaining refusal blocking
            # previous_response_id forever. The generator's finally failed it.
            assert body["status"] == "failed", body["status"]
            assert body["error"]["message"] == "The response was interrupted before it finished."
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_a_delete_mid_stream_is_not_resurrected_by_the_final_write(tmp_path):
    up = CountedStreamUpstream(chunks=40, delay=0.02)
    server, thread, base = _server(tmp_path, up)
    try:
        with httpx.Client(timeout=10) as streamer, httpx.Client(timeout=10) as admin:
            rid = None
            deleted = False
            completed = False
            with streamer.stream("POST", f"{base}/v1/responses",
                                 json={"model": "chord-1-poly", "input": "go", "stream": True}) as r:
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    rid = rid or (event.get("response") or {}).get("id")
                    kind = event.get("type")
                    if kind == "response.output_text.delta" and not deleted and rid:
                        removed = admin.delete(f"{base}/v1/responses/{rid}")
                        assert removed.status_code == 200, removed.text
                        deleted = True
                    if kind == "response.completed":
                        completed = True
                        break
            assert deleted and completed, "the stream must survive the delete and finish"
            time.sleep(0.3)
            # The final write found no row and stopped: the tombstone holds.
            assert admin.get(f"{base}/v1/responses/{rid}").status_code == 404
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_an_interrupted_stream_stores_the_text_it_already_sent(tmp_path):
    """Review 2026-09-24 B21: the disconnect write stored the message as it was
    opened -- empty, in_progress, with an internal `_index` key -- although its
    deltas had reached the client. It now stores what was said so far."""
    up = CountedStreamUpstream(chunks=60, delay=0.02)
    server, thread, base = _server(tmp_path, up)
    try:
        with httpx.Client(timeout=10) as client:
            rid, heard = None, ""
            with client.stream("POST", f"{base}/v1/responses",
                               json={"model": "chord-1-poly", "input": "go", "stream": True}) as r:
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    rid = rid or (event.get("response") or {}).get("id")
                    if event.get("type") == "response.output_text.delta":
                        heard += event["delta"]
                        if heard.count("w") >= 3:
                            break                   # drop the connection mid-message
            assert rid and heard
            time.sleep(0.5)
            body = client.get(f"{base}/v1/responses/{rid}").json()
            assert body["status"] == "failed", body["status"]
            [message] = body["output"]
            assert message["status"] == "incomplete" and "_index" not in message, message
            [part] = message["content"]
            assert part["type"] == "output_text" and part["text"].startswith(heard), (part["text"], heard)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
