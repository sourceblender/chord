"""speak() must not rewrite the upstream's chunk dicts in place.

The stream loop mutates `delta` -- the citation/guard hold sets content to
"", the caption strip pops tool-call keys, the separator hold empties
content -- and it used to do all of that ON THE DICTS THE UPSTREAM YIELDED.
Real upstreams build fresh dicts per request, so production never noticed;
shared objects did. It already cost one order-dependent failure: a search-route
test holding the citation path rewrote a module-level TEXT fixture through
the graph, and an unrelated recovery test in another file 502'd when the
files ran in the wrong order (batch 4's top item, review).

Copy-on-write at the top of the loop: every mutation lands on our dicts, and
the upstream's objects pass through the whole pipeline untouched.
"""
import pytest  # noqa: F401  (kept for marker parity with other async tests)

from test_skeleton import FakeUpstream, make

BASE = {"model": "chord-1-poly", "messages": [{"role": "user", "content": "hi"}]}


class KeepsItsDeltas(FakeUpstream):
    """Streams reasoning, then a lone '\\n' content delta (exactly what the
    separator hold rewrites), then real content -- and keeps the dict objects
    it yielded so the test can inspect them after the turn."""

    def __init__(self):
        super().__init__()
        self.yielded: list = []

    async def stream(self, body):
        self.bodies.append(body)
        yield None, {}
        d1 = {"reasoning_content": "thinking"}
        d2 = {"content": "\n"}
        d3 = {"content": "hi there"}
        self.yielded.extend([d1, d2, d3])
        for d in (d1, d2, d3):
            yield {"object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": d, "finish_reason": None}]}, {}
        yield {"object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}


def test_the_stream_loop_never_rewrites_the_upstreams_dicts(tmp_path):
    up = KeepsItsDeltas()
    _, client = make(tmp_path, up)
    r = client.post("/v1/chat/completions", json={**BASE, "stream": True})
    assert r.status_code == 200, r.text
    assert "hi there" in r.text, "the wire must still carry the released content"

    assert up.yielded[0] == {"reasoning_content": "thinking"}
    assert up.yielded[1] == {"content": "\n"}, \
        f"the separator hold rewrote the upstream's own delta: {up.yielded[1]}"
    assert up.yielded[2] == {"content": "hi there"}, \
        f"the separator release rewrote the upstream's own delta: {up.yielded[2]}"
