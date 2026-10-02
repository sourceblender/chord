"""A third request in the gap after a shared transcription finishes must not break the
waiters still resuming from it (review 2026-10-01, bug 7)."""
import asyncio

from chord import audio


class _Trace:
    def add_stt(self, **_kw):
        pass


class _Upstream:
    def __init__(self):
        self.gate = asyncio.Event()
        self.calls = 0

    async def transcribe(self, raw, fmt, mime, model):
        self.calls += 1
        if self.calls == 1:
            await self.gate.wait()
        return f"text-{self.calls}", {}


def test_a_new_request_in_the_completion_gap_does_not_fail_the_old_waiters(monkeypatch):
    async def run():
        up = _Upstream()
        t = audio.Transcriber(up, "stt")
        real = asyncio.ensure_future
        started: list = []

        def spy(coro, **kw):
            task = real(coro, **kw)
            if not started:
                started.append(task)
                # Runs first among the shared task's done callbacks, so the third request is
                # scheduled before the two waiters resume: exactly the window in the review.
                task.add_done_callback(lambda _t: started.append(
                    real(t._hear(b"a", "wav", "d", _Trace()))))
            return task

        monkeypatch.setattr(audio.asyncio, "ensure_future", spy)
        a = real(t._hear(b"a", "wav", "d", _Trace()))
        b = real(t._hear(b"a", "wav", "d", _Trace()))
        await asyncio.sleep(0)
        up.gate.set()
        ra, rb = await asyncio.gather(a, b)
        third = await started[1]
        return ra.text, rb.text, third.text

    texts = asyncio.run(run())
    assert texts[0] == texts[1] == "text-1"
    assert texts[2] in ("text-1", "text-2")
