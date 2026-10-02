"""One admission budget for everything that waits on the ComfyUI GPU.

Videos, image edits and image variations serialise on one lock in
`video.VideoBackend`. Videos were admitted against `videos.MAX_OUTSTANDING`;
edits and variations waited on the same lock INLINE, uncounted, each pinning
its upload (up to 25 MB for an edit, 4 MB for a variation) and its connection
for up to one render timeout waiting plus one rendering (review 2026-09-24
B10). Now all three share the one budget: video rows that are queued or
rendering, plus edits and variations currently in flight.

The budget lives on `app.state`, so every app (and every test app) has its own.
Counting and admitting happen with no await between them, so on one event loop
the check and the increment cannot interleave with another request's.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from . import videos


class RenderAdmission:
    def __init__(self) -> None:
        self.inline = 0
        # Replaced by the video route with its store's queued+rendering count.
        self.video_rows: Callable[[], int] = lambda: 0

    def outstanding(self) -> int:
        return self.video_rows() + self.inline

    def full(self) -> bool:
        # Read at call time so the one constant (and a test's monkeypatch of
        # it) governs every door.
        return self.outstanding() >= videos.MAX_OUTSTANDING

    async def run_inline(self, what: str,
                         handler: Callable[[], Awaitable[Response]]) -> Response:
        """Run an inline render request inside one slot of the budget, or
        refuse it with the video route's 429 before `handler` reads anything."""
        if self.full():
            return refusal(what)
        self.inline += 1
        try:
            return await handler()
        finally:
            self.inline -= 1


def refusal(what: str) -> JSONResponse:
    """The video route's queue-full envelope (the spec knows no such status)."""
    limit = videos.MAX_OUTSTANDING
    return JSONResponse(
        {"error": {
            "message": (f"too many {what} are already queued; videos, image edits and "
                        f"image variations share one GPU, retry once fewer than {limit} "
                        f"are outstanding"),
            "type": "rate_limit_error",
            "param": None,
            "code": "rate_limit_exceeded",
        }},
        status_code=429,
    )


def admission_for(app: FastAPI) -> RenderAdmission:
    """This app's budget, created by whichever family registers first."""
    admission = getattr(app.state, "render_admission", None)
    if admission is None:
        admission = RenderAdmission()
        app.state.render_admission = admission
    return admission
