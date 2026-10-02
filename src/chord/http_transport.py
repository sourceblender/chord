"""Shared HTTP transport behavior for streamed OpenAI-compatible routes."""

from __future__ import annotations

from fastapi.responses import StreamingResponse
from starlette.requests import Request

# Multipart boundaries and the non-file fields around the capped part.
BODY_SLACK = 1024 * 1024


async def read_body_within_cap(request: Request, cap: int) -> bytes | None:
    """The whole request body if it fits in `cap` bytes, else None.

    The declared Content-Length is consulted first (an honest oversize reads
    zero bytes), then the stream is counted, so a chunked or lying body is
    refused BEFORE Starlette's form parser spools it to disk without any
    bound at all -- the per-part checks downstream can only refuse what has
    already landed (2026-09-22, #3). The caller assigns the result
    to request._body so form() then parses the capped copy. /v1/files keeps
    its own equivalent with its own pinned envelopes."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > cap:
                return None
        except ValueError:
            pass                       # an unparsable declaration: the counter decides
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > cap:
            return None
    return bytes(buf)


class ClosingStreamingResponse(StreamingResponse):
    """Always close the body iterator when a client leaves.

    Under ASGI 2.4 and later, a failed send does not guarantee the generator is
    closed. Explicit closure is what propagates cancellation to upstream model
    requests instead of leaving them generating for a disconnected client.
    """

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()
