"""Per-client and service-wide admission for the public API process."""

from __future__ import annotations

import asyncio
import time
import math
from collections import defaultdict, deque
from typing import Any

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .trace import Trace, TraceSink


def _now() -> float:
    return time.monotonic()


class AdmissionLease:
    """One in-flight reservation, optionally handed to a background task."""

    def __init__(self, admission: ClientAdmission, client: str) -> None:
        self.admission = admission
        self.client = client
        self.transferred = False
        self.released = False

    def transfer_to(self, task: asyncio.Task[Any]) -> None:
        self.transferred = True
        # A task cancelled before its first step never runs its coroutine's
        # finally block. The completion callback still runs in that case.
        task.add_done_callback(lambda _task: self.release())

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.admission.active -= 1
        self.admission.active_by_client[self.client] -= 1


class ClientAdmission:
    """Bound work after authentication, holding a slot through streaming send.

    Chord serves one Uvicorn worker. There is no await between checking and
    reserving a slot, so requests on that event loop cannot over-admit. A
    multi-worker deployment needs a shared limiter before enabling these caps.
    Zero disables an individual cap during the client-key rollout.
    """

    def __init__(self, app: ASGIApp, *, global_inflight: int,
                 client_inflight: int, global_per_minute: int,
                 client_per_minute: int, traces: TraceSink | None = None) -> None:
        self.app = app
        self.global_inflight = global_inflight
        self.client_inflight = client_inflight
        self.global_per_minute = global_per_minute
        self.client_per_minute = client_per_minute
        self.traces = traces
        self.active = 0
        self.active_by_client: dict[str, int] = defaultdict(int)
        self.global_starts: deque[float] = deque()
        self.starts_by_client: dict[str, deque[float]] = defaultdict(deque)

    @staticmethod
    def _trim(starts: deque[float], now: float) -> None:
        while starts and starts[0] <= now - 60:
            starts.popleft()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path") or ""
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or (path.startswith("/v1/responses/") and path.endswith("/cancel"))):
            await self.app(scope, receive, send)
            return
        state: dict[str, Any] = scope.get("state") or {}
        client = state.get("client_id")
        if not isinstance(client, str) or not client:
            # This middleware must sit inside authentication. Fail closed if
            # a future middleware reorder bypasses the identity assignment.
            await self._refuse(scope, receive, send, "client identity unavailable", 1)
            return
        now = _now()
        self._trim(self.global_starts, now)
        client_starts = self.starts_by_client[client]
        self._trim(client_starts, now)
        global_busy = self.global_inflight and self.active >= self.global_inflight
        client_busy = self.client_inflight and self.active_by_client[client] >= self.client_inflight
        global_rate = self.global_per_minute and len(self.global_starts) >= self.global_per_minute
        client_rate = self.client_per_minute and len(client_starts) >= self.client_per_minute
        if global_busy or client_busy or global_rate or client_rate:
            retry = max(1,
                        math.ceil(self.global_starts[0] + 60 - now) if global_rate else 1,
                        math.ceil(client_starts[0] + 60 - now) if client_rate else 1)
            if self.traces is not None:
                trace = Trace(persona_id=None, model_id_requested=None)
                trace.set(endpoint=f"{scope.get('method')} {path}", result_status="failed",
                          http_status=429, client_id=client, retry_after_s=retry,
                          admission_limits=[name for name, hit in (
                              ("global_inflight", global_busy), ("client_inflight", client_busy),
                              ("global_per_minute", global_rate), ("client_per_minute", client_rate)
                          ) if hit])
                try:
                    self.traces.write(trace)
                except OSError:
                    pass  # a trace-disk failure must not change the 429 contract
            await self._refuse(scope, receive, send, "request limit reached", retry)
            return
        self.active += 1
        self.active_by_client[client] += 1
        lease = AdmissionLease(self, client)
        state["admission_lease"] = lease
        if self.global_per_minute:
            self.global_starts.append(now)
        if self.client_per_minute:
            client_starts.append(now)
        try:
            await self.app(scope, receive, send)
        finally:
            if not lease.transferred:
                lease.release()

    @staticmethod
    async def _refuse(scope: Scope, receive: Receive, send: Send, message: str,
                      retry_after: int) -> None:
        response = JSONResponse({"error": {
            "message": message, "type": "rate_limit_error", "param": None,
            "code": "rate_limit_exceeded"}}, status_code=429,
            headers={"Retry-After": str(retry_after)})
        await response(scope, receive, send)
