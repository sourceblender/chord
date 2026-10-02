"""Provenance is written on purpose, field by field.

Checkpoints hold state; they do not record which model or prompt answered.
Every node that makes a decision writes it here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ulid import ULID

from . import retention

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    # None on doors that answer for a model the service does not serve (the
    # moderations door traces the request even when persona_for refuses it).
    persona_id: str | None
    model_id_requested: str | None  # None: the Images door had no model (T08)
    path: str = "public"  # "public" | "eval"
    trace_id: str = field(default_factory=lambda: str(ULID()))
    fields: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    progress: list[dict] = field(default_factory=list)
    stt: list[dict] = field(default_factory=list)
    tts: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    _t0: float = field(default_factory=time.monotonic, repr=False)

    def set(self, **kwargs) -> None:
        self.fields.update(kwargs)

    def add_stt(self, **entry) -> None:
        """One speech-to-text call (or cache hit) for an input_audio part."""
        self.stt.append(entry)

    def timed(self, node: str):
        trace = self

        class _Timer:
            def __enter__(self):
                self.t = time.monotonic()

            def __exit__(self, *exc):
                trace.latency_ms[node] = round((time.monotonic() - self.t) * 1000)
                return False

        return _Timer()

    def record(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "at": datetime.now(timezone.utc).isoformat(),
            "path": self.path,
            "persona_id": self.persona_id,
            "model_id_requested": self.model_id_requested,
            **self.fields,
            "tool_calls": self.tool_calls,
            "artifacts": self.artifacts,
            "checks": self.checks,
            "progress": self.progress,
            "stt": self.stt,
            "tts": self.tts,
            "latency_ms": {**self.latency_ms, "total": round((time.monotonic() - self._t0) * 1000)},
        }


class TraceSink:
    def __init__(self, root: Path, *, data_dir: Path | None = None, retention_days: int = 0) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # Volume retention (retention.py): swept once per UTC day, riding the
        # day's first trace write. data_dir None disables (unit constructions);
        # retention_days 0 keeps traces, artifacts and render attempts
        # (videos still expire on their own 24 h).
        self._data_dir = data_dir
        self._retention_days = retention_days
        self._last_swept_day: str | None = None

    def write(self, trace: Trace) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._sweep_once(day)
        with (self.root / f"{day}.jsonl").open("a") as fh:
            fh.write(json.dumps(trace.record(), default=str) + "\n")

    def _sweep_once(self, day: str) -> None:
        if self._data_dir is None or day == self._last_swept_day:
            return
        # Marked before sweeping: a failing sweep retries tomorrow, not on
        # every write of today. And a sweep failure must never fail the write
        # it rode in on -- the trace is the contract, the cleanup is ours.
        self._last_swept_day = day
        data_dir, days = self._data_dir, self._retention_days

        def run() -> None:
            try:
                retention.sweep(data_dir, days)
            except OSError as exc:
                logger.warning("retention sweep failed (%s); traces unaffected", type(exc).__name__)

        # Off the loop: the day's first write used to walk the whole volume
        # inline, and the public and internal servers share ONE loop -- a
        # grown directory stalled every concurrent stream for a filesystem
        # walk (batch 4). A daemon thread: the sweep is idempotent, at most
        # one runs per UTC day (the flag above), and a process exit mid-sweep
        # loses nothing.
        threading.Thread(target=run, daemon=True, name="retention-sweep").start()
