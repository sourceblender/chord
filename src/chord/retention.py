"""One retention window for the data volume (review 2026-09-22).

The sqlite stores keep their own retention (chats/responses 30 days, videos
24 h per spec, files by expires_at). Everything else on the volume used to
grow forever: the daily trace JSONL, the artifact store, and the attempt
directories failed renders keep as evidence. This sweeps all three under one
window, triggered by TraceSink on the first write of each UTC day -- no
background task, O(1) amortized, and the first write after a deploy sweeps
immediately.

Videos ride the same daily trigger but not the same window: their expiry is
the spec's 24 h, enforced by the video store on every read. A volume nobody
read kept every expired MP4 forever, so the sweep also runs the store's own
expiry -- the path GET and list use -- whatever the window, including 0
(review 2026-09-24 B22).

Two preservation rules, both fail-open:
- Trace files are dated by FILENAME (YYYY-MM-DD.jsonl) and removed only when
  that whole day is past the window; a filename that does not parse is kept.
  Never delete what cannot be dated.
- Artifacts are served only through signed links, and no link outlives the
  window. With PUBLIC_ARTIFACT_BASE set, a chat completion delivers each image
  as a signed link with the full ARTIFACT_URL_TTL_S (7 days by default), and a
  STORED completion keeps that link in its history -- so stored completions
  do hold file references, not only inline bytes. The Images API caps its own
  links at 1 h. Both are safe because config refuses a window shorter than
  ARTIFACT_URL_TTL_S: every link expires before the bytes it names are swept,
  and an expired link is refused as unauthenticated before any file lookup
  (review 2026-09-24 B17). Without PUBLIC_ARTIFACT_BASE images are
  inline data: URIs and reference no file at all. A swept artifact's trace
  row keeps its sha256 and metadata; only the bytes retire.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def sweep(data_dir: Path, retention_days: int) -> dict[str, int]:
    """Remove volume data past the window. 0 (or less) keeps traces,
    artifacts and render attempts forever -- but NOT expired videos: their 24 h
    is the spec's, every reader already enforces it, and the sweep drops them
    whatever the window (review 2026-09-24 B22).

    Returns per-directory removal counts, for the log line and for tests."""
    removed = {"traces": 0, "artifacts": 0, "render_attempts": 0, "videos": 0}
    removed["videos"] = _sweep_videos(data_dir)
    if retention_days <= 0:
        _log(removed, retention_days)
        return removed
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_ts = cutoff.timestamp()

    traces = data_dir / "traces"
    if traces.is_dir():
        for path in traces.glob("*.jsonl"):
            try:
                day = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue                     # undatable name: kept, never guessed at
            if day + timedelta(days=1) <= cutoff and path.is_file():
                path.unlink(missing_ok=True)
                removed["traces"] += 1

    artifacts = data_dir / "artifacts"
    if artifacts.is_dir():
        for path in artifacts.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff_ts:
                    path.unlink()
                    removed["artifacts"] += 1
            except OSError:                  # a file lost to a concurrent reader is already gone
                continue

    attempts = data_dir / "render-attempts"
    if attempts.is_dir():
        for path in attempts.iterdir():
            try:
                # One uuid4 directory per render, written once; its mtime is
                # the attempt's time. Successful attempts are removed by the
                # specialist itself -- what remains here is failure evidence.
                if path.is_dir() and path.stat().st_mtime < cutoff_ts:
                    shutil.rmtree(path, ignore_errors=True)
                    removed["render_attempts"] += 1
            except OSError:
                continue

    _log(removed, retention_days)
    return removed


def _log(removed: dict[str, int], retention_days: int) -> None:
    if any(removed.values()):
        logger.info("retention sweep (window %dd) removed %s", retention_days, removed)


def _sweep_videos(data_dir: Path) -> int:
    """Expired video rows and their MP4s, through the store's own expiry.

    `recover_interrupted=False`: this runs beside the live route's store, where
    an in_progress row is a render still running, not a crash to fail. A video
    volume that was never created is left uncreated. A locked or unreadable
    index is logged and skipped -- the other passes still run, and the next
    reader expires the rows anyway."""
    index = data_dir / "videos" / "index.sqlite3"
    if not index.is_file():
        return 0
    from .videos import VideoStore
    try:
        store = VideoStore(index.parent, recover_interrupted=False)
        try:
            return store.sweep_expired()
        finally:
            store.close()
    except (sqlite3.Error, OSError) as exc:
        logger.warning("video retention sweep failed (%s)", type(exc).__name__)
        return 0
