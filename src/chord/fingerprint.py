"""Opaque, installation-scoped backend fingerprinting."""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path


_KEY = b""


def configure(data_dir: Path) -> None:
    """Load or create the per-install key before constructing public responses."""
    global _KEY
    key_file = data_dir / "fingerprint.key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_CREAT|O_EXCL with mode 0600: the key is never world-readable even
        # briefly (write_text creates 0644&umask and chmods after), and an
        # existing key is never overwritten -- a racing second process or a
        # re-run after volume restore would silently invalidate every
        # fingerprint already issued (review 2026-09-22).
        fd = os.open(key_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(os.urandom(32).hex())
    except FileExistsError:
        pass
    _KEY = key_file.read_text().strip().encode()
    if not _KEY:
        # Same crash window as the compaction key: an empty file loaded
        # silently makes every "opaque" fingerprint computable by anyone
        # Refuse to start; the remedy is one
        # file deletion.
        raise RuntimeError(f"{key_file} is empty; delete it to regenerate the install key")


def opaque(value):
    """Preserve change detection without publishing backend identity/version."""
    if not isinstance(value, str) or not value:
        return value
    return "fp_" + hmac.new(_KEY, value.encode(), hashlib.sha256).hexdigest()[:16]
