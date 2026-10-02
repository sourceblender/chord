"""system_fingerprint names nothing internal (S10). Found by the independent S12 gate on
prod 0c0fb79 (2026-09-17): the stream's usage chunk carried "vllm-0.28.0-96f65c05"."""
import json

import pytest

from fastapi.testclient import TestClient

from chord import server
from chord.config import Settings
from chord.server import Deps, create_app
from test_skeleton import FakeUpstream

RAW = "vllm-0.28.0-96f65c05"


class Fingerprinted(FakeUpstream):
    def __init__(self, fp=RAW):
        super().__init__(); self.fp = fp

    async def stream(self, body):
        async for chunk, dep in super().stream(body):
            if chunk and chunk.get("usage"):
                chunk = {**chunk, "system_fingerprint": self.fp}
            yield chunk, dep


def stream_fingerprints(tmp_path, fp=RAW):
    client = TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=Fingerprinted(fp), model=lambda n: None)))
    with client.stream("POST", "/v1/chat/completions", json={"model": "chord-1-poly", "stream": True,
                       "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": "hi"}]}) as r:
        chunks = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: {")]
    return [c["system_fingerprint"] for c in chunks if "system_fingerprint" in c]


def test_the_streamed_fingerprint_is_opaque(tmp_path):
    fps = stream_fingerprints(tmp_path)
    assert fps and all(f.startswith("fp_") and len(f) == 19 for f in fps)
    assert not any(part in json.dumps(fps) for part in ("vllm", "0.28", "96f65c05"))


def test_it_is_stable_per_install_and_changes_with_the_backend(tmp_path):
    a = stream_fingerprints(tmp_path / "i")[0]
    assert stream_fingerprints(tmp_path / "i")[0] == a                    # same install, same backend
    assert stream_fingerprints(tmp_path / "i", "vllm-0.29.0-aaaa")[0] != a   # backend changed: the value says so
    assert stream_fingerprints(tmp_path / "other")[0] != a                # another install's key: not comparable across


def test_empty_stays_empty():
    assert server.opaque_fingerprint("") == "" and server.opaque_fingerprint(None) is None


def test_the_key_is_created_0600_and_never_overwritten(tmp_path):
    """write_text-then-chmod left a world-readable window, and exists()-then-
    write let a racing second process silently replace the key, invalidating
    every fingerprint already issued. O_CREAT|O_EXCL|0600 closes both
    (review 2026-09-22)."""
    from chord import fingerprint

    fingerprint.configure(tmp_path)
    key_file = tmp_path / "fingerprint.key"
    assert key_file.stat().st_mode & 0o777 == 0o600
    first = key_file.read_text()
    fingerprint.configure(tmp_path)           # a restart, a second process
    assert key_file.read_text() == first      # issued fingerprints stay valid


def test_an_empty_key_file_refuses_startup(tmp_path):
    """Same crash window as the compaction key: an empty file loaded silently
    makes every opaque fingerprint computable by anyone (review, severity
    raised). Refuse; the remedy is deleting the file."""
    from chord import fingerprint

    (tmp_path / "fingerprint.key").write_text("")
    with pytest.raises(RuntimeError, match="empty"):
        fingerprint.configure(tmp_path)
    (tmp_path / "fingerprint.key").unlink()
    fingerprint.configure(tmp_path)            # a fresh install regenerates
    assert (tmp_path / "fingerprint.key").read_text().strip()
