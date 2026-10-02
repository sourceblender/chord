"""Artifacts: the one way a file becomes something a client can fetch.

Specialists call `register` with the exact file their tool produced and return
the descriptor. The source path is recorded in the trace, never returned.
"""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Callable
from pathlib import Path

from ulid import ULID

from .contract import ArtifactDescriptor

def looks_like_mp3(b: bytes) -> bool:
    """An ID3v2 tag, or a valid MPEG audio frame header.

    Three magic byte pairs used to stand in for the header, and they rejected
    valid frames: MPEG-1 with CRC (``\\xff\\xfa``) and both MPEG-2.5 forms
    (``\\xff\\xe3``, ``\\xff\\xe2``) (review 2026-09-24 B18). The header is
    read instead: 11 sync bits set, a version and a layer that are not the
    reserved values, a bitrate index that is not 1111 and a sample-rate index
    that is not 11. Anything else behind an 0xFF byte stays refused. Shared by the speech
    format check and artifact registration, so the two cannot disagree about
    the same TTS bytes."""
    if b[:3] == b"ID3":
        return True
    # A frame header is four bytes; three that look right are not a frame
    # (Copilot on #331, review 2026-09-24 B18).
    if len(b) < 4 or b[0] != 0xFF or b[1] & 0xE0 != 0xE0:
        return False
    version, layer = (b[1] >> 3) & 0b11, (b[1] >> 1) & 0b11
    bitrate, rate = b[2] >> 4, (b[2] >> 2) & 0b11
    return version != 0b01 and layer != 0b00 and bitrate != 0b1111 and rate != 0b11


# mime -> (type, extension, magic-byte check)
_KINDS: dict[str, tuple[str, str, Callable[[bytes], bool]]] = {
    "image/png": ("image", "png", lambda b: b.startswith(b"\x89PNG\r\n\x1a\n")),
    "image/jpeg": ("image", "jpg", lambda b: b.startswith(b"\xff\xd8\xff")),
    "image/webp": ("image", "webp", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP"),
    "audio/wav": ("audio", "wav", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WAVE"),
    "audio/mpeg": ("audio", "mp3", looks_like_mp3),
}


class ArtifactError(ValueError):
    pass


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CRITICAL = frozenset({b"IHDR", b"PLTE", b"IDAT", b"IEND"})
# Ancillary chunks that change how the pixels render, all fixed binary fields.
# Every other ancillary chunk goes: ComfyUI writes its whole workflow
# (checkpoints, sampler, seed, expanded prompt, archive node) into tEXt, and
# zTXt/iTXt/eXIf/tIME carry the same kind of thing (T01, red team 2026-09-15).
# iCCP goes too: its profile name is free text (#176); sRGB is kept.
_PNG_RENDERING = frozenset({b"gAMA", b"cHRM", b"sRGB", b"cICP", b"mDCv", b"cLLi", b"sBIT", b"tRNS", b"bKGD", b"pHYs"})


def scrub_png(data: bytes) -> tuple[bytes, list[str]]:
    """The PNG with only the four critical chunks and the rendering chunks, and
    the chunk types dropped. Chunks are copied whole, so IHDR/PLTE/IDAT are
    byte-identical and the pixels cannot change. Refused (fail closed) unless
    it parses as a PNG: every CRC right, IHDR first and once (13 bytes), PLTE
    at most once and before IDAT, IDAT present and contiguous, IEND once, empty
    and last with nothing after it, and no critical chunk
    PNG doesn't define, since an unknown one would be kept by definition."""
    if not data.startswith(_PNG_SIGNATURE):
        raise ArtifactError("bytes do not look like image/png")
    kept, dropped, seen, pos = [_PNG_SIGNATURE], [], [], len(_PNG_SIGNATURE)
    while True:
        if pos + 12 > len(data):
            raise ArtifactError("PNG ends before IEND")
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data) or not ctype.isalpha():
            raise ArtifactError("malformed PNG chunk")
        if zlib.crc32(data[pos + 4:end - 4]) & 0xFFFFFFFF != int.from_bytes(data[end - 4:end], "big"):
            raise ArtifactError(f"PNG chunk {ctype!r} fails its CRC")
        critical = not ctype[0] & 0x20                        # uppercase first letter
        if critical and ctype not in _PNG_CRITICAL:
            raise ArtifactError(f"unknown critical PNG chunk {ctype!r}")
        if not seen and (ctype != b"IHDR" or length != 13):
            raise ArtifactError("PNG must start with a 13-byte IHDR")
        if seen and ctype == b"IHDR":
            raise ArtifactError("PNG has a second IHDR")
        if ctype == b"IDAT" and b"IDAT" in seen and seen[-1] != b"IDAT":
            raise ArtifactError("PNG IDAT chunks are not contiguous")
        if ctype == b"PLTE" and (b"PLTE" in seen or b"IDAT" in seen):
            raise ArtifactError("PNG PLTE must come once, before IDAT")
        seen.append(ctype)
        if critical or ctype in _PNG_RENDERING:
            kept.append(data[pos:end])
        else:
            dropped.append(ctype.decode("ascii"))
        pos = end
        if ctype == b"IEND":
            if length:
                raise ArtifactError("PNG IEND is not empty")
            break
    if b"IDAT" not in seen:
        raise ArtifactError("PNG has no image data")
    if data[pos:]:
        raise ArtifactError("bytes after PNG IEND")          # IEND is last (#176)
    return b"".join(kept), dropped


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, source: Path | bytes, mime: str) -> ArtifactDescriptor:
        if mime not in _KINDS:
            raise ArtifactError(f"unsupported mime {mime!r}")
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        if not data:
            raise ArtifactError("empty artifact")
        kind, ext, looks_right = _KINDS[mime]
        if not looks_right(data):
            raise ArtifactError(f"bytes do not look like {mime}")
        if mime == "image/png":
            data, _ = scrub_png(data)
        artifact_id = str(ULID())
        (self.root / f"{artifact_id}.{ext}").write_bytes(data)
        return ArtifactDescriptor(
            id=artifact_id, type=kind, mime=mime, sha256=hashlib.sha256(data).hexdigest()
        )

    def locate(self, artifact_id: str) -> tuple[Path, str] | None:
        # ULIDs are Crockford base32; refuse anything that could walk the filesystem.
        if not artifact_id.isalnum():
            return None
        for mime, (_, ext, _) in _KINDS.items():
            path = self.root / f"{artifact_id}.{ext}"
            if path.is_file():
                return path, mime
        return None
