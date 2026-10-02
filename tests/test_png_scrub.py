"""T01 (red team pass 1, 2026-09-15; T-IMG-001/011/012/013/018): every
generated PNG carried ComfyUI's whole workflow in a tEXt chunk: checkpoint
filenames, sampler, seed, the expanded prompt and the archive node. The
artifact store now keeps only critical and rendering chunks. Chunks are
copied whole, so the pixels are provably unchanged."""
import base64
import io
import json
import struct
import zlib

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from chord.artifacts import ArtifactError, ArtifactStore, scrub_png

WORKFLOW = json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 1234, "sampler_name": "dpmpp_2m"}},
                       "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "secret-model.safetensors"}},
                       "9": {"class_type": "SaveToImmich", "inputs": {"album": "private-album"}}})


def chunks(png: bytes) -> list[tuple[bytes, bytes]]:
    out, pos = [], 8
    while pos < len(png):
        n = int.from_bytes(png[pos:pos + 4], "big")
        out.append((png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n]))
        pos += 12 + n
    return out


def raw_chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def comfy_png(side: int = 32) -> bytes:
    """What ComfyUI hands us, plus every other kind of text/metadata chunk and
    the rendering chunks that must survive."""
    im = Image.new("RGB", (side, side))
    im.putdata([((x * 8) % 256, (y * 8) % 256, ((x ^ y) * 8) % 256)
                for y in range(side) for x in range(side)])
    info = PngInfo()
    info.add_text("prompt", WORKFLOW)                        # tEXt, as ComfyUI writes it
    info.add_text("workflow", WORKFLOW, zip=True)           # zTXt
    info.add_itxt("parameters", "steps: 30, cfg: 5", lang="en", tkey="parameters")  # iTXt
    exif = Image.Exif()
    exif[0x010F] = "Comfy host"
    out = io.BytesIO()
    im.save(out, format="PNG", pnginfo=info, dpi=(72, 72), exif=exif)   # pHYs, eXIf
    png = out.getvalue()
    at = png.index(b"IDAT") - 4                              # tIME, sRGB and a private chunk before IDAT
    extra = (raw_chunk(b"tIME", bytes([7, 234, 9, 15, 12, 0, 0])) + raw_chunk(b"gAMA", struct.pack(">I", 45455))
             + raw_chunk(b"sRGB", b"\x00") + raw_chunk(b"prVt", b"host=render-host") + iccp_named(b"icc from secret-model"))
    return png[:at] + extra + png[at:]


def test_the_comfy_fixture_really_carries_the_leak():
    kinds = {c for c, _ in chunks(comfy_png())}
    assert {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME", b"prVt", b"iCCP"} <= kinds
    assert b"secret-model.safetensors" in comfy_png()


def test_only_critical_and_rendering_chunks_survive_and_the_pixels_are_identical(tmp_path):
    original = comfy_png()
    store = ArtifactStore(tmp_path)
    d = store.register(original, "image/png")
    stored = (tmp_path / f"{d.id}.png").read_bytes()

    kinds = [c for c, _ in chunks(stored)]
    assert set(kinds) == {b"IHDR", b"pHYs", b"gAMA", b"sRGB", b"IDAT", b"IEND"}
    for leak in (b"secret-model", b"KSampler", b"SaveToImmich", b"Comfy host", b"render-host", b"steps: 30"):
        assert leak not in stored
    # IHDR and IDAT byte-identical, in order: the pixels cannot have changed.
    def keep(png):
        return [(c, b) for c, b in chunks(png) if c in (b"IHDR", b"PLTE", b"IDAT", b"IEND")]

    assert keep(stored) == keep(original)
    a, b = Image.open(io.BytesIO(original)), Image.open(io.BytesIO(stored))
    assert (
        a.size == b.size
        and a.mode == b.mode
        and list(a.get_flattened_data()) == list(b.get_flattened_data())
    )
    import hashlib
    assert d.sha256 == hashlib.sha256(stored).hexdigest()   # the hash describes what is served


def test_what_was_dropped_is_named():
    _, dropped = scrub_png(comfy_png())
    assert set(dropped) == {"tEXt", "zTXt", "iTXt", "eXIf", "tIME", "prVt", "iCCP"}


def test_a_clean_png_is_unchanged():
    out = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(out, format="PNG")
    assert scrub_png(out.getvalue()) == (out.getvalue(), [])


@pytest.mark.parametrize("bad", [
    b"\x89PNG\r\n\x1a\n",                                    # no chunks at all
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,                       # the old test filler
    b"\x89PNG\r\n\x1a\nfixture-only",
    raw_chunk(b"IHDR", b"x" * 13).join([b"\x89PNG\r\n\x1a\n", b""]),   # no IEND
    b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 999) + b"IDAT" + b"short",  # length past the end
    b"\x89PNG\r\n\x1a\n" + raw_chunk(b"ID1T", b"") + raw_chunk(b"IEND", b""),  # a type that isn't letters
])
def test_a_png_that_does_not_parse_is_refused(tmp_path, bad):
    with pytest.raises(ArtifactError):
        ArtifactStore(tmp_path).register(bad, "image/png")
    assert list(tmp_path.iterdir()) == []                   # nothing stored


def test_the_images_door_and_the_artifact_route_serve_the_scrubbed_bytes(tmp_path, monkeypatch):
    from test_images_api import configured_comfy_client

    original = comfy_png(side=1024)
    client, _, _ = configured_comfy_client(tmp_path, monkeypatch, original)
    r = client.post("/v1/images/generations", json={"model": "chord-1-poly", "prompt": "a gradient", "size": "1024x1024"})
    assert r.status_code == 200, r.text
    served = base64.b64decode(r.json()["data"][0]["b64_json"])
    assert b"tEXt" not in served and b"secret-model" not in served and b"SaveToImmich" not in served
    assert list(Image.open(io.BytesIO(served)).get_flattened_data()) == list(
        Image.open(io.BytesIO(original)).get_flattened_data()
    )


# the four bypasses on d3a5aa2, and the structure they exposed.
def clean_png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (4, 4), (9, 8, 7)).save(out, format="PNG")
    return out.getvalue()


def insert_before_idat(png: bytes, extra: bytes) -> bytes:
    at = png.index(b"IDAT") - 4
    return png[:at] + extra + png[at:]


def iccp_named(name: bytes) -> bytes:
    return raw_chunk(b"iCCP", name + b"\x00\x00" + zlib.compress(b"not really an icc profile"))


def test_iccp_is_dropped_because_its_profile_name_is_free_text(tmp_path):
    png = insert_before_idat(clean_png(), iccp_named(b"secret-model.safetensors"))
    d = ArtifactStore(tmp_path).register(png, "image/png")
    stored = (tmp_path / f"{d.id}.png").read_bytes()
    assert b"secret-model" not in stored and b"iCCP" not in [c for c, _ in chunks(stored)]


@pytest.mark.parametrize("name,bad", [
    ("unknown critical chunk", lambda: insert_before_idat(clean_png(), raw_chunk(b"SECR", b"internal-host-and-prompt"))),
    ("corrupted CRC", lambda: (lambda p: p[:p.index(b"IDAT") + 8] + bytes([p[p.index(b"IDAT") + 8] ^ 1]) + p[p.index(b"IDAT") + 9:])(clean_png())),
    ("signature + IEND only", lambda: b"\x89PNG\r\n\x1a\n" + raw_chunk(b"IEND", b"")),
    ("IHDR not first", lambda: (lambda p: p[:8] + raw_chunk(b"gAMA", struct.pack(">I", 45455)) + p[8:])(clean_png())),
    ("IHDR twice", lambda: insert_before_idat(clean_png(), clean_png()[8:33])),
    ("IHDR the wrong length", lambda: b"\x89PNG\r\n\x1a\n" + raw_chunk(b"IHDR", b"x" * 12) + raw_chunk(b"IDAT", b"") + raw_chunk(b"IEND", b"")),
    ("no IDAT", lambda: b"\x89PNG\r\n\x1a\n" + clean_png()[8:33] + raw_chunk(b"IEND", b"")),
    ("IDAT split around another chunk", lambda: (lambda p, i: p[:i] + raw_chunk(b"IDAT", b"") + raw_chunk(b"gAMA", struct.pack(">I", 1)) + p[i:])(clean_png(), clean_png().index(b"IDAT") - 4)),
    ("IEND not empty", lambda: clean_png()[:-12] + raw_chunk(b"IEND", b"x")),
])
def test_what_is_not_a_well_formed_png_is_refused(tmp_path, name, bad):
    with pytest.raises(ArtifactError):
        ArtifactStore(tmp_path).register(bad(), "image/png")
    assert list(tmp_path.iterdir()) == []


# Regression from 99abda6: the critical-chunk ordering the PNG spec requires.
def palette_png() -> bytes:
    out = io.BytesIO()
    Image.new("P", (4, 4), 1).save(out, format="PNG")
    return out.getvalue()


def test_the_palette_fixture_is_valid(tmp_path):
    assert b"PLTE" in palette_png()
    ArtifactStore(tmp_path).register(palette_png(), "image/png")


def plte_of(png: bytes) -> bytes:
    at = png.index(b"PLTE") - 4
    return png[at:at + 12 + int.from_bytes(png[at:at + 4], "big")]


@pytest.mark.parametrize("name,bad", [
    ("a second PLTE", lambda p: insert_before_idat(p, plte_of(p))),
    ("PLTE after IDAT", lambda p: p[:-12] + plte_of(p) + p[-12:]),
    ("a second IEND", lambda p: p + raw_chunk(b"IEND", b"")),
    ("bytes after IEND", lambda p: p + b"trailing"),
])
def test_the_png_chunk_order_is_enforced(tmp_path, name, bad):
    with pytest.raises(ArtifactError):
        ArtifactStore(tmp_path).register(bad(palette_png()), "image/png")
    assert list(tmp_path.iterdir()) == []
