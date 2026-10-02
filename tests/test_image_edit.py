"""POST /v1/images/edits and /v1/images/variations through the operator's workflow backends."""
import base64
import json

from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO

from chord.config import Settings
from chord.server import Deps, create_app
from test_skeleton import BOMB_PNG, PNG, FakeUpstream


def _png() -> bytes:
    out = BytesIO()
    Image.new("RGB", (8, 8), "navy").save(out, format="PNG")
    return out.getvalue()


class EditBackend:
    def __init__(self, png: bytes = PNG, fail: bool = False) -> None:
        self.png = png
        self.fail = fail
        self.calls: list[tuple[bytes, str, str]] = []

    async def edit_image(self, image: bytes, filename: str, prompt: str) -> bytes:
        self.calls.append((image, filename, prompt))
        if self.fail:
            raise RuntimeError("comfy down")
        return self.png


def _client(tmp_path, backend):
    settings = Settings(data_dir=tmp_path, service_api_key="k")
    app = create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None,
                          edit_backend=backend, variation_backend=backend))
    return TestClient(app)




def test_variations_return_one_png_per_requested_seed(tmp_path):
    class VaryBackend(EditBackend):
        async def vary_images(self, image, filename, n):
            self.calls.append((image, filename, str(n)))
            return [self.png for _ in range(n)]

    png = _png()
    backend = VaryBackend(png)
    client = _client(tmp_path, backend)
    response = client.post(
        "/v1/images/variations",
        data={"n": "2", "size": "256x256", "model": "dall-e-2"},
        files={"image": ("square.png", png, "image/png")},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2
    assert backend.calls[0][2] == "2"
    refused = client.post(
        "/v1/images/variations",
        data={"n": "2"},
        files={"image": ("wide.png", b"not-a-png", "image/png")},
        headers={"Authorization": "Bearer k"},
    )
    assert refused.status_code == 400
    assert refused.json()["error"]["param"] == "image"




def test_an_edit_returns_the_png_the_backend_rendered(tmp_path):
    backend = EditBackend()
    client = _client(tmp_path, backend)
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "replace the cat with a dalmatian", "model": "chord-1-poly"},
        files={"image": ("cat.png", _png(), "image/png")},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == PNG
    image, filename, prompt = backend.calls[0]
    assert filename == "cat.png"
    assert prompt == "replace the cat with a dalmatian"
    assert image.startswith(b"\x89PNG")


def test_a_stored_file_id_is_the_image(tmp_path):
    backend = EditBackend()
    client = _client(tmp_path, backend)
    headers = {"Authorization": "Bearer k"}
    uploaded = client.post(
        "/v1/files",
        data={"purpose": "user_data"},
        files={"file": ("cat.png", _png(), "image/png")},
        headers=headers,
    )
    assert uploaded.status_code == 200, uploaded.text
    response = client.post(
        "/v1/images/edits",
        json={"prompt": "add a hat", "images": [{"file_id": uploaded.json()["id"]}]},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert backend.calls[0][1] == "cat.png"
    assert backend.calls[0][2] == "add a hat"
    refused = client.post(
        "/v1/images/edits",
        json={"prompt": "add a hat", "images": [{"image_url": "https://example.com/a.png"}]},
        headers=headers,
    )
    assert refused.status_code == 400
    assert refused.json()["error"]["param"] == "image_url"


def test_a_decompression_bomb_is_refused_by_edits_and_variations(tmp_path):
    # Pillow raises DecompressionBombError at Image.open for the 66-byte bomb;
    # it inherits from Exception alone, so before the fix both doors answered
    # an unhandled 500 instead of a structured 400 (review 2026-09-22).
    backend = EditBackend()
    client = _client(tmp_path, backend)
    headers = {"Authorization": "Bearer k"}
    edit = client.post(
        "/v1/images/edits",
        data={"prompt": "add a hat", "model": "chord-1-poly"},
        files={"image": ("bomb.png", BOMB_PNG, "image/png")},
        headers=headers,
    )
    assert edit.status_code == 400
    assert edit.json()["error"]["param"] == "image"
    variation = client.post(
        "/v1/images/variations",
        data={"n": "1", "size": "256x256", "model": "dall-e-2"},
        files={"image": ("bomb.png", BOMB_PNG, "image/png")},
        headers=headers,
    )
    assert variation.status_code == 400
    assert variation.json()["error"]["param"] == "image"
    assert backend.calls == []


def test_mask_and_a_missing_image_are_refused(tmp_path):
    backend = EditBackend()
    client = _client(tmp_path, backend)
    headers = {"Authorization": "Bearer k"}
    masked = client.post(
        "/v1/images/edits",
        data={"prompt": "erase the sign"},
        files={"image": ("cat.png", _png(), "image/png"), "mask": ("mask.png", _png(), "image/png")},
        headers=headers,
    )
    assert masked.status_code == 400
    assert masked.json()["error"]["param"] == "mask"
    missing = client.post(
        "/v1/images/edits",
        files={"prompt": (None, "erase the sign")},
        headers=headers,
    )
    assert missing.status_code == 400
    assert missing.json()["error"]["param"] == "image"
    assert backend.calls == []


def test_a_backend_failure_does_not_return_the_exception(tmp_path):
    client = _client(tmp_path, EditBackend(fail=True))
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "replace the cat"},
        files={"image": ("cat.png", _png(), "image/png")},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "the image could not be edited right now"
    assert "comfy" not in json.dumps(response.json())


def test_a_scalar_field_sent_as_a_file_is_named_in_the_refusal(tmp_path):
    """`n` and `response_format` are text. Sent as file parts they used to fail quietly.

    A multipart value is `str | UploadFile`. `int(UploadFile)` landed in the
    ValueError branch and answered "n must be between 1 and 10" — true of a value
    the caller never sent. A misfiled `response_format` is truthy and never equals
    "url", so it was ignored and the caller silently got b64_json. Both now name
    the field that is the wrong shape.
    """
    class VaryBackend(EditBackend):
        async def vary_images(self, image, filename, n):
            return [self.png for _ in range(n)]

    png = _png()
    client = _client(tmp_path, VaryBackend(png))

    misfiled_n = client.post(
        "/v1/images/variations",
        files={"image": ("square.png", png, "image/png"), "n": ("n.txt", b"2", "text/plain")},
        headers={"Authorization": "Bearer k"},
    )
    assert misfiled_n.status_code == 400, misfiled_n.text
    assert misfiled_n.json()["error"]["param"] == "n"
    assert "not a file" in misfiled_n.json()["error"]["message"]

    misfiled_format = client.post(
        "/v1/images/edits",
        data={"prompt": "make it dusk"},
        files={
            "image": ("square.png", png, "image/png"),
            "response_format": ("f.txt", b"url", "text/plain"),
        },
        headers={"Authorization": "Bearer k"},
    )
    assert misfiled_format.status_code == 400, misfiled_format.text
    assert misfiled_format.json()["error"]["param"] == "response_format"


def test_an_oversized_stored_file_is_refused_by_edits(tmp_path, monkeypatch):
    """The file_id branch is capped before the bytes are resident; the
    downstream _edit_source check can only refuse what is already in memory
    (review 2026-09-22). The spy pins WHICH guarantee answered: the old code
    returned the same 413, but only after loading the whole file."""
    from chord import files_api

    monkeypatch.setattr("chord.images_api.MAX_EDIT_BYTES", 8)
    seen_caps: list = []
    real_load = files_api.load_stored_file

    def spy(store, files_dir, file_id, max_bytes=None):
        seen_caps.append(max_bytes)
        return real_load(store, files_dir, file_id, max_bytes)

    monkeypatch.setattr(files_api, "load_stored_file", spy)
    backend = EditBackend()
    client = _client(tmp_path, backend)
    headers = {"Authorization": "Bearer k"}
    uploaded = client.post("/v1/files", data={"purpose": "user_data"},
                           files={"file": ("cat.png", _png(), "image/png")}, headers=headers)
    assert uploaded.status_code == 200, uploaded.text
    r = client.post("/v1/images/edits",
                    json={"prompt": "add a hat", "images": [{"file_id": uploaded.json()["id"]}]},
                    headers=headers)
    assert r.status_code == 413, r.text
    assert r.json()["error"]["param"] == "image"
    assert r.json()["error"]["code"] == "file_too_large"
    assert seen_caps == [8]          # capped AT LOAD, not measured after
    assert backend.calls == []


def _chunk_types(png: bytes) -> set[bytes]:
    kinds, pos = set(), 8
    while pos < len(png):
        length = int.from_bytes(png[pos:pos + 4], "big")
        kinds.add(png[pos + 4:pos + 8])
        pos += 12 + length
    return kinds


def test_an_edit_returned_as_b64_json_carries_no_render_metadata(tmp_path):
    """Review 2026-09-24 A4: ComfyUI writes the whole workflow (UNET, LoRAs, prompt)
    into a tEXt chunk. The url path scrubbed it through ArtifactStore.register; the
    b64_json path returned the backend's bytes as they were -- T01's leak, closed for
    generations, still open for edits."""
    from test_png_scrub import comfy_png
    client = _client(tmp_path, EditBackend(comfy_png()))
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "make it night", "model": "chord-1-poly", "response_format": "b64_json"},
        files={"image": ("cat.png", _png(), "image/png")},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200, response.text
    served = base64.b64decode(response.json()["data"][0]["b64_json"])
    assert not _chunk_types(served) & {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}
    assert b"IDAT" in _chunk_types(served)


def test_every_variation_returned_as_b64_json_carries_no_render_metadata(tmp_path):
    """Copilot on #330: variations were scrubbed in the same change as edits, but only
    the edit had a metadata-bearing test. Two layers strip it on this path: the source
    scrub, and the downscale every variation goes through (a PIL re-encode, which
    drops ancillary chunks). The property that matters is the outcome, so that is
    what this pins: red-proofed by removing both layers (it fails), not either alone
    (the other one still holds)."""
    from test_png_scrub import comfy_png

    class VaryBackend(EditBackend):
        async def vary_images(self, image, filename, n):
            return [comfy_png() for _ in range(n)]

    square = BytesIO()
    Image.new("RGB", (64, 64), "teal").save(square, format="PNG")
    client = _client(tmp_path, VaryBackend())
    response = client.post(
        "/v1/images/variations",
        data={"model": "chord-1-poly", "n": "3", "response_format": "b64_json"},
        files={"image": ("square.png", square.getvalue(), "image/png")},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200, response.text
    served = [base64.b64decode(item["b64_json"]) for item in response.json()["data"]]
    assert len(served) == 3
    for png in served:
        assert not _chunk_types(png) & {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}


def _queued_video(i: int) -> dict:
    import time
    return {"id": f"video_seed{i}", "object": "video", "model": "sora-2",
            "status": "queued", "progress": 0, "created_at": int(time.time()),
            "completed_at": None, "expires_at": None, "error": None, "prompt": "seed",
            "remixed_from_video_id": None, "seconds": "4", "size": "720x1280"}


def test_edits_and_variations_share_the_video_admission_budget(tmp_path, monkeypatch):
    """Review 2026-09-24 B10: videos were capped at MAX_OUTSTANDING, but edits
    and variations waited inline on the same GPU lock with no count, each
    pinning its upload (25 MB / 4 MB) for up to two render timeouts. One
    budget now covers all three; past it an edit or a variation gets the
    video route's 429, and the upload body is never read."""
    from chord import images_api
    from chord.videos import VideoStore

    monkeypatch.setattr("chord.videos.MAX_OUTSTANDING", 1)
    reads: list[int] = []
    real_read = images_api.read_body_within_cap

    async def counting_read(request, cap):
        reads.append(cap)
        return await real_read(request, cap)

    monkeypatch.setattr(images_api, "read_body_within_cap", counting_read)

    class VaryBackend(EditBackend):
        async def vary_images(self, image, filename, n):
            self.calls.append((image, filename, str(n)))
            return [self.png for _ in range(n)]

    backend = VaryBackend(_png())
    client = _client(tmp_path, backend)
    VideoStore(tmp_path / "videos").put(_queued_video(0))   # the one slot is a queued video

    square = BytesIO()
    Image.new("RGB", (64, 64), "teal").save(square, format="PNG")
    for path, data, files in (
        ("/v1/images/edits", {"prompt": "night sky", "model": "chord-1-poly"},
         {"image": ("cat.png", _png(), "image/png")}),
        ("/v1/images/variations", {"n": "1"},
         {"image": ("square.png", square.getvalue(), "image/png")}),
    ):
        refused = client.post(path, data=data, files=files, headers={"Authorization": "Bearer k"})
        assert refused.status_code == 429, (path, refused.text)
        error = refused.json()["error"]
        assert error["code"] == "rate_limit_exceeded"
        assert error["type"] == "rate_limit_error"
    assert backend.calls == []
    assert reads == [], "the upload body was read before admission refused it"


def test_an_edit_in_flight_holds_a_slot_of_the_video_budget(tmp_path, monkeypatch):
    """Review 2026-09-24 B10, the other direction: an edit waiting on or
    holding the GPU counts against the same budget a new video is admitted
    by, and its slot frees when the edit returns."""
    import asyncio
    import threading

    monkeypatch.setattr("chord.videos.MAX_OUTSTANDING", 1)

    class HeldBackend(EditBackend):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        async def edit_image(self, image, filename, prompt):
            self.started.set()
            while not self.release.is_set():
                await asyncio.sleep(0.01)
            return await super().edit_image(image, filename, prompt)

    backend = HeldBackend()
    settings = Settings(data_dir=tmp_path, service_api_key="k")
    # The same fake is also the video backend, so /v1/videos reaches admission.
    app = create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None,
                          video_backend=backend, edit_backend=backend, variation_backend=backend))
    headers = {"Authorization": "Bearer k"}

    def edit(client):
        return client.post("/v1/images/edits",
                           data={"prompt": "night sky", "model": "chord-1-poly"},
                           files={"image": ("cat.png", _png(), "image/png")}, headers=headers)

    with TestClient(app) as client:
        results: list = []
        worker = threading.Thread(target=lambda: results.append(edit(client)))
        worker.start()
        try:
            assert backend.started.wait(5), "the edit never reached the backend"
            refused = client.post("/v1/videos", json={"prompt": "waves"}, headers=headers)
            assert refused.status_code == 429, refused.text
            assert refused.json()["error"]["type"] == "rate_limit_error"
        finally:
            backend.release.set()
            worker.join(10)
        assert results and results[0].status_code == 200, results
        assert edit(client).status_code == 200            # the slot came back
