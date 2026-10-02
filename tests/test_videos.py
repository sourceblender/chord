from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from PIL import Image

from chord.config import Settings
from chord.server import Deps, create_app, create_internal_app
from chord.__main__ import effective_config_view
from chord import videos as videos_module
from test_skeleton import BOMB_PNG


MP4 = base64.b64decode(
    "AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAA/5tb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAAD6AAB"
    "AAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAACAAACanRyYWsAAABcdGtoZAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAD6AAAAAAAAAAAAAAAAAAA"
    "AAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAEAAAABAAAAAAACRlZHRzAAAAHGVsc3QAAAAA"
    "AAAAAQAAA+gAAAAAAAEAAAAAAeJtZGlhAAAAIG1kaGQAAAAAAAAAAAAAAAAAAEAAAABAAFXEAAAAAAAtaGRscgAA"
    "AAAAAAAAdmlkZQAAAAAAAAAAAAAAAFZpZGVvSGFuZGxlcgAAAAGNbWluZgAAABR2bWhkAAAAAQAAAAAAAAAAAAAA"
    "JGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAABTXN0YmwAAADpc3RzZAAAAAAAAAABAAAA2W1wNHYA"
    "AAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAEAAQAEgAAABIAAAAAAAAAAESTGF2YzYzLjEuMTAyIG1wZWc0AAAAAAAA"
    "AAAAAAAAAAAY//8AAABfZXNkcwAAAAADgICATgABAASAgIBAIBEAAAAAAw1AAAAAiAWAgIAuAAABsAEAAAG1iRMA"
    "AAEAAAABIADEjYgADQCEAhRjAAABskxhdmM2My4xLjEwMgaAgIABAgAAABBwYXNwAAAAAQAAAAEAAAAUYnRydAAA"
    "AAAAAw1AAAAAiAAAABhzdHRzAAAAAAAAAAEAAAABAABAAAAAABxzdHNjAAAAAAAAAAEAAAABAAAAAQAAAAEAAAAU"
    "c3RzegAAAAAAAAARAAAAAQAAABRzdGNvAAAAAAAAAAEAAAQqAAABIHVkdGEAAAEYbWV0YQAAAAAAAAAhaGRscgAA"
    "AAAAAAAAbWR0YQAAAAAAAAAAAAAAAAAAAABMa2V5cwAAAAAAAAAEAAAAD21kdGFjb21tZW50AAAADm1kdGFwcm9t"
    "cHQAAAAQbWR0YXdvcmtmbG93AAAAD21kdGFlbmNvZGVyAAAAn2lsc3QAAAAoAAAAAQAAACBkYXRhAAAAAQAAAABQ"
    "TEFOVEVEX1dPUktGTE9XAAAAJgAAAAIAAAAeZGF0YQAAAAEAAAAAUExBTlRFRF9QUk9NUFQAAAAlAAAAAwAAAB1k"
    "YXRhAAAAAQAAAABQTEFOVEVEX0dSQVBIAAAAJAAAAAQAAAAcZGF0YQAAAAEAAAAATGF2ZjYzLjEuMTAyAAAACGZy"
    "ZWUAAAAZbWRhdAAAAbMAEAcAAAG2FgUYI9t+"
)


class VideoBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail = False

    async def render(self, kind: str, params: dict, on_start=None, on_submit=None,
                     on_settled=None) -> tuple[bytes, str]:
        self.calls.append((kind, params))
        if on_start is not None:
            on_start()          # the real backend fires this once the GPU lock is held
        if self.fail:
            raise RuntimeError("private comfy failure")
        return MP4, "video/mp4"


class SlowBackend(VideoBackend):
    """Holds the render until released, so a test can act on a live job.

    `release` is the LATEST render's event; sequential renders overwrite it,
    which is what the delete tests want (one live job at a time). The gate
    lock makes the fake serialize like the real GPU: a second render WAITS,
    which is the state the queued-status test observes."""

    def __init__(self) -> None:
        super().__init__()
        self.release: asyncio.Event | None = None
        self.gate = asyncio.Lock()

    async def render(self, kind: str, params: dict, on_start=None, on_submit=None,
                     on_settled=None) -> tuple[bytes, str]:
        async with self.gate:
            self.calls.append((kind, params))
            if on_start is not None:
                on_start()
            self.release = asyncio.Event()
            await self.release.wait()
        return MP4, "video/mp4"


@contextlib.contextmanager
def _client(tmp_path, backend: VideoBackend | None = None, **settings_over):
    backend = backend or VideoBackend()
    settings = Settings(service_api_key="service-key",
                        data_dir=tmp_path, **settings_over)
    deps = Deps(settings, upstream=object(), model=lambda _: None, video_backend=backend)
    # Context-managed on purpose: renders run as tasks on the portal's loop,
    # not as BackgroundTasks inside the create request (that change is what
    # makes DELETE able to cancel the work -- review 2026-09-22, #1). The
    # `with` keeps one loop alive across requests; terminal states are reached
    # by polling wait_status, as the Responses background tests already do.
    with TestClient(create_app(deps)) as client:
        yield client, backend


def wait_status(client, video_id: str, statuses, timeout: float = 5.0) -> dict:
    end = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < end:
        response = client.get(f"/v1/videos/{video_id}", headers=_auth())
        if response.status_code == 200:
            body = response.json()
            if body.get("status") in statuses:
                return body
        time.sleep(0.02)
    raise AssertionError(f"{video_id} never reached {statuses}; last {body}")


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer service-key"}


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "navy").save(out, "PNG")
    return out.getvalue()


def test_prompt_only_renders_t2v_and_content_is_downloadable(tmp_path) -> None:
    with _client(tmp_path) as (client, backend):
        response = client.post("/v1/videos", json={"prompt": "clouds over a lake", "seconds": 8,
                                                    "size": "1280x720"}, headers=_auth())

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "video"
        assert body["status"] == "queued"
        assert body["progress"] == 0
        body = wait_status(client, body["id"], {"completed"})
        assert body["progress"] == 100
        assert body["model"] == "sora-2"
        assert body["seconds"] == "8"
        assert body["size"] == "1280x720"
        kind, params = backend.calls[0]
        assert kind == "t2v"
        assert {k: params[k] for k in ("prompt", "width", "height", "seconds", "reference")} == {
            "prompt": "clouds over a lake", "width": 1280, "height": 720, "seconds": 8, "reference": None,
        }

        content = client.get(f"/v1/videos/{body['id']}/content", headers=_auth())
        assert content.status_code == 200
        assert content.headers["content-type"] == "video/mp4"
        assert content.content != MP4
        assert b"PLANTED_PROMPT" not in content.content
        assert b"PLANTED_GRAPH" not in content.content
        assert b"PLANTED_WORKFLOW" not in content.content


def test_video_metadata_is_removed_without_reencoding(tmp_path) -> None:
    clean = videos_module._scrub_mp4_metadata(MP4)
    assert clean != MP4
    assert all(marker not in clean for marker in (b"PLANTED_PROMPT", b"PLANTED_GRAPH", b"PLANTED_WORKFLOW"))
    path = tmp_path / "clean.mp4"
    path.write_bytes(clean)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags:stream=codec_name",
                            "-of", "json", str(path)], capture_output=True, check=True, text=True)
    metadata = json.loads(probe.stdout)
    assert metadata["streams"][0]["codec_name"] == "mpeg4"
    assert not {"comment", "prompt", "workflow"} & metadata["format"]["tags"].keys()


def test_video_scrub_refuses_without_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(videos_module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="scrubber is unavailable"):
        videos_module._scrub_mp4_metadata(MP4)


def test_video_create_refuses_before_queue_when_tools_are_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(videos_module.shutil, "which", lambda _: None)
    settings = Settings(data_dir=tmp_path, comfy_base_url="http://127.0.0.1:8188")
    assert effective_config_view(settings, yaml_mode=False)["routes"]["video"] == {
        "configured": False, "origin": "http://127.0.0.1:8188",
        "kinds": [], "tools_available": False, "serving": False,
    }
    deps = Deps(settings, upstream=object(), model=lambda _: None, video_backend=VideoBackend())
    health = TestClient(create_internal_app(deps)).get("/internal/health").json()
    assert health["capabilities"]["video"] is False
    with _client(tmp_path) as (client, backend):
        response = client.post("/v1/videos", json={"prompt": "clouds over a lake"}, headers=_auth())
        assert (response.status_code, response.json()["error"]["code"]) == (503, "video_tools_unavailable")
        assert backend.calls == []
        assert client.get("/v1/videos", headers=_auth()).json()["data"] == []


def test_reference_video_refuses_when_only_t2v_workflow_is_configured(tmp_path) -> None:
    backend = VideoBackend()
    backend.supports = lambda kind: kind == "t2v"
    with _client(tmp_path, backend=backend) as (client, _):
        response = client.post("/v1/videos", data={"prompt": "turn", "size": "720x1280"},
                               files={"input_reference": ("ref.png", _png(720, 1280), "image/png")},
                               headers=_auth())
        assert (response.status_code, response.json()["error"]["code"]) == (503, "video_workflow_unavailable")
        assert backend.calls == []


def test_video_scrub_keeps_audio_stream(tmp_path) -> None:
    source = tmp_path / "with-audio.mp4"
    chapter = tmp_path / "chapter.ffmeta"
    chapter.write_text(";FFMETADATA1\ncomment=PLANTED_WORKFLOW\n[CHAPTER]\nTIMEBASE=1/1000\n"
                       "START=0\nEND=500\ntitle=PLANTED_CHAPTER\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=16x16:r=1", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
                    "-f", "ffmetadata", "-i", str(chapter), "-t", "1", "-map", "0:v", "-map", "1:a",
                    "-map_metadata", "2", "-map_chapters", "2", "-metadata:s:v:0", "handler_name=PLANTED_STREAM",
                    "-c:v", "mpeg4", "-c:a", "aac", "-movflags", "+use_metadata_tags",
                    "-y", str(source)], capture_output=True, check=True)
    clean = tmp_path / "clean-audio.mp4"
    clean.write_bytes(videos_module._scrub_mp4_metadata(source.read_bytes()))

    def probe(path):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                 "-show_chapters", "-of", "json", str(path)],
                                capture_output=True, check=True, text=True)
        return json.loads(result.stdout)

    assert all(marker in json.dumps(probe(source)) for marker in
               ("PLANTED_WORKFLOW", "PLANTED_STREAM", "PLANTED_CHAPTER"))
    result = probe(clean)
    assert [stream["codec_name"] for stream in result["streams"]] == ["mpeg4", "aac"]
    assert result["chapters"] == []
    assert all(marker not in json.dumps(result) for marker in
               ("PLANTED_WORKFLOW", "PLANTED_STREAM", "PLANTED_CHAPTER"))


def test_video_scrub_refuses_malformed_media() -> None:
    with pytest.raises(ValueError, match="video metadata scrub failed"):
        videos_module._scrub_mp4_metadata(b"not an MP4")


def test_video_job_fails_closed_without_scrubber(tmp_path, monkeypatch) -> None:
    # The tools can disappear after admission but before the worker remuxes.
    def unavailable(_data: bytes) -> bytes:
        raise RuntimeError("video metadata scrubber is unavailable")

    monkeypatch.setattr(videos_module, "_scrub_mp4_metadata", unavailable)
    with _client(tmp_path) as (client, _backend):
        created = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()
        failed = wait_status(client, created["id"], {"failed"})
        assert failed["error"]["code"] == "generation_failed"
        assert not (tmp_path / "videos" / f"{created['id']}.mp4").exists()


def test_delete_drains_an_inflight_video_scrub(tmp_path, monkeypatch) -> None:
    import threading

    started = threading.Event()
    finished = threading.Event()
    real_scrub = videos_module._scrub_mp4_metadata

    def slow_scrub(data: bytes) -> bytes:
        started.set()
        time.sleep(0.3)
        try:
            return real_scrub(data)
        finally:
            finished.set()

    monkeypatch.setattr(videos_module, "_scrub_mp4_metadata", slow_scrub)
    with _client(tmp_path) as (client, _backend):
        vid = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()["id"]
        assert started.wait(5)
        removed = client.delete(f"/v1/videos/{vid}", headers=_auth())
        assert removed.status_code == 200
        assert finished.is_set(), "DELETE returned while the scrubber still ran"
        assert not (tmp_path / "videos" / f"{vid}.mp4").exists()


def test_list_and_delete_use_the_stored_video(tmp_path) -> None:
    with _client(tmp_path) as (client, _backend):
        first = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()
        second = client.post("/v1/videos", json={"prompt": "two"}, headers=_auth()).json()
        wait_status(client, first["id"], {"completed"})
        wait_status(client, second["id"], {"completed"})

        listed = client.get("/v1/videos", headers=_auth())
        assert listed.status_code == 200
        body = listed.json()
        assert body["object"] == "list"
        assert [item["id"] for item in body["data"]] == [second["id"], first["id"]]
        assert body["first_id"] == second["id"] and body["last_id"] == first["id"]
        assert body["has_more"] is False

        page = client.get(f"/v1/videos?limit=1&after={second['id']}", headers=_auth()).json()
        assert [item["id"] for item in page["data"]] == [first["id"]]
        assert page["has_more"] is False

        removed = client.delete(f"/v1/videos/{second['id']}", headers=_auth())
        assert removed.status_code == 200
        assert removed.json() == {"id": second["id"], "object": "video.deleted", "deleted": True}
        assert client.get(f"/v1/videos/{second['id']}", headers=_auth()).status_code == 404
        assert client.get(f"/v1/videos/{second['id']}/content", headers=_auth()).status_code == 404
        assert [item["id"] for item in client.get("/v1/videos", headers=_auth()).json()["data"]] == [first["id"]]
        missing = client.delete(f"/v1/videos/{second['id']}", headers=_auth())
        assert missing.status_code == 404


def test_a_stored_file_id_is_a_video_reference(tmp_path) -> None:
    with _client(tmp_path) as (client, backend):
        headers = _auth()
        uploaded = client.post(
            "/v1/files",
            data={"purpose": "vision"},
            files={"file": ("reference.png", _png(720, 1280), "image/png")},
            headers=headers,
        )
        assert uploaded.status_code == 200, uploaded.text
        response = client.post(
            "/v1/videos",
            json={"prompt": "the figure turns", "size": "720x1280",
                  "input_reference": {"file_id": uploaded.json()["id"]}},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        wait_status(client, response.json()["id"], {"completed"})
        kind, params = backend.calls[0]
        assert kind == "r2v"
        assert params["reference"].startswith(b"\x89PNG")
        url = client.post(
            "/v1/videos",
            json={"prompt": "the figure turns", "input_reference": {"image_url": "https://example.com/a.png"}},
            headers=headers,
        )
        assert url.status_code == 400
        assert url.json()["error"]["param"] == "image_url"


def test_openai_sdk_multipart_reference_selects_r2v(tmp_path) -> None:
    with _client(tmp_path) as (client, backend):
        sdk = OpenAI(api_key="service-key", base_url="http://testserver/v1", max_retries=0, http_client=client)

        result = sdk.videos.create(
            prompt="the figure turns toward camera",
            model="sora-2-pro",
            seconds="4",
            size="720x1280",
            input_reference=("reference.png", _png(720, 1280), "image/png"),
        )

        assert result.status == "queued"
        end = time.monotonic() + 5.0
        while time.monotonic() < end:
            result = sdk.videos.retrieve(result.id)
            if result.status == "completed":
                break
            time.sleep(0.02)
        assert result.status == "completed"
        assert result.model == "sora-2-pro"
        kind, params = backend.calls[0]
        assert kind == "r2v"
        assert params["reference_filename"] == "reference.png"
        assert params["reference"].startswith(b"\x89PNG")


def test_a_video_survives_restart_until_it_expires(tmp_path) -> None:
    from chord.videos import VideoStore
    root = tmp_path / "videos"
    first = VideoStore(root)
    record = {"id": "video_01hzz", "object": "video", "status": "completed", "progress": 100,
              "expires_at": int(time.time()) + 3600, "prompt": "clouds"}
    first.put(record)
    (root / "video_01hzz.mp4").write_bytes(b"mp4")
    assert VideoStore(root).get("video_01hzz")["status"] == "completed"
    assert VideoStore(root).file("video_01hzz") is not None

    expired = {**record, "expires_at": int(time.time()) - 1}
    VideoStore(root).put(expired)
    assert VideoStore(root).get("video_01hzz") is None
    assert not (root / "video_01hzz.mp4").exists()


def test_a_thumbnail_is_one_frame_of_the_stored_video(tmp_path, monkeypatch) -> None:
    with _client(tmp_path) as (client, _backend):
        created = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()
        wait_status(client, created["id"], {"completed"})
        png = _png(8, 8)
        monkeypatch.setattr("chord.videos._first_frame_png", lambda path: png)
        thumb = client.get(f"/v1/videos/{created['id']}/content?variant=thumbnail", headers=_auth())
        assert thumb.status_code == 200
        assert thumb.headers["content-type"] == "image/png"
        assert thumb.content == png


def test_a_spritesheet_is_a_grid_of_the_stored_video(tmp_path, monkeypatch) -> None:
    """`variant=spritesheet` is served (#321). The stored MP4 here is the fake
    first-draft blob, which no decoder can read, so the extraction is patched --
    the real ffmpeg path is exercised by
    `test_a_spritesheet_is_a_real_grid_from_a_real_video` below."""
    with _client(tmp_path) as (client, _backend):
        created = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()
        wait_status(client, created["id"], {"completed"})
        sheet = _png(64, 48)
        monkeypatch.setattr("chord.videos._spritesheet_png", lambda path: sheet)
        r = client.get(f"/v1/videos/{created['id']}/content?variant=spritesheet", headers=_auth())
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.content == sheet
        assert f'{created["id"]}.png' in r.headers["content-disposition"]


def test_an_unknown_variant_is_still_refused_by_name(tmp_path) -> None:
    """Adding spritesheet must not turn the variant check into a catch-all:
    an unknown value is still a named 400, and the message lists all three."""
    with _client(tmp_path) as (client, _backend):
        created = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()
        wait_status(client, created["id"], {"completed"})
        r = client.get(f"/v1/videos/{created['id']}/content?variant=poster", headers=_auth())
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "variant"
        assert r.json()["error"]["code"] == "unsupported_value"
        assert "spritesheet" in r.json()["error"]["message"]


def test_an_undecodable_video_is_a_502_named_by_variant(tmp_path) -> None:
    """A damaged stored MP4 cannot be probed, so the sheet cannot be built. That is a
    502 naming the variant, never a 500 and never a silent empty 200."""
    with _client(tmp_path) as (client, _backend):
        created = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()
        wait_status(client, created["id"], {"completed"})
        (tmp_path / "videos" / f"{created['id']}.mp4").write_bytes(b"mp4")
        r = client.get(f"/v1/videos/{created['id']}/content?variant=spritesheet", headers=_auth())
        assert r.status_code == 502, r.text
        assert r.json()["error"]["code"] == "spritesheet_unavailable"
        assert r.json()["error"]["param"] == "variant"


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg and ffprobe to build a real sheet")
def test_a_spritesheet_is_a_real_grid_from_a_real_video(tmp_path) -> None:
    """The live proof #321 asks for, at unit scope: a REAL 4-second MP4 goes in
    and a 4x4 grid of evenly-spaced frames comes out. Nothing is patched --
    ffprobe reads the frame count and ffmpeg does the select+tile -- so this
    fails if the filter expression, the step arithmetic, or the probe parsing
    regresses."""
    from chord.videos import SPRITESHEET_GRID, SPRITE_W, _probe_frame_count, _spritesheet_png

    mp4 = tmp_path / "real.mp4"
    built = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=4:size=320x240:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True, check=False,
    )
    if built.returncode != 0:
        pytest.skip(f"ffmpeg could not build a test video: {built.stderr[:200]!r}")

    # 4 seconds at 24 fps is 96 frames -- the exact number the issue names for
    # the 4-second artifact.
    assert _probe_frame_count(mp4) == 96

    png = _spritesheet_png(mp4)
    assert png is not None and png.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(png)) as sheet:
        # 16 frames sampled every 6th (96 // 16), each scaled to SPRITE_W wide.
        # 320x240 scaled to 160 wide is 160x120, tiled 4x4 => 640x480.
        assert sheet.size == (SPRITE_W * SPRITESHEET_GRID, 120 * SPRITESHEET_GRID)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_the_frame_probe_reads_a_real_container(tmp_path) -> None:
    """End-to-end probe integration: a real generated MP4 yields its real frame
    count. This exercises the ffprobe invocation and the JSON shape, not the
    fallback branch -- a generated MP4 carries nb_frames, so the pure-parser
    tests below are what actually pin the fallback."""
    from chord.videos import _probe_frame_count

    mp4 = tmp_path / "real.mp4"
    built = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=64x64:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True, check=False,
    )
    if built.returncode != 0:
        pytest.skip(f"ffmpeg could not build a test video: {built.stderr[:200]!r}")
    # 2 seconds at 10 fps.
    assert _probe_frame_count(mp4) == 20


@pytest.mark.parametrize("stream,expected", [
    # Primary path: nb_frames present and usable.
    ({"nb_frames": "96", "duration": "4.0", "avg_frame_rate": "24/1"}, 96),
    ({"nb_frames": 96}, 96),
    # Fallback path: nb_frames absent, empty, or the "N/A" ffprobe emits for
    # containers that do not carry it. THIS is the branch the end-to-end test
    # above cannot reach, because ffmpeg always writes nb_frames.
    ({"duration": "4.000000", "avg_frame_rate": "24/1"}, 96),
    ({"nb_frames": "N/A", "duration": "4.000000", "avg_frame_rate": "24/1"}, 96),
    ({"nb_frames": "", "duration": "2.0", "avg_frame_rate": "10/1"}, 20),
    # Fractional and unusual frame rates.
    ({"duration": "1.0", "avg_frame_rate": "30000/1001"}, 30),
    ({"duration": "10.0", "avg_frame_rate": "1/2"}, 5),
    # nb_frames present but non-positive: not a usable count, and there is no
    # duration to fall back on either.
    ({"nb_frames": "0"}, None),
    ({"nb_frames": "-5"}, None),
    # Unreadable / missing everything.
    ({}, None),
    ({"duration": "0", "avg_frame_rate": "24/1"}, None),
    ({"duration": "4.0", "avg_frame_rate": "0/0"}, None),   # zero denominator
    ({"duration": "4.0", "avg_frame_rate": "0/1"}, None),   # zero rate
    ({"duration": "not-a-number", "avg_frame_rate": "24/1"}, None),
])
def test_the_probe_parser_falls_back_only_when_nb_frames_is_unusable(stream, expected) -> None:
    """The fallback is a PURE function of one probe entry, so it is testable
    without a container that happens to omit nb_frames. Copilot was right that
    the end-to-end version of this test never reached the branch it claimed to
    cover; asserting the parse directly does.

    The contract is what `_probe_frame_count` relies on: either a usable count,
    or None, or an exception from the (ValueError, TypeError) set that it
    catches and maps to None. An unreadable probe must never escape as a 500."""
    from chord.videos import _frames_from_probe

    def as_the_caller_sees_it(entry):
        try:
            return _frames_from_probe(entry)
        except (ValueError, TypeError):
            return None

    assert as_the_caller_sees_it(stream) == expected


@pytest.mark.parametrize("frames,expected_step,expected_selected", [
    (96, 6, 16),      # exact multiple: 16 frames, the case the real-MP4 test hits
    (100, 7, 15),     # floor would give step 6 -> 17 selected, tile discards the 17th
    (20, 2, 10),      # floor would give step 1 -> all 20, sheet never reaches the end
    (16, 1, 16),
    (17, 2, 9),
    (1, 1, 1),        # a one-frame clip
    (1000, 63, 16),
])
def test_the_sprite_step_never_over_selects(frames, expected_step, expected_selected) -> None:
    """Ceiling, not floor. Floor division over-selects whenever the count is not
    a multiple of the grid, and `tile=4x4` keeps only the FIRST 16 -- so a
    100-frame clip stopped at frame 90 and a 20-frame clip showed only frames
    0-15, never reaching the end. The sheet is supposed to span the whole clip
    (Copilot review, #329)."""
    from chord.videos import SPRITESHEET_FRAMES, sprite_step

    step = sprite_step(frames)
    assert step == expected_step
    selected = len(range(0, frames, step))
    assert selected == expected_selected
    assert selected <= SPRITESHEET_FRAMES, "tile discards the overflow, so over-selecting loses the tail"
    # And the last sampled frame is near the end of the clip, which is the
    # property "spans the whole clip" actually means.
    assert max(range(0, frames, step)) >= frames - step


def test_the_sprite_step_is_at_least_one() -> None:
    """A step of 0 would make `mod(n,0)` a division by zero inside ffmpeg."""
    from chord.videos import sprite_step

    assert sprite_step(1) == 1
    assert sprite_step(0) == 1


@pytest.mark.parametrize("stream", [
    {"duration": "inf", "avg_frame_rate": "24/1"},      # float("inf") -> int() OverflowError
    {"duration": "1e400", "avg_frame_rate": "24/1"},    # parses to inf
    {"duration": "4.0", "avg_frame_rate": "inf"},       # infinite rate
    {"nb_frames": "1" + "0" * 400},                     # a 400-digit count
    {"nb_frames": "-" + "9" * 400},
])
def test_an_absurd_probe_value_is_unreadable_not_an_exception(stream) -> None:
    """A corrupt container can report infinite duration or an astronomically
    large nb_frames. Both used to escape the caller's narrow (ValueError,
    TypeError, JSONDecodeError) handler -- OverflowError is none of those -- and
    reach the route as a 500. The parser now refuses them as unreadable."""
    from chord.videos import _frames_from_probe

    assert _frames_from_probe(stream) is None


def test_a_plausible_probe_value_is_still_honoured() -> None:
    """The ceiling must not be so tight that a real long video is refused."""
    from chord.videos import MAX_PLAUSIBLE_FRAMES, _frames_from_probe

    assert _frames_from_probe({"nb_frames": str(MAX_PLAUSIBLE_FRAMES)}) == MAX_PLAUSIBLE_FRAMES
    assert _frames_from_probe({"nb_frames": str(MAX_PLAUSIBLE_FRAMES + 1)}) is None
    # An hour at 120 fps is ordinary and must survive.
    assert _frames_from_probe({"nb_frames": str(120 * 3600)}) == 120 * 3600


@pytest.mark.parametrize("stdout", [
    b"[]",                                   # top level is a list, not an object
    b'"a string"',                           # top level is a scalar
    b'{"streams": ["not-a-mapping"]}',       # first stream entry is not an object
    b'{"streams": {}}',                      # streams is a mapping, not a list
    b'{"streams": []}',                      # no streams at all
    b"not json at all",
    b"",
])
def test_an_unreadable_probe_shape_is_a_502_not_a_500(tmp_path, monkeypatch, stdout) -> None:
    """ffprobe's output is an external tool's answer about a caller-supplied
    container. Every shape assumption is now checked and the handler is wide,
    because the contract is "could not read this container" -> None -> 502.
    Narrow typing let `.get` raise AttributeError on a top-level list and
    int() raise OverflowError on an infinite duration (Copilot review, #329)."""
    from chord import videos as videos_mod

    fake = tmp_path / "video.mp4"
    fake.write_bytes(b"not really a video")

    class _Proc:
        returncode = 0
        stderr = b""

        def __init__(self, out):
            self.stdout = out

    monkeypatch.setattr(videos_mod.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(videos_mod.subprocess, "run",
                        lambda *a, **k: _Proc(stdout))
    assert videos_mod._probe_frame_count(fake) is None


def test_the_spritesheet_route_turns_an_unreadable_probe_into_a_502(tmp_path, monkeypatch) -> None:
    """End to end: when the probe cannot be read the content route answers
    spritesheet_unavailable, never a 500 and never an empty 200."""
    from chord import videos as videos_mod

    monkeypatch.setattr(videos_mod, "_probe_frame_count", lambda path: None)
    with _client(tmp_path) as (client, _backend):
        created = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()
        wait_status(client, created["id"], {"completed"})
        r = client.get(f"/v1/videos/{created['id']}/content?variant=spritesheet", headers=_auth())
        assert r.status_code == 502, r.text
        assert r.json()["error"]["code"] == "spritesheet_unavailable"


@pytest.mark.parametrize("url,why", [
    # The marker must BE a parameter, not merely occur inside one.
    (";base64evil", "must be base64-encoded"),
    (";foo=base64", "must be base64-encoded"),
    (";xbase64x", "must be base64-encoded"),
    # Whitespace in the subtype is not a valid MIME token, and .strip() used to
    # launder it into one.
    ("subtype-spaces", "not a valid MIME token"),
    ("subtype-tab", "not a valid MIME token"),
])
def test_a_malformed_data_uri_header_is_refused_not_decoded(tmp_path, url, why) -> None:
    """Two header-parsing bugs the substring check and .strip() hid: `;base64evil`
    satisfied `";base64" in header`, and `data:image/ png ;base64,` was stripped
    into a valid token. Both let a reference the caller never validly declared
    reach the renderer (Copilot review, #329)."""
    import base64 as _b64

    from chord.videos import _decode_data_image

    payload = _b64.b64encode(_png(8, 8)).decode()
    if url.startswith(";"):
        candidate = f"data:image/png{url},{payload}"
    elif url == "subtype-spaces":
        candidate = f"data:image/ png ;base64,{payload}"
    else:
        candidate = f"data:image/\tpng;base64,{payload}"
    result = _decode_data_image(candidate)
    assert isinstance(result, str), f"{candidate!r} was accepted"
    assert why in result


@pytest.mark.parametrize("header", [
    "data:image/png;base64",
    "data:image/png;BASE64",                       # the marker is case-insensitive
    "data:image/png;charset=binary;base64",         # a real parameter before it
    "data:image/svg+xml;base64",                    # structured-syntax suffix
    "data:image/jpeg;base64",
])
def test_a_well_formed_data_uri_header_is_still_accepted(header) -> None:
    """Tightening the marker check must not refuse legitimate forms: an
    uppercase marker, a parameter ahead of it, and a structured-syntax suffix
    are all valid RFC 2397 / RFC 6838 shapes."""
    import base64 as _b64

    from chord.videos import _decode_data_image

    png = _png(8, 8)
    result = _decode_data_image(f"{header},{_b64.b64encode(png).decode()}")
    assert not isinstance(result, str), result
    data, name = result
    assert data == png
    assert name.startswith("reference.")
    # No path separator can ever reach the derived filename.
    assert "/" not in name and "\\" not in name


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg and ffprobe to build a real sheet")
def test_a_spritesheet_from_a_clip_whose_frame_count_is_not_a_grid_multiple(tmp_path) -> None:
    """100 frames is not a multiple of 16, which is exactly the case where floor
    division over-selected: step 6 picks 17 frames, `tile=4x4` keeps the first
    16 and discards the 17th, so the sheet stopped at frame 90. With the ceiling
    step the sheet is still a well-formed 4x4 and reaches the tail of the clip."""
    from chord.videos import SPRITESHEET_GRID, SPRITE_W, _probe_frame_count, _spritesheet_png, sprite_step

    mp4 = tmp_path / "hundred.mp4"
    built = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=4:size=320x240:rate=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True, check=False,
    )
    if built.returncode != 0:
        pytest.skip(f"ffmpeg could not build a test video: {built.stderr[:200]!r}")

    frames = _probe_frame_count(mp4)
    assert frames == 100
    step = sprite_step(frames)
    assert step == 7, "ceiling, not floor: floor would give 6 and over-select 17"
    assert len(range(0, frames, step)) <= 16

    png = _spritesheet_png(mp4)
    assert png is not None and png.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(png)) as sheet:
        # A partly-filled grid still tiles to the full 4x4 canvas.
        assert sheet.size == (SPRITE_W * SPRITESHEET_GRID, 120 * SPRITESHEET_GRID)


def test_an_interrupted_video_is_failed_on_startup(tmp_path) -> None:
    from chord.videos import VideoStore
    root = tmp_path / "videos"
    VideoStore(root).put({"id": "video_01hyy", "status": "queued", "expires_at": None})
    reopened = VideoStore(root).get("video_01hyy")
    assert reopened["status"] == "failed"
    assert reopened["error"]["message"] == "video generation was interrupted"


def test_an_oversized_reference_is_rejected_before_the_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("chord.videos.MAX_REFERENCE_BYTES", 8)
    with _client(tmp_path) as (client, backend):
        response = client.post(
            "/v1/videos",
            data={"prompt": "move"},
            files={"input_reference": ("big.png", b"\x89PNG\r\n\x1a\n" + b"x" * 20, "image/png")},
            headers=_auth(),
        )
        assert response.status_code == 413
        assert response.json()["error"]["param"] == "input_reference"
        assert backend.calls == []


def test_reference_must_match_target_size(tmp_path) -> None:
    with _client(tmp_path) as (client, backend):
        response = client.post(
            "/v1/videos",
            data={"prompt": "move", "size": "720x1280"},
            files={"input_reference": ("wrong.png", _png(1280, 720), "image/png")},
            headers=_auth(),
        )

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "input_reference"
        assert backend.calls == []


def test_a_decompression_bomb_reference_is_refused_before_the_backend(tmp_path) -> None:
    # A 66-byte PNG claiming 14000x14000 pixels. Pillow raises
    # DecompressionBombError at Image.open, and that error inherits from
    # Exception alone -- before the fix it escaped the (UnidentifiedImageError,
    # OSError) handler as an unhandled 500 (review 2026-09-22).
    with _client(tmp_path) as (client, backend):
        response = client.post(
            "/v1/videos",
            data={"prompt": "move", "size": "720x1280"},
            files={"input_reference": ("bomb.png", BOMB_PNG, "image/png")},
            headers=_auth(),
        )
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "input_reference"
        assert backend.calls == []


def test_request_validation_happens_before_backend(tmp_path) -> None:
    with _client(tmp_path) as (client, backend):
        cases = [
            ({}, "prompt"),
            ({"prompt": "x", "seconds": 5}, "seconds"),
            ({"prompt": "x", "size": "640x640"}, "size"),
            ({"prompt": "x", "model": "other"}, "model"),
            ({"prompt": "x", "surprise": True}, "surprise"),
        ]
        for body, param in cases:
            response = client.post("/v1/videos", json=body, headers=_auth())
            assert response.status_code == 400
            assert response.json()["error"]["param"] == param
        assert backend.calls == []


def test_backend_failure_is_a_failed_video_without_internal_detail(tmp_path, caplog) -> None:
    with _client(tmp_path) as (client, backend):
        backend.fail = True
        response = client.post("/v1/videos", json={"prompt": "x"}, headers=_auth())

        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        body = wait_status(client, response.json()["id"], {"failed"})
        assert body["error"] == {"code": "generation_failed", "message": "video generation failed",
                                  "misalignment": None}
        assert "private" not in response.text and "private" not in json.dumps(body)
        assert "RuntimeError" in caplog.text
        assert body["id"] in caplog.text


def test_videos_require_service_authentication(tmp_path) -> None:
    with _client(tmp_path) as (client, backend):
        assert client.post("/v1/videos", json={"prompt": "x"}).status_code == 401
        assert backend.calls == []


def test_every_terminal_row_carries_an_expiry(tmp_path) -> None:
    """A failed row must expire too, or it is retained forever.

    `_drop_expired` only acts on an int, and a failed record inherited the queued
    row's `expires_at: None`. Nothing ever removed it: the sqlite index grew by one
    row per failed render and `GET /v1/videos` kept listing them for the life of the
    volume. Both failure paths are covered — the background one and the
    interrupted-at-startup one, which is the row a crash loop produces.
    """
    from chord.videos import RETENTION_S, VideoStore

    with _client(tmp_path) as (client, backend):
        backend.fail = True
        created = client.post("/v1/videos", json={"prompt": "x"}, headers=_auth()).json()
        failed = wait_status(client, created["id"], {"failed"})

        assert isinstance(failed["expires_at"], int)
        assert failed["expires_at"] > int(time.time())
        # 24h, the spec's window — not a shorter local guess. A caller that retrieves
        # the object, then the content, then a thumbnail must not race the expiry.
        assert RETENTION_S == 86_400
        assert failed["expires_at"] - int(time.time()) > 86_000

    root = tmp_path / "interrupted"
    VideoStore(root).put({"id": "video_01hyz", "status": "in_progress", "expires_at": None})
    reopened = VideoStore(root).get("video_01hyz")
    assert reopened["status"] == "failed"
    assert isinstance(reopened["expires_at"], int)

    # The control: with no expiry the row outlives every reader, which is the
    # state this test exists to keep out.
    stale = tmp_path / "stale"
    store = VideoStore(stale)
    store.put({"id": "video_01j000", "status": "failed", "expires_at": None})
    assert store.get("video_01j000") is not None


def test_an_oversized_stored_reference_is_refused_before_the_backend(tmp_path, monkeypatch) -> None:
    """The multipart branch bounds its read; the file_id branch used to load
    whatever was stored -- up to 512 MB -- whole into memory before anything
    measured it (review 2026-09-22). Same cap, same 413, same param."""
    monkeypatch.setattr("chord.videos.MAX_REFERENCE_BYTES", 8)
    with _client(tmp_path) as (client, backend):
        headers = _auth()
        uploaded = client.post("/v1/files", data={"purpose": "vision"},
                               files={"file": ("reference.png", _png(720, 1280), "image/png")},
                               headers=headers)
        assert uploaded.status_code == 200, uploaded.text
        response = client.post(
            "/v1/videos",
            json={"prompt": "the figure turns", "size": "720x1280",
                  "input_reference": {"file_id": uploaded.json()["id"]}},
            headers=headers,
        )
        assert response.status_code == 413, response.text
        assert response.json()["error"]["param"] == "input_reference"
        assert response.json()["error"]["code"] == "file_too_large"
        assert backend.calls == []


def test_admission_is_bounded_by_outstanding_jobs(tmp_path, monkeypatch) -> None:
    """Renders serialise on one GPU and every queued job pins its reference in
    a task closure, so an unbounded admission queue let any authenticated
    caller hold the GPU for days (review 2026-09-22). Past the bound the door
    answers the API 429 envelope; a terminal row frees its slot."""
    from chord.videos import VideoStore

    monkeypatch.setattr("chord.videos.MAX_OUTSTANDING", 2)
    with _client(tmp_path) as (client, backend):
        seed = VideoStore(tmp_path / "videos")    # route store already constructed, empty

        def queued(i: int) -> dict:
            return {"id": f"video_seed{i}", "object": "video", "model": "sora-2",
                    "status": "queued", "progress": 0, "created_at": int(time.time()),
                    "completed_at": None, "expires_at": None, "error": None, "prompt": "seed",
                    "remixed_from_video_id": None, "seconds": "4", "size": "720x1280"}

        seed.put(queued(0))
        seed.put(queued(1))
        refused = client.post("/v1/videos", json={"prompt": "one more"}, headers=_auth())
        assert refused.status_code == 429, refused.text
        assert refused.json()["error"]["code"] == "rate_limit_exceeded"
        assert refused.json()["error"]["type"] == "rate_limit_error"
        assert backend.calls == []

        seed.put({**queued(0), "status": "completed", "progress": 100,
                  "completed_at": int(time.time()), "expires_at": int(time.time()) + 86_400})
        accepted = client.post("/v1/videos", json={"prompt": "one more"}, headers=_auth())
        assert accepted.status_code == 200, accepted.text
        wait_status(client, accepted.json()["id"], {"completed"})


def test_delete_during_a_render_cancels_the_work_and_the_row_stays_deleted(tmp_path) -> None:
    """DELETE removed the row but not the work: the render kept running, and
    its terminal unconditional put brought the video back as `completed` --
    with a fresh MP4 on the volume -- after the caller deleted it (review
    2026-09-22, #1, reproduced in production semantics). This test is
    green-only by construction: the old BackgroundTasks harness ran the render
    INSIDE the create request, so a concurrent delete was not even expressible
    in a test -- which is why the bug only ever showed live."""
    backend = SlowBackend()
    with _client(tmp_path, backend) as (client, _):
        created = client.post("/v1/videos", json={"prompt": "waves"}, headers=_auth()).json()
        vid = created["id"]
        wait_status(client, vid, {"in_progress"})       # the render is live, holding its event

        removed = client.delete(f"/v1/videos/{vid}", headers=_auth())
        assert removed.status_code == 200
        assert removed.json()["deleted"] is True

        backend.release.set()                           # a straggler backend returns
        time.sleep(0.3)                                 # any resurrection write gets its chance
        assert client.get(f"/v1/videos/{vid}", headers=_auth()).status_code == 404
        assert not (tmp_path / "videos" / f"{vid}.mp4").exists()


def test_delete_frees_the_admission_slot_even_mid_render(tmp_path, monkeypatch) -> None:
    """The create-4/delete-4 loop that stacked orphaned GPU renders past
    MAX_OUTSTANDING (review, #1): admission counted rows while the lock
    queued tasks. With the delete cancelling the task, the slot frees at
    delete and the orphaned render never reaches the GPU."""
    monkeypatch.setattr("chord.videos.MAX_OUTSTANDING", 1)
    backend = SlowBackend()
    with _client(tmp_path, backend) as (client, _):
        first = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()["id"]
        wait_status(client, first, {"in_progress"})

        assert client.post("/v1/videos", json={"prompt": "two"}, headers=_auth()).status_code == 429

        assert client.delete(f"/v1/videos/{first}", headers=_auth()).status_code == 200
        second = client.post("/v1/videos", json={"prompt": "two"}, headers=_auth())
        assert second.status_code == 200, second.text

        backend.release.set()                           # the second render finishes
        wait_status(client, second.json()["id"], {"completed"})


def test_a_list_where_a_string_belongs_is_a_named_400(tmp_path) -> None:
    """`model not in MODELS` hashes, and a list is unhashable: the caller's
    malformed body used to arrive as a 500 TypeError (review 2026-09-22,
    #5, reproduced)."""
    with _client(tmp_path) as (client, backend):
        for body, param in [
            ({"prompt": "x", "model": ["sora-2"]}, "model"),
            ({"prompt": "x", "size": ["720x1280"]}, "size"),
        ]:
            r = client.post("/v1/videos", json=body, headers=_auth())
            assert r.status_code == 400, body
            assert r.json()["error"]["param"] == param
            assert r.json()["error"]["code"] == "invalid_value"
        assert backend.calls == []


def test_a_job_waiting_for_the_gpu_stays_queued(tmp_path) -> None:
    """`in_progress` used to be stamped before the lock was acquired: a job
    waiting behind another render claimed a GPU it did not have, and `queued`
    was a status no caller could ever observe (batch 4). The row flips only
    when the backend reports the GPU is actually ours."""
    backend = SlowBackend()
    with _client(tmp_path, backend) as (client, _):
        first = client.post("/v1/videos", json={"prompt": "one"}, headers=_auth()).json()["id"]
        wait_status(client, first, {"in_progress"})
        second = client.post("/v1/videos", json={"prompt": "two"}, headers=_auth()).json()["id"]
        time.sleep(0.3)
        body = client.get(f"/v1/videos/{second}", headers=_auth()).json()
        assert body["status"] == "queued", f"a job waiting for the GPU claimed it: {body['status']}"

        backend.release.set()                        # the first render finishes
        wait_status(client, first, {"completed"})
        wait_status(client, second, {"in_progress"}) # the second holds the GPU now
        backend.release.set()
        wait_status(client, second, {"completed"})


# --- input_reference via image_url (#321) --------------------------------------


def _data_url(png: bytes, subtype: str = "png") -> str:
    return f"data:image/{subtype};base64," + base64.b64encode(png).decode()


def test_a_data_uri_reference_selects_r2v(tmp_path) -> None:
    """The spec's ImageRefParam accepts `image_url` as "a fully qualified URL or
    base64-encoded data URL". A data URI needs no network and no allowlist
    entry, so it is the default-allowed form -- the same posture chat's
    image_url parts have (data: by default, http(s) only by exact-host opt-in)."""
    with _client(tmp_path) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "the figure turns", "size": "720x1280",
            "input_reference": {"image_url": _data_url(_png(720, 1280))},
        }, headers=_auth())
        assert r.status_code == 200, r.text
        wait_status(client, r.json()["id"], {"completed"})
        kind, params = backend.calls[0]
        assert kind == "r2v"
        assert params["reference"].startswith(b"\x89PNG")


@pytest.mark.parametrize("url", [
    "https://example.com/a.png",              # host not on the allowlist
    "http://169.254.169.254/latest/meta-data", # cloud metadata, never allowlisted
    "file:///etc/passwd",                      # a local file is not a reference
    "ftp://example.com/a.png",                 # scheme the backend never names
    "gopher://example.com/a.png",
])
def test_a_reference_url_outside_the_policy_is_refused(tmp_path, url) -> None:
    """Every non-allowlisted and non-data scheme is refused BEFORE any fetch,
    and the refusal never enumerates the allowlist (chat's image_url policy,
    reused rather than restated so the two doors cannot drift)."""
    with _client(tmp_path) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "x", "input_reference": {"image_url": url},
        }, headers=_auth())
        assert r.status_code == 400, (url, r.text)
        assert r.json()["error"]["param"] == "image_url"
        assert backend.calls == [], "a refused reference must never reach the renderer"


@pytest.mark.parametrize("url", [
    "https://evil\\@allowed.example/a.png",   # backslash: urllib3 and urlparse disagree
    "https://evil%40allowed.example/a.png",   # percent-encoded @
    "https://allowed.example:notaport/a.png",  # malformed port
])
def test_parser_divergence_bypasses_are_refused(tmp_path, url) -> None:
    """A probe exposed the userinfo/backslash/percent class on chat's door:
    two URL parsers reading one authority as two different hosts. Refused
    regardless of allowlist tier, because neither form has a legitimate use."""
    with _client(tmp_path) as (client, _backend):
        r = client.post("/v1/videos", json={
            "prompt": "x",
            "input_reference": {"image_url": url},
        }, headers=_auth())
        assert r.status_code == 400, (url, r.text)
        assert r.json()["error"]["param"] == "image_url"


@pytest.mark.parametrize("url,why", [
    ("data:image/png;base64,not!valid!base64", "not valid base64"),
    ("data:image/png;base64,", "empty"),
    ("data:image/png,AAAA", "must be base64-encoded"),
    ("data:image/png", "malformed"),
    ("data:text/plain;base64,AAAA", "data:image"),
])
def test_a_malformed_data_uri_is_refused_by_name(tmp_path, url, why) -> None:
    with _client(tmp_path) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "x", "input_reference": {"image_url": url},
        }, headers=_auth())
        assert r.status_code == 400, (url, r.text)
        assert r.json()["error"]["param"] == "image_url"
        assert why.lower() in r.json()["error"]["message"].lower(), r.json()["error"]["message"]
        assert backend.calls == []


@pytest.mark.parametrize("subtype", [
    "../../etc/passwd",        # path traversal into the derived filename
    "png/../../evil",          # a slash in the subtype
    'png"; DROP TABLE x; --',  # quotes and a statement separator
    "png\\evil",               # backslash
    "png\nevil",               # a control character / header injection shape
    "png evil",                # whitespace
    "",                        # empty
])
def test_a_data_uri_subtype_cannot_shape_the_reference_filename(tmp_path, subtype) -> None:
    """The subtype becomes part of the filename handed to ComfyUI's upload, so
    it is validated as a MIME token. Without that, `data:image/../../evil;base64,...`
    puts a path-shaped string into a filename -- the multipart branch sanitizes
    with os.path.basename for exactly this reason, and a data URI has no
    basename to take (Copilot review, #329)."""
    payload = base64.b64encode(_png(720, 1280)).decode()
    url = f"data:image/{subtype};base64,{payload}"
    with _client(tmp_path) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "x", "size": "720x1280",
            "input_reference": {"image_url": url},
        }, headers=_auth())
        assert r.status_code == 400, (subtype, r.text)
        assert r.json()["error"]["param"] == "image_url"
        assert backend.calls == [], "a refused subtype must never reach the renderer"
        # And nothing path-shaped can have leaked into a filename.
        for _kind, params in backend.calls:
            assert "/" not in str(params.get("reference_name", ""))


def test_a_legitimate_structured_syntax_subtype_is_accepted(tmp_path) -> None:
    """`svg+xml` is a valid RFC 6838 structured-syntax suffix, so the token rule
    must not be so strict that it refuses real subtypes. The derived filename
    drops the suffix (reference.svg), which is what the split is for."""
    from chord.videos import _decode_data_image

    decoded = _decode_data_image(_data_url(_png(8, 8), subtype="svg+xml"))
    assert not isinstance(decoded, str), decoded
    _data, name = decoded
    assert name == "reference.svg"


@pytest.mark.parametrize("url", [
    "http://[::1",                    # unterminated IPv6 literal: urlparse RAISES
    "http://[::1/a.png",
    "https://[2001:db8::1",
])
def test_an_unparseable_reference_url_is_a_named_400_not_a_500(tmp_path, url) -> None:
    """The shared policy calls urlparse, which raises ValueError on a malformed
    authority rather than returning a refusal. Unguarded that is a 500 for input
    that is simply a bad URL; the API rule is malformed input is a named 400
    (Copilot review, #329)."""
    with _client(tmp_path) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "x", "input_reference": {"image_url": url},
        }, headers=_auth())
        assert r.status_code == 400, (url, r.text)
        assert r.json()["error"]["param"] == "image_url"
        assert backend.calls == []


def test_a_data_uri_of_the_wrong_size_is_refused_by_the_dimension_check(tmp_path) -> None:
    """The reference bytes go through the same size validation whichever form
    the caller used, so a data URI cannot smuggle past the multipart check."""
    with _client(tmp_path) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "x", "size": "720x1280",
            "input_reference": {"image_url": _data_url(_png(8, 8))},
        }, headers=_auth())
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "input_reference"
        assert backend.calls == []


def test_image_url_and_file_id_together_are_refused(tmp_path) -> None:
    """The spec says provide EXACTLY one of image_url or file_id. Both named is
    ambiguous, and picking one silently would be the caller's intent guessed."""
    with _client(tmp_path) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "x",
            "input_reference": {"image_url": _data_url(_png(720, 1280)), "file_id": "file-1"},
        }, headers=_auth())
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "input_reference"
        assert backend.calls == []


@contextlib.contextmanager
def _local_image_server(png: bytes, redirect_to: str | None = None):
    """A real HTTP server on 127.0.0.1 so the allowlisted-fetch path is
    exercised over a real socket, not a stub. Serves `png`, or 302s to
    `redirect_to` when asked -- which is how the no-redirect rule is proved."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if redirect_to is not None:
                self.send_response(302)
                self.send_header("Location", redirect_to)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_an_allowlisted_host_is_fetched_and_selects_r2v(tmp_path) -> None:
    """The opt-in half of the policy: an exact host[:port] on
    IMAGE_URL_ALLOWED_HOSTS is fetched for real and rendered."""
    png = _png(720, 1280)
    with _local_image_server(png) as netloc:
        with _client(tmp_path, image_url_allowed_hosts=frozenset({netloc})) as (client, backend):
            r = client.post("/v1/videos", json={
                "prompt": "the figure turns", "size": "720x1280",
                "input_reference": {"image_url": f"http://{netloc}/reference.png"},
            }, headers=_auth())
            assert r.status_code == 200, r.text
            wait_status(client, r.json()["id"], {"completed"})
            kind, params = backend.calls[0]
            assert kind == "r2v"
            assert params["reference"] == png


def test_a_redirect_from_an_allowlisted_host_is_refused_not_followed(tmp_path) -> None:
    """The load-bearing half of the fetch: the allowlist names an exact host, so
    a 3xx from that host to an internal address is the bypass the allowlist
    exists to prevent. Redirects are OFF, so a redirect is a refusal."""
    png = _png(720, 1280)
    with _local_image_server(png, redirect_to="http://169.254.169.254/latest/meta-data") as netloc:
        with _client(tmp_path, image_url_allowed_hosts=frozenset({netloc})) as (client, backend):
            r = client.post("/v1/videos", json={
                "prompt": "x", "size": "720x1280",
                "input_reference": {"image_url": f"http://{netloc}/reference.png"},
            }, headers=_auth())
            assert r.status_code == 400, r.text
            assert r.json()["error"]["param"] == "image_url"
            assert backend.calls == [], "a redirected reference must never reach the renderer"


def test_an_unreachable_allowlisted_host_is_a_400_not_a_500(tmp_path) -> None:
    """A host on the allowlist that does not answer is the caller's URL, not our
    outage: a named 400, and never the transport's own detail on the wire."""
    with _client(tmp_path, image_url_allowed_hosts=frozenset({"127.0.0.1:1"})) as (client, backend):
        r = client.post("/v1/videos", json={
            "prompt": "x", "input_reference": {"image_url": "http://127.0.0.1:1/a.png"},
        }, headers=_auth())
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "image_url"
        assert "could not be fetched" in r.json()["error"]["message"]
        assert backend.calls == []


class SubmittingBackend(VideoBackend):
    """Reports a ComfyUI prompt id the way the real backend does once it has
    submitted, then renders until the process goes away."""

    def __init__(self, prompt_id: str = "prompt-orphan") -> None:
        super().__init__()
        self.prompt_id = prompt_id
        self.cancelled: list[str] = []

    async def render(self, kind: str, params: dict, on_start=None, on_submit=None,
                     on_settled=None) -> tuple[bytes, str]:
        self.calls.append((kind, params))
        if on_start is not None:
            on_start()
        if on_submit is not None:
            on_submit(self.prompt_id)
        while True:                      # a render that never finishes in this process
            await asyncio.sleep(0.01)

    async def cancel_orphan(self, prompt_id: str) -> bool:
        self.cancelled.append(prompt_id)
        return True


def test_a_restart_cancels_the_comfy_prompt_an_interrupted_video_left_behind(tmp_path) -> None:
    """Review 2026-09-24 B20: a restart marked in-flight rows failed but never
    told ComfyUI, so the orphaned prompt kept the GPU (shared with other jobs).
    The submitted prompt id is kept on the row, never in the public
    object, and startup recovery cancels it through the backend's safe
    cancel."""
    first = SubmittingBackend()
    with _client(tmp_path, first) as (client, _):
        created = client.post("/v1/videos", json={"prompt": "waves"}, headers=_auth()).json()
        body = wait_status(client, created["id"], {"in_progress"})
        assert "prompt-orphan" not in json.dumps(body), "the ComfyUI prompt id leaked into the video object"
        end = time.monotonic() + 5
        while time.monotonic() < end:
            import sqlite3
            db = sqlite3.connect(tmp_path / "videos" / "index.sqlite3")
            try:
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                stored = (db.execute("SELECT prompt_id FROM comfy_prompts WHERE video_id = ?",
                                     (created["id"],)).fetchone()
                          if "comfy_prompts" in tables else None)
            finally:
                db.close()
            if stored and stored[0] == "prompt-orphan":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("the submitted prompt id was never recorded")

    second = SubmittingBackend()
    with _client(tmp_path, second) as (client, _):
        end = time.monotonic() + 5
        while time.monotonic() < end and not second.cancelled:
            time.sleep(0.02)
        assert second.cancelled == ["prompt-orphan"]
        reopened = client.get(f"/v1/videos/{created['id']}", headers=_auth()).json()
        assert reopened["status"] == "failed"
    assert first.cancelled == []


def test_a_failing_orphan_cancel_never_blocks_startup(tmp_path) -> None:
    """Review 2026-09-24 B20: the recovery cancel is best effort. A ComfyUI
    that is down at startup is logged, and the service still serves."""
    from chord.videos import VideoStore

    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_01orphan", "object": "video", "model": "sora-2",
               "status": "in_progress", "progress": 0, "created_at": int(time.time()),
               "completed_at": None, "expires_at": None, "error": None, "prompt": "x",
               "remixed_from_video_id": None, "seconds": "4", "size": "720x1280"})
    store.set_prompt_id("video_01orphan", "prompt-dead")

    class DownBackend(VideoBackend):
        async def cancel_orphan(self, prompt_id: str) -> None:
            raise RuntimeError("comfy is down")

    with _client(tmp_path, DownBackend()) as (client, _):
        assert client.get("/v1/videos/video_01orphan", headers=_auth()).json()["status"] == "failed"


# --- review 2026-09-24 B22 ----------------------------------------------------


def test_a_delete_that_races_the_mp4_write_leaves_no_file(tmp_path, monkeypatch) -> None:
    """The MP4 write runs in a worker thread, and cancelling the task does not
    stop the thread: DELETE cancelled, removed the row and unlinked a file that
    was not there yet, and then the thread landed the bytes -- an orphan MP4
    beside no row, which no reader and no expiry would ever remove (review
    2026-09-24 B22). The write is slowed so the delete lands inside it."""
    import pathlib
    import threading

    real_write = pathlib.Path.write_bytes
    writing = threading.Event()
    written = threading.Event()

    def slow_write(self, data):
        if self.suffix != ".mp4":
            return real_write(self, data)
        writing.set()
        time.sleep(0.4)
        try:
            return real_write(self, data)
        finally:
            written.set()

    monkeypatch.setattr(pathlib.Path, "write_bytes", slow_write)
    with _client(tmp_path) as (client, _backend):
        vid = client.post("/v1/videos", json={"prompt": "waves"}, headers=_auth()).json()["id"]
        assert writing.wait(5), "the MP4 write never started"

        removed = client.delete(f"/v1/videos/{vid}", headers=_auth())
        assert removed.status_code == 200, removed.text

        assert written.wait(5), "the MP4 write never finished"
        time.sleep(0.1)                                  # any post-write cleanup gets its turn
        assert client.get(f"/v1/videos/{vid}", headers=_auth()).status_code == 404
        assert not (tmp_path / "videos" / f"{vid}.mp4").exists(), "orphan MP4 left beside no row"


def test_the_daily_sweep_drops_expired_videos_and_their_files(tmp_path) -> None:
    """Expired MP4s were removed only when someone READ the store: a volume no
    caller listed kept every MP4 forever, and the daily sweep never looked in
    data_dir/videos (review 2026-09-24 B22). The sweep now expires video rows
    through the store's own path -- and must not mark a live render failed,
    which is what constructing a VideoStore at startup does."""
    import sqlite3

    from chord import retention
    from chord.videos import VideoStore

    root = tmp_path / "videos"
    store = VideoStore(root)
    now = int(time.time())
    store.put({"id": "video_old", "status": "completed", "expires_at": now - 10})
    store.put({"id": "video_new", "status": "completed", "expires_at": now + 3600})
    store.put({"id": "video_live", "status": "in_progress", "expires_at": None})
    for vid in ("video_old", "video_new"):
        (root / f"{vid}.mp4").write_bytes(MP4)

    removed = retention.sweep(tmp_path, 30)

    assert removed["videos"] == 1
    assert not (root / "video_old.mp4").exists()
    assert (root / "video_new.mp4").exists()
    raw = sqlite3.connect(root / "index.sqlite3")
    rows = {vid: json.loads(record)["status"] for vid, record in raw.execute("SELECT id, record FROM videos")}
    raw.close()
    assert rows == {"video_new": "completed", "video_live": "in_progress"}

    # The spec's 24 h is not the volume window: CHORD_RETENTION_DAYS=0 keeps
    # traces and artifacts, but an expired video is gone for every reader
    # already, so the sweep drops it too.
    store.put({"id": "video_old2", "status": "failed", "expires_at": now - 10})
    assert retention.sweep(tmp_path, 0)["videos"] == 1


def test_a_refused_request_never_fetches_its_reference_url(tmp_path, monkeypatch) -> None:
    """The input_reference image_url fetch ran inside body parsing, BEFORE the
    prompt check and the 429 admission check: a request that was going to be
    refused still made an outbound fetch of up to 25 MB (review 2026-09-24 B22).
    The cheap checks run first now."""
    from chord.videos import VideoStore

    fetched: list[str] = []

    async def recording_fetch(url):
        fetched.append(url)
        return _png(720, 1280), "reference.png"

    monkeypatch.setattr("chord.videos._fetch_reference", recording_fetch)
    monkeypatch.setattr("chord.videos.MAX_OUTSTANDING", 1)
    netloc = "images.example.test:8443"
    ref = {"image_url": f"https://{netloc}/reference.png"}
    with _client(tmp_path, image_url_allowed_hosts=frozenset({netloc})) as (client, backend):
        bad = client.post("/v1/videos", json={"prompt": "", "input_reference": ref}, headers=_auth())
        assert bad.status_code == 400, bad.text
        assert bad.json()["error"]["param"] == "prompt"
        assert fetched == [], "a 400 request fetched its reference"

        VideoStore(tmp_path / "videos").put({"id": "video_seed0", "status": "queued", "expires_at": None})
        full = client.post("/v1/videos", json={"prompt": "x", "input_reference": ref}, headers=_auth())
        assert full.status_code == 429, full.text
        assert fetched == [], "a 429 request fetched its reference"
        assert backend.calls == []


def test_a_delete_cancelled_while_the_render_drains_still_deletes_the_row(tmp_path, monkeypatch) -> None:
    """DELETE cancels the render, then awaits it draining. If the DELETE itself
    was cancelled in that await (the client left), it re-raised without ever
    removing the row: the render had already been cancelled, so the row sat
    `in_progress` forever, pinning an admission slot (Copilot on #334, review
    2026-09-24 B22). The drain is a slow MP4 write so the cancel lands inside it."""
    import pathlib
    import threading

    from chord.videos import VideoStore

    real_write = pathlib.Path.write_bytes
    writing = threading.Event()
    written = threading.Event()

    def slow_write(self, data):
        if self.suffix != ".mp4":
            return real_write(self, data)
        writing.set()
        time.sleep(0.4)
        try:
            return real_write(self, data)
        finally:
            written.set()

    monkeypatch.setattr(pathlib.Path, "write_bytes", slow_write)
    with _client(tmp_path) as (client, _backend):
        vid = client.post("/v1/videos", json={"prompt": "waves"}, headers=_auth()).json()["id"]
        assert writing.wait(5), "the MP4 write never started"
        route = next(r for r in client.app.routes
                     if getattr(r, "path", "") == "/v1/videos/{video_id}" and "DELETE" in r.methods)

        async def delete_then_leave() -> None:
            deleting = asyncio.ensure_future(route.endpoint(vid))
            await asyncio.sleep(0.1)                     # inside the drain await
            deleting.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await deleting

        client.portal.call(delete_then_leave)
        assert written.wait(5), "the MP4 write never finished"
        time.sleep(0.2)                                  # cleanup after the drain gets its turn
        assert client.get(f"/v1/videos/{vid}", headers=_auth()).status_code == 404
        assert not (tmp_path / "videos" / f"{vid}.mp4").exists()
        assert VideoStore(tmp_path / "videos", recover_interrupted=False).count_outstanding() == 0


def test_an_mp4_whose_unlink_failed_is_removed_by_the_next_sweep(tmp_path, monkeypatch) -> None:
    """The expiry deleted the row and then unlinked the MP4: one OSError on the
    unlink left a file no later sweep could find, because the row that named it
    was already gone (Copilot on #334, review 2026-09-24 B22). The next sweep
    must remove it."""
    import pathlib

    from chord import retention
    from chord.videos import VideoStore

    root = tmp_path / "videos"
    store = VideoStore(root)
    store.put({"id": "video_old", "status": "completed", "expires_at": int(time.time()) - 10})
    (root / "video_old.mp4").write_bytes(MP4)

    real_unlink = pathlib.Path.unlink
    failures = {"left": 1}

    def flaky_unlink(self, missing_ok=False):
        if self.name == "video_old.mp4" and failures["left"]:
            failures["left"] -= 1
            raise PermissionError("busy")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", flaky_unlink)
    retention.sweep(tmp_path, 30)
    assert failures["left"] == 0, "the first sweep never tried the unlink"
    retention.sweep(tmp_path, 30)
    assert not (root / "video_old.mp4").exists(), "orphan MP4 survived the retry sweep"


def test_an_orphan_cancel_that_did_not_settle_is_retried_at_the_next_start(tmp_path, monkeypatch) -> None:
    """Copilot on #336: a startup cancel that could not reach ComfyUI was
    logged and forgotten -- the row was already `failed`, so no later start
    looked at it again, and the prompt could hold the shared GPU forever. The
    prompt id is now the marker: it stays on the row until a cancel settles."""
    from chord.videos import VideoStore

    monkeypatch.setattr("chord.videos.ORPHAN_RETRY_DELAYS", ())
    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_01orphan", "object": "video", "model": "sora-2",
               "status": "in_progress", "progress": 0, "created_at": int(time.time()),
               "completed_at": None, "expires_at": None, "error": None, "prompt": "x",
               "remixed_from_video_id": None, "seconds": "4", "size": "720x1280"})
    store.set_prompt_id("video_01orphan", "prompt-kept")

    class Backend(VideoBackend):
        def __init__(self, settles: bool) -> None:
            super().__init__()
            self.settles = settles
            self.cancelled: list[str] = []

        async def cancel_orphan(self, prompt_id: str) -> bool:
            self.cancelled.append(prompt_id)
            return self.settles

    for settles, expected in ((False, ["prompt-kept"]), (True, ["prompt-kept"]), (True, [])):
        backend = Backend(settles)
        with _client(tmp_path, backend) as (client, _):
            end = time.monotonic() + 5
            while time.monotonic() < end and backend.cancelled != expected:
                time.sleep(0.02)
            time.sleep(0.05)                     # the clear lands after the cancel returns
            assert backend.cancelled == expected
            assert client.get("/v1/videos/video_01orphan", headers=_auth()).json()["status"] == "failed"


def test_the_admission_count_is_answered_by_an_index(tmp_path) -> None:
    """Copilot on #336: every image edit and variation now asks the video
    store how deep the queue is, and that read loaded and JSON-parsed every
    row the day retains. The count is SQLite's, over an index on status."""
    import sqlite3

    from chord.videos import VideoStore

    store = VideoStore(tmp_path / "videos")
    store.put({"id": "video_q", "status": "queued", "expires_at": None})
    store.put({"id": "video_r", "status": "in_progress", "expires_at": None})
    store.put({"id": "video_c", "status": "completed", "expires_at": int(time.time()) + 60})
    assert store.count_outstanding() == 2
    db = sqlite3.connect(tmp_path / "videos" / "index.sqlite3")
    try:
        plan = " ".join(str(r) for r in db.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM videos "
            "WHERE json_extract(record, '$.status') IN ('queued', 'in_progress')"))
    finally:
        db.close()
    assert "videos_status" in plan, plan


def _outstanding(store, vid: str) -> None:
    store.put({"id": vid, "object": "video", "model": "sora-2", "status": "in_progress",
               "progress": 0, "created_at": int(time.time()), "completed_at": None,
               "expires_at": None, "error": None, "prompt": "x",
               "remixed_from_video_id": None, "seconds": "4", "size": "720x1280"})


def test_deleting_a_recovered_row_keeps_its_prompt_for_the_next_start(tmp_path) -> None:
    """Copilot on #336: a recovered row whose cancel had not settled could be
    DELETEd -- nothing in `renders` to cancel -- and the delete took the retry
    marker with it, so the prompt had no record anywhere. Submitted prompts
    are kept apart from the rows, and only a settled stop removes one."""
    from chord.videos import VideoStore

    store = VideoStore(tmp_path / "videos")
    _outstanding(store, "video_01gone")
    store.set_prompt_id("video_01gone", "prompt-gone")
    assert store.delete("video_01gone")
    assert VideoStore(tmp_path / "videos").orphaned_prompts == ["prompt-gone"]


def test_expiring_a_failed_row_keeps_its_prompt_for_the_next_start(tmp_path) -> None:
    """Copilot on #336: the failed row carrying the retry marker expired on the
    normal 24 h clock, and the sweep deleted it with the marker, so a ComfyUI
    down for longer than retention left a prompt nothing would ever retry."""
    from chord import retention
    from chord.videos import VideoStore

    root = tmp_path / "videos"
    store = VideoStore(root)
    store.put({"id": "video_01old", "status": "failed", "expires_at": int(time.time()) - 10})
    store.set_prompt_id("video_01old", "prompt-old")
    assert retention.sweep(tmp_path, 30)["videos"] == 1
    assert VideoStore(root).orphaned_prompts == ["prompt-old"]
