"""The OpenAI Images API door: pixels or an honest error,
never a success without an image."""
import base64
import json
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from chord import artifact_links, specialists
from chord.config import ImageWorkflowConfig, Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app, load_specialists

from test_skeleton import PNG, FakeUpstream

load_specialists()  # so a test's stand-in isn't overwritten by the first Deps()
URL = "/v1/images/generations"


def client_for(tmp_path, monkeypatch, specialist, routes=frozenset({"image"})):
    if specialist is not None:
        monkeypatch.setitem(specialists.SPECIALISTS, "image", specialist)
    up = FakeUpstream()
    settings = Settings(data_dir=tmp_path, experimental_routes=routes)
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: None))), up, settings


def last_trace(settings):
    lines = [line for p in sorted(Path(settings.trace_dir).glob("*.jsonl")) for line in p.read_text().splitlines()]
    return json.loads(lines[-1])


def rendered(summary="a mug"):
    async def run(job, ctx):
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d],
                      summary=summary, provenance={"prompt": "a glazed mug, dawn light"})
    return run


def configured_comfy_client(tmp_path, monkeypatch, png: bytes, **settings_over):
    workflow = tmp_path / "still.json"
    workflow.write_text(json.dumps({
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }))
    settings = Settings(
        data_dir=tmp_path, image_comfy_base_url="http://localhost:8188",
        image_workflow=ImageWorkflowConfig(workflow, tmp_path, "6", "9"),
        experimental_routes=frozenset(),
        **settings_over,
    )
    calls = []

    class FakeImageBackend:
        async def render(self, prompt):
            calls.append(prompt)
            return png, "comfy-prompt-1"

        async def aclose(self):
            pass

    async def bypass_must_not_run(job, ctx):
        raise AssertionError("the Images API must use its configured provider")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", bypass_must_not_run)
    deps = Deps(settings, upstream=FakeUpstream(), model=lambda n: None,
                image_backend=FakeImageBackend())
    return TestClient(create_app(deps)), calls, settings


@pytest.mark.parametrize("side", [512, 1024])
def test_configured_comfy_generations_use_prompt_only_and_preserve_delivery(tmp_path, monkeypatch, side):
    output = BytesIO()
    metadata = PngInfo()
    metadata.add_text("workflow", "private-checkpoint-and-graph")
    Image.new("RGB", (1024, 1024), "blue").save(output, format="PNG", pnginfo=metadata)
    client, calls, settings = configured_comfy_client(tmp_path, monkeypatch, output.getvalue())
    response = client.post(URL, json={"model": "chord-1-poly", "prompt": "a blue mug",
                                      "size": f"{side}x{side}", "response_format": "b64_json"})
    assert response.status_code == 200, response.text
    assert calls == ["a blue mug"]
    delivered = base64.b64decode(response.json()["data"][0]["b64_json"])
    assert b"private-checkpoint-and-graph" not in delivered
    with Image.open(BytesIO(delivered)) as image:
        assert image.size == (side, side)
    trace = last_trace(settings)
    assert trace["image_backend"] == "comfyui"
    assert trace["comfy_prompt_id"] == "comfy-prompt-1"
    assert trace["result_status"] == "completed"
    assert trace["image_metadata_removed"] == ["tEXt"]


def test_unconfigured_generations_refuse_without_rendering(tmp_path, monkeypatch):
    client, _, _ = client_for(tmp_path, monkeypatch, rendered())
    response = client.post(URL, json={"prompt": "a mug"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "capability_unavailable"


def test_bracket_marker_is_plain_prompt_text(tmp_path, monkeypatch):
    output = BytesIO()
    Image.new("RGB", (1024, 1024), "blue").save(output, format="PNG")
    client, calls, _ = configured_comfy_client(tmp_path, monkeypatch, output.getvalue())
    response = client.post(URL, json={"prompt": "a sign reading [[example:abc]]"})
    assert response.status_code == 200
    assert calls == ["a sign reading [[example:abc]]"]


def test_configured_image_refuses_a_lone_surrogate_before_submitting(tmp_path, monkeypatch):
    output = BytesIO()
    Image.new("RGB", (1024, 1024), "blue").save(output, format="PNG")
    client, calls, _ = configured_comfy_client(tmp_path, monkeypatch, output.getvalue())
    response = client.post(URL, content=json.dumps({"prompt": "a mug \ud800"}),
                           headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_prompt"
    assert calls == []


def test_configured_generations_use_default_model_and_require_key(tmp_path, monkeypatch):
    output = BytesIO()
    Image.new("RGB", (1024, 1024), "blue").save(output, format="PNG")
    client, calls, _ = configured_comfy_client(
        tmp_path, monkeypatch, output.getvalue(), service_api_key="k")
    assert client.post(URL, json={"prompt": "a mug"}).status_code == 401
    response = client.post(URL, json={"prompt": "a mug"},
                           headers={"Authorization": "Bearer k"})
    assert response.status_code == 200
    assert calls == ["a mug"]


def test_configured_url_format_is_a_signed_artifact_link(tmp_path, monkeypatch):
    output = BytesIO()
    Image.new("RGB", (1024, 1024), "blue").save(output, format="PNG")
    client, _, settings = configured_comfy_client(
        tmp_path, monkeypatch, output.getvalue(),
        public_artifact_base="http://images.example",
        artifact_signing_key="new-artifact-signing-secret-for-tests")
    response = client.post(URL, json={"prompt": "a mug", "response_format": "url"})
    assert response.status_code == 200
    url = response.json()["data"][0]["url"]
    parsed = urlparse(url)
    artifact_id = parsed.path.rsplit("/", 1)[1]
    query = parse_qs(parsed.query)
    assert query["sig"][0] == artifact_links.signature(
        settings.artifact_signing_key, artifact_id, int(query["expires"][0]))
    assert client.get(parsed.path + "?" + parsed.query).status_code == 200


def test_configured_comfy_refuses_wrong_output_dimensions(tmp_path, monkeypatch):
    client, calls, settings = configured_comfy_client(tmp_path, monkeypatch, PNG)
    response = client.post(URL, json={"model": "chord-1-poly", "prompt": "a mug"})
    assert response.status_code == 502 and "data" not in response.json()
    assert response.headers["x-should-retry"] == "false"
    assert calls == ["a mug"]
    assert last_trace(settings)["image_backend_error"] == "ValueError"


@pytest.mark.parametrize("extra,param", [
    ({"n": 2}, "n"), ({"n": True}, "n"), ({"n": 1.0}, "n"),
    ({"size": "1024x768"}, "size"), ({"size": "2048x2048"}, "size"),
    ({"style": "vivid"}, "style"), ({"quality": "hd"}, "quality"), ({"output_format": "webp"}, "output_format"),
])
def test_what_we_cannot_honour_is_refused_not_ignored(tmp_path, monkeypatch, extra, param):
    client, _, _ = client_for(tmp_path, monkeypatch, rendered())
    r = client.post(URL, json={"model": "chord-1-poly", "prompt": "a mug", **extra})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter" and r.json()["error"]["param"] == param


@pytest.mark.parametrize("n", [True, 1.0, 2, 0])
def test_edits_rejects_an_n_that_is_not_one_with_strict_typing(tmp_path, monkeypatch, n):
    """The three image doors used to disagree on what counts as a count: edits
    accepted `n: true` and `n: 1.0` because `True == 1` and `1.0 == 1`, generations
    refused both by type, and variations took any parseable string. Pinned here
    so the doors agree: `None` or an int in `[1, 1]`, nothing else."""
    client = _plain_client(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer k"}
    r = client.post("/v1/images/edits",
                    data={"prompt": "add a hat", "model": "chord-1-poly", "n": n},
                    files={"image": ("cat.png", _png_file(), "image/png")},
                    headers=headers)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "n"
    assert r.json()["error"]["code"] == "invalid_value"


@pytest.mark.parametrize("n", [True, 1.0, 11, 0, "abc"])
def test_variations_rejects_an_n_that_is_not_an_int_in_range_with_strict_typing(tmp_path, monkeypatch, n):
    """Same family as the edits test: variations' range is 1..10, but the strictness
    around bool/float/non-int-string must match the other doors."""
    client = _plain_client(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer k"}
    r = client.post("/v1/images/variations",
                    data={"model": "dall-e-2", "n": n},
                    files={"image": ("square.png", _png_file(), "image/png")},
                    headers=headers)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "n"
    assert r.json()["error"]["code"] == "invalid_value"


def test_edits_accepts_n_as_an_int_or_string_or_absent(tmp_path, monkeypatch):
    """The tightened doors still accept the three shapes OpenAI documents:
    omitted, int, and string-that-parses. Pinned beside the rejection tests."""
    client = _plain_client(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer k"}
    for n in (None, 1, "1"):
        r = client.post("/v1/images/edits",
                        data={"prompt": "add a hat", "model": "chord-1-poly", **({"n": n} if n is not None else {})},
                        files={"image": ("cat.png", _png_file(), "image/png")},
                        headers=headers)
        # The point is that `n` itself was accepted, not refused as malformed.
        # A successful edit returns 200; without a video_backend it returns
        # 503 backend_unavailable. Both are post-validation outcomes.
        assert r.status_code != 400, (n, r.text)


@pytest.mark.parametrize("n", [True, 1.0, 2, 0])
def test_edits_json_path_rejects_a_non_count_n_before_any_backend_work(tmp_path, monkeypatch, n):
    """The JSON door for edits does not coerce `n` through _text(), so a raw
    bool or float slipped past the old `count not in (None, "", "1", 1)` check
    (True == 1, 1.0 == 1) and the request hit the backend with `n=True` --
    a 503 backend_unavailable rather than the honest 400 the caller deserves.
    Pinned so a regression is caught at the door, not as 'why does the edit
    endpoint 503 on n=true'."""
    # Need a real file_id for the JSON door.
    client = _plain_client(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer k"}
    upload = client.post("/v1/files", data={"purpose": "user_data"},
                         files={"file": ("cat.png", _png_file(), "image/png")},
                         headers=headers)
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["id"]

    r = client.post("/v1/images/edits",
                    json={"prompt": "add a hat", "model": "chord-1-poly",
                          "n": n, "images": [{"file_id": file_id}]},
                    headers=headers)
    assert r.status_code == 400, f"n={n!r} should be rejected as invalid_value"
    assert r.json()["error"]["param"] == "n"
    assert r.json()["error"]["code"] == "invalid_value"


@pytest.mark.parametrize("body,status", [
    ({"model": "chord-1-open", "prompt": "a mug"}, 404),
    ({"model": "gpt-image-1", "prompt": "a mug"}, 404),
    ({"model": "chord-1-poly"}, 400),
    ({"model": "chord-1-poly", "prompt": "   "}, 400),
    ({"model": "chord-1-poly", "prompt": 42}, 400),
    ([1, 2], 400),
])
def test_malformed_requests_are_structured_errors(tmp_path, monkeypatch, body, status):
    client, _, _ = client_for(tmp_path, monkeypatch, rendered())
    r = client.post(URL, json=body)
    assert r.status_code == status and "error" in r.json()


def real_png(side=1024):
    from io import BytesIO
    from PIL import Image
    out = BytesIO()
    Image.new("RGB", (side, side), (200, 120, 90)).save(out, format="PNG")
    return out.getvalue()


def corrupt_idat(png: bytes) -> bytes:
    """Whole, well-formed chunks with correct CRCs, but IDAT's compressed data
    is garbage: it registers, and fails only when decoded (#176)."""
    import zlib
    at = png.index(b"IDAT")
    length = int.from_bytes(png[at - 4:at], "big")
    garbage = bytes(b ^ 0x5A for b in png[at + 4:at + 4 + length])
    crc = (zlib.crc32(b"IDAT" + garbage) & 0xFFFFFFFF).to_bytes(4, "big")
    return png[:at + 4] + garbage + crc + png[at + 8 + length:]


# T08 (red team pass 1, 2026-09-15; T-IMG-014): CreateImageRequest does not
# require `model`. Omitted, the door answered 404 "model None not served".
@pytest.mark.parametrize("model", ["", "chord/", "chord-1-open", "dall-e-3", 5, ["chord-1-poly"]])
def test_a_named_image_model_we_do_not_serve_is_still_a_404(tmp_path, monkeypatch, model):
    client, _, _ = client_for(tmp_path, monkeypatch, rendered())
    r = client.post(URL, json={"model": model, "prompt": "a mug"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found" and r.json()["error"]["param"] == "model"


def _plain_client(tmp_path, monkeypatch, **settings_over):
    monkeypatch.setitem(specialists.SPECIALISTS, "image", rendered())
    settings = Settings(data_dir=tmp_path,
                        experimental_routes=frozenset({"image"}), service_api_key="k", **settings_over)
    return TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None)))


def test_url_format_is_refused_by_name_where_there_is_no_public_route(tmp_path, monkeypatch):
    """Decision #2 (review 2026-09-22): a deployment that cannot serve a URL
    says so -- it does not silently substitute a data URI inside the url field.
    All three doors refuse identically, before any backend work."""
    client = _plain_client(tmp_path, monkeypatch)              # no public_artifact_base
    headers = {"Authorization": "Bearer k"}
    want = ("unsupported_value", "response_format")

    gen = client.post(URL, json={"model": "chord-1-poly", "prompt": "a mug", "response_format": "url"},
                      headers=headers)
    assert gen.status_code == 400
    assert (gen.json()["error"]["code"], gen.json()["error"]["param"]) == want

    edit = client.post("/v1/images/edits",
                       data={"prompt": "add a hat", "model": "chord-1-poly", "response_format": "url"},
                       files={"image": ("cat.png", _png_file(), "image/png")}, headers=headers)
    assert edit.status_code == 400
    assert (edit.json()["error"]["code"], edit.json()["error"]["param"]) == want

    var = client.post("/v1/images/variations",
                      data={"model": "dall-e-2", "response_format": "url"},
                      files={"image": ("square.png", _png_file(), "image/png")},
                      headers=headers)
    assert var.status_code == 400
    assert (var.json()["error"]["code"], var.json()["error"]["param"]) == want


def _png_file() -> bytes:
    from io import BytesIO
    from PIL import Image
    out = BytesIO()
    Image.new("RGB", (8, 8), "navy").save(out, format="PNG")
    return out.getvalue()
