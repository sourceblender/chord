"""The official-SDK lane (#131): the code real clients run parses what we send.

The JSON Schema lane validates against the pinned spec; this is the second
opinion from openai-python itself, offline, over the real app (TestClient is an
httpx.Client, so the SDK's own transport, retries off, drives it). Parsing alone
is lenient (pydantic keeps unknown keys), so every parsed object is also walked
for `model_extra`: a key the SDK's types don't declare is a failure here."""
import base64
import json
from io import BytesIO

import openai
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image

from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app, load_specialists
from test_audio_input import HearingUpstream, voice
from test_audio_output import SpeakingUpstream
from test_progress import FixedRouter
from test_skeleton import AvailableImageBackend, PNG, FakeUpstream
from test_images_api import configured_comfy_client

load_specialists()
MODEL = "chord-1-poly"


def sdk(app) -> OpenAI:
    return OpenAI(base_url="http://testserver/v1", api_key="k", max_retries=0, http_client=TestClient(app))


def undeclared(obj, path="$") -> list[str]:
    """Every key the SDK's own types don't declare, anywhere in the object."""
    out = []
    if isinstance(obj, BaseModel):
        out += [f"{path}.{k}" for k in (obj.model_extra or {})]
        for name in type(obj).model_fields:
            out += undeclared(getattr(obj, name), f"{path}.{name}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            out += undeclared(item, f"{path}[{i}]")
    return out


def plain_app(tmp_path, upstream=None, **settings):
    return create_app(Deps(Settings(data_dir=tmp_path, **settings),
                           upstream=upstream or FakeUpstream(), model=lambda n: None))


def image_app(tmp_path, monkeypatch):
    async def image(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")], summary="a mug")
    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    settings = Settings(data_dir=tmp_path, router_enabled=True, enabled_routes=frozenset({"image"}))
    return create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                           image_backend=AvailableImageBackend()))


def test_models_list_and_retrieve(tmp_path):
    client = sdk(plain_app(tmp_path))
    listing = client.models.list()
    one = client.models.retrieve(listing.data[0].id)
    assert one.id == MODEL
    assert undeclared(listing.data) == [] and undeclared(one) == []


@pytest.mark.parametrize("routed", [False, True], ids=["plain", "routed-image"])
def test_chat_non_stream(tmp_path, monkeypatch, routed):
    client = sdk(image_app(tmp_path, monkeypatch) if routed else plain_app(tmp_path))
    c = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "draw a mug"}])
    assert c.object == "chat.completion" and c.choices[0].finish_reason == "stop"
    assert ("![image](data:image/png;base64," in c.choices[0].message.content) is routed
    assert undeclared(c) == []


@pytest.mark.parametrize("routed", [False, True], ids=["plain", "routed-image"])
def test_chat_stream_with_usage(tmp_path, monkeypatch, routed):
    client = sdk(image_app(tmp_path, monkeypatch) if routed else plain_app(tmp_path))
    chunks = list(client.chat.completions.create(model=MODEL, stream=True, stream_options={"include_usage": True},
                                                 messages=[{"role": "user", "content": "draw a mug"}]))
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
    assert text.startswith("hi there") and ("![image](" in text) is routed
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 7
    assert [ch.choices[0].finish_reason for ch in chunks if ch.choices and ch.choices[0].finish_reason] == ["stop"]
    assert [u for ch in chunks for u in undeclared(ch)] == []


def test_audio_output(tmp_path):
    client = sdk(plain_app(tmp_path, SpeakingUpstream()))
    c = client.chat.completions.create(model=MODEL, modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"},
                                       messages=[{"role": "user", "content": "hi"}])
    assert base64.b64decode(c.choices[0].message.audio.data).startswith(b"RIFF")
    assert undeclared(c) == []


def test_unheard_voice_is_a_typed_sdk_error(tmp_path):
    client = sdk(plain_app(tmp_path, HearingUpstream(heard="")))
    with pytest.raises(openai.BadRequestError) as exc:
        client.chat.completions.create(model=MODEL, messages=[voice()])
    assert exc.value.code == "audio_unintelligible" and exc.value.param == "messages"


def test_a_stream_failing_after_headers_raises_in_the_sdk(tmp_path):
    class Breaks(FakeUpstream):
        async def stream(self, body):
            yield None, {}
            yield {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}, {}
            raise RuntimeError("backend fell over")

    client = sdk(plain_app(tmp_path, Breaks()))
    with pytest.raises(openai.APIError, match="the response failed while streaming"):
        list(client.chat.completions.create(model=MODEL, stream=True, messages=[{"role": "user", "content": "hi"}]))


def test_images_generate(tmp_path, monkeypatch):
    output = BytesIO()
    Image.new("RGB", (1024, 1024), "blue").save(output, format="PNG")
    client, _, _ = configured_comfy_client(tmp_path, monkeypatch, output.getvalue())
    app = client.app
    img = sdk(app).images.generate(model=MODEL, prompt="a mug", size="256x256", response_format="b64_json", n=1)
    assert base64.b64decode(img.data[0].b64_json).startswith(b"\x89PNG")
    assert undeclared(img) == []


def test_unknown_model_is_a_typed_not_found(tmp_path):
    with pytest.raises(openai.NotFoundError):
        sdk(plain_app(tmp_path)).models.retrieve("nope")


def test_the_lane_catches_an_undeclared_field():
    """Red proof of the walker itself: a key the SDK doesn't declare is reported."""
    from openai.types.chat import ChatCompletion
    c = ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 0, "model": MODEL, "outcome": "chat",
                                       "choices": [{"index": 0, "finish_reason": "stop",
                                                    "message": {"role": "assistant", "content": "hi", "trace_id": "T"}}]})
    assert sorted(undeclared(c)) == ["$.choices[0].message.trace_id", "$.outcome"]
    json.dumps(c.model_dump())
