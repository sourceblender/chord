"""The public wire is the pinned OpenAI spec, with nothing added (2026-09-16):
"the external interfaces should work and return within spec and not try to
optimize for the harness." Whatever the graph does inside, every chat body and
stream chunk that leaves the service validates with fields="strict": no
`outcome`, `artifacts`, `trace_id`, `job`, `images`, `image`, `event`,
`provider_specific_fields` or `audio_failed`. What a caller can use still
arrives, in the spec's own shapes."""
import json

import pytest
from fastapi.testclient import TestClient

from qa.conformance.schema import Spec, parse_sse, validate_payload
from chord import specialists
from chord.config import Settings
from chord.contract import Outcome, Result
from chord.server import Deps, create_app, load_specialists

from test_audio_input import HearingUpstream, voice
from test_progress import FixedRouter
from test_skeleton import AvailableImageBackend, PNG, FakeUpstream

load_specialists()
SPEC = Spec()


def strict(payload, kind):
    row = validate_payload(payload, kind=kind, spec=SPEC, fields="strict")
    assert row["verdict"] == "pass", json.dumps(row["evidence"], indent=1)[:2000]


def chat(client, stream, content="draw a mug", messages=None):
    body = {"model": "chord-1-poly", "stream": stream,
            "messages": messages or [{"role": "user", "content": content}]}
    if not stream:
        return client.post("/v1/chat/completions", json=body)
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        r.read()
        return r


def strict_response(r, stream):
    if not stream:
        strict(r.json(), "chat")
        return r.json()["choices"][0]["message"]["content"]
    chunks, done, framing = parse_sse(r.content)
    assert done and not framing
    for chunk in chunks:
        strict(chunk, "chat-stream")
    return "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))


@pytest.mark.parametrize("stream", [False, True])
def test_a_plain_chat_reply_is_strict(tmp_path, stream):
    client = TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=FakeUpstream(), model=lambda n: None)))
    r = chat(client, stream, "hello")
    assert r.status_code == 200
    assert strict_response(r, stream) == "hi there"
    assert r.headers["x-request-id"] == r.headers["x-chord-trace-id"]


@pytest.mark.parametrize("stream", [False, True])
def test_a_routed_image_turn_is_strict_and_the_image_arrives_as_markdown(tmp_path, monkeypatch, stream):
    async def fake_image(job, ctx):
        ctx.progress("preparing")
        d = ctx.artifacts.register(PNG, "image/png")
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed, artifacts=[d], summary="a mug")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", fake_image)
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        experimental_routes=frozenset({"image"}))
    client = TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: FixedRouter(),
                                        image_backend=AvailableImageBackend())))
    r = chat(client, stream)
    assert r.status_code == 200
    content = strict_response(r, stream)
    assert content.count("![") == 1 and "Preparing" not in content


@pytest.mark.parametrize("stream", [False, True])
def test_an_unheard_voice_message_is_an_error_not_a_reply(tmp_path, stream):
    client = TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=HearingUpstream(heard=""), model=lambda n: None)))
    r = chat(client, stream, messages=[voice()])
    assert r.status_code == 400
    strict(r.json(), "error")
    assert r.json()["error"]["param"] == "messages"


def test_models_are_strict(tmp_path):
    client = TestClient(create_app(Deps(Settings(data_dir=tmp_path), upstream=FakeUpstream(), model=lambda n: None)))
    listing = client.get("/v1/models").json()
    strict(listing, "models")
    one = client.get(f"/v1/models/{listing['data'][0]['id']}").json()
    strict({"object": "list", "data": [one]}, "models")
