"""The double render (red-team run b3, 2026-09-13): OpenClaw offered its own
image_generate, the service routed the selfie ask to its image specialist and
rendered too. Two pictures; OpenClaw dropped ours. When the request offers the
client's tool for the routed capability, the service does NOT do that work: the
turn is chat and the tools reach the model untouched."""
import pytest
from fastapi.testclient import TestClient

from chord import specialists
from chord.config import Settings
from chord.server import Deps, create_app, load_specialists
from test_progress import FixedRouter, last_trace
from test_skeleton import AvailableImageBackend, FakeUpstream

load_specialists()

IMAGE_TOOL = {"type": "function", "function": {"name": "image_generate", "parameters": {"type": "object"}}}
OTHER_TOOL = {"type": "function", "function": {"name": "web_fetch", "parameters": {"type": "object"}}}
SELFIE = "Send me a selfie in a blue coat beside a red door."


class AudioRouter:
    async def ainvoke(self, msgs):
        class R: content = '{"route": "audio", "intent": "say good morning"}'
        return R()


def client(tmp_path, monkeypatch, router=FixedRouter, upstream=None):
    renders = []

    async def image(job, ctx):
        renders.append(job)
        raise AssertionError("the service rendered")

    monkeypatch.setitem(specialists.SPECIALISTS, "image", image)
    settings = Settings(data_dir=tmp_path, router_enabled=True,
                        enabled_routes=frozenset({"image"}))
    up = upstream or FakeUpstream()
    return TestClient(create_app(Deps(settings, upstream=up, model=lambda n: router(),
                                      image_backend=AvailableImageBackend()))), up, settings, renders


def post(c, stream=False, **extra):
    body = {"model": "chord-1-poly", "messages": [{"role": "user", "content": SELFIE}], **extra}
    if stream:
        with c.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
            assert r.status_code == 200
            list(r.iter_lines())
        return
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200


@pytest.mark.parametrize("stream", [False, True])
def test_an_offered_image_generate_means_the_service_does_not_render(tmp_path, monkeypatch, stream):
    c, up, settings, renders = client(tmp_path, monkeypatch)
    post(c, stream, tools=[OTHER_TOOL, IMAGE_TOOL])
    # Zero specialist invocations (a stub, not a real render): the live b3 rerun
    # establishes actual zero service renders plus one door render (#91).
    assert renders == []
    assert up.bodies[0]["tools"] == [OTHER_TOOL, IMAGE_TOOL]      # her own tool reaches her
    t = last_trace(settings)
    assert t["router"] == "skipped_client_tools" and "specialist" not in t


def test_any_declared_tool_means_no_render_and_without_tools_it_renders(tmp_path, monkeypatch):
    """S04 (2026-09-16) widened #91: not only the matching client tool, any
    declared tool, under any tool_choice, keeps specialists out. Control: no tools."""
    c, up, settings, renders = client(tmp_path, monkeypatch)
    post(c, tools=[OTHER_TOOL])
    post(c, tools=[OTHER_TOOL, IMAGE_TOOL], tool_choice="none")
    assert renders == []
    post(c)
    assert len(renders) == 1


def test_a_call_forced_to_another_tool_is_the_clients_not_a_render(tmp_path, monkeypatch):
    """image_generate can't be called, but web_fetch must be: the answer is that
    call, so nothing renders (R1a, red team pass 1 S03, 2026-09-15). This was
    a render until then, and the forced call was lost."""
    from test_client_constraints import ObedientUpstream
    c, up, settings, renders = client(tmp_path, monkeypatch, upstream=ObedientUpstream())
    choice = {"type": "function", "function": {"name": "web_fetch"}}
    post(c, tools=[OTHER_TOOL, IMAGE_TOOL], tool_choice=choice)
    assert renders == []
    assert up.bodies[-1]["tool_choice"] == choice
    assert last_trace(settings)["router"] == "skipped_client_constraint"


def test_an_offered_tts_means_audio_is_chat_not_unavailable(tmp_path, monkeypatch):
    c, up, settings, renders = client(tmp_path, monkeypatch, router=AudioRouter)
    tts = {"type": "function", "function": {"name": "tts", "parameters": {"type": "object"}}}
    post(c, tools=[tts])
    t = last_trace(settings)
    assert t["router"] == "skipped_client_tools" and "route_unavailable" not in t
    assert "can't send voice" not in up.bodies[0]["messages"][0]["content"]
