import pytest
from chord.image_caption import CaptionImages, clean_caption


@pytest.mark.parametrize("markup", ["![watering can](image)", "![can](https://invented.invalid/a.png)",
    "![can](images/a(1).png)", "![a [small] can][picture]", r"![can\]](image)"])
def test_image_markup_removed_at_every_split(markup):
    source = "Here it is. " + markup + " Enjoy!"
    expected = "Here it is.  Enjoy!"
    assert clean_caption(source) == (expected, 1)
    for split in range(len(source) + 1):
        f = CaptionImages()
        assert f.feed(source[:split]) + f.feed(source[split:]) + f.feed("", final=True) == expected
        assert f.removed == 1
    f = CaptionImages()
    assert "".join(f.feed(c) for c in source) + f.feed("", final=True) == expected


@pytest.mark.parametrize("text", ["Here it is!", "[a link](https://example.com)", "![unfinished", "![label] plain", "Code: x != y"])
def test_non_image_text_preserved(text):
    f = CaptionImages()
    assert "".join(f.feed(c) for c in text) + f.feed("", final=True) == text
    assert f.removed == 0

@pytest.mark.parametrize("action_caption", [False, True, "bracket"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("image_turn", [False, True])
def test_endpoint_filters_only_completed_image_captions(tmp_path, monkeypatch, stream, image_turn, action_caption):
    import json
    from fastapi.testclient import TestClient
    from chord import specialists
    from chord.config import Settings
    from chord.contract import Result, Outcome
    from chord.server import Deps, create_app, load_specialists
    from test_progress import FixedRouter, last_trace
    from test_skeleton import AvailableImageBackend, FakeUpstream, PNG
    from test_caption_no_tools import split_images
    load_specialists()
    async def render(job, ctx):
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                      artifacts=[ctx.artifacts.register(PNG, "image/png")])
    monkeypatch.setitem(specialists.SPECIALISTS, "image", render)
    caption = "Here it is. [Image: a lighthouse] Enjoy!" if action_caption == "bracket" else C2 if action_caption else "Here it is. ![watering can](image) Enjoy!"
    class Voice(FakeUpstream):
        async def complete(self, body):
            data, dep = await super().complete(body)
            data["choices"][0]["message"]["content"] = caption
            return data, dep
        async def stream(self, body):
            for char in caption:
                yield {"choices": [{"index": 0, "delta": {"content": char}}]}, {}
            yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, {}
    settings = Settings(data_dir=tmp_path, router_enabled=image_turn,
                        experimental_routes=frozenset({"image"}))
    client = TestClient(create_app(Deps(settings, upstream=Voice(), model=lambda _: FixedRouter(),
                                        image_backend=AvailableImageBackend())))
    r = client.post("/v1/chat/completions", json={"model": "chord-1-poly", "stream": stream,
        "messages": [{"role": "user", "content": "a watering can"}]})
    assert r.status_code == 200
    if stream:
        deltas = [json.loads(line[6:])["choices"][0]["delta"] for line in r.text.splitlines() if line.startswith("data: {") and json.loads(line[6:]).get("choices")]
        text, images = split_images("".join(d.get("content") or "" for d in deltas))
    else:
        text, images = split_images(r.json()["choices"][0]["message"]["content"])
    expected = "Got it. Let me create that for you.\n\n" if action_caption is True else "Here it is.  Enjoy!"
    assert text == (expected if image_turn else caption)
    assert bool(images) is image_turn
    assert last_trace(settings).get("caption_image_markup_removed", 0) == int(image_turn and not action_caption)
    assert last_trace(settings).get("caption_action_json_removed", 0) == int(image_turn and action_caption is True)
    assert last_trace(settings).get("caption_bracket_narration_removed", 0) == int(image_turn and action_caption == "bracket")

C2 = 'Got it. Let me create that for you.\n\n{\n  "action": "media_generation",\n  "action_input": {\n    "prompt": "A striking red lighthouse standing on a rugged rocky coast during a golden sunset. Warm orange and pink skies blending into the horizon, the sun dipping low and casting long reflections on the water. Jagged dark rocks in the foreground, gentle waves crashing against the stones. Cinematic, atmospheric, painterly quality. No people, no text, no lettering anywhere in the image.",\n    "image_size": "1024x1536"\n  }\n}'

@pytest.mark.parametrize("payload", [
    C2,
    'Before ```json\n{"action":"render","action_input":{"prompt":"a cup"}}\n``` after',
    'Before {"tool":"render_image","arguments":{"prompt":"a {red} lighthouse"}} after',
])
def test_fake_action_json_removed_at_every_split(payload):
    expected = 'Got it. Let me create that for you.\n\n' if payload == C2 else 'Before  after'
    for split in range(len(payload) + 1):
        f = CaptionImages()
        assert f.feed(payload[:split]) + f.feed(payload[split:]) + f.feed('', final=True) == expected
    f = CaptionImages()
    assert ''.join(f.feed(c) for c in payload) + f.feed('', final=True) == expected

@pytest.mark.parametrize("payload", [
    '{"action":"running","colour":"red"}',
    '{"arguments":["lighting","composition"]}',
    '```json\n{"colour":"red"}\n```',
    '```python\nprint("hello")\n```',
    '{"action":"unfinished', '```json\n{"action":"unfinished',
])
def test_ordinary_or_incomplete_json_preserved(payload):
    f = CaptionImages()
    assert ''.join(f.feed(c) for c in payload) + f.feed('', final=True) == payload


@pytest.mark.parametrize("source, expected", [
    ('Here ![cat](image) :{ then {"action":"render","action_input":{}}', 'Here  :{ then '),
    ('```JSON\n{"action":"render","action_input":{}}\n```', ''),
    ('{"action":"render","action_input":{"flag":true,"other":null}}', ''),
])
def test_resynchronizes_after_invalid_json_and_accepts_uppercase_fence(source, expected):
    for split in range(len(source) + 1):
        f = CaptionImages()
        assert f.feed(source[:split]) + f.feed(source[split:]) + f.feed('', final=True) == expected
        assert f.actions_removed == 1
    f = CaptionImages()
    assert ''.join(f.feed(c) for c in source) + f.feed('', final=True) == expected
    assert f.actions_removed == 1


def test_undecidable_brace_is_preserved_at_eof():
    f = CaptionImages()
    assert f.feed('Hello { then ordinary text') == 'Hello '
    assert f.feed('', final=True) == '{ then ordinary text'


@pytest.mark.parametrize("value", [r'"a \"quoted\" word"', r'"\u263a"', 'true', 'false', 'null', '-1.5e+2', '{"nested":[1,2]}'])
def test_partial_json_values_stay_buffered(value):
    source = '{"action":"render","action_input":{"value":' + value + '}}'
    for split in range(len(source) + 1):
        f = CaptionImages()
        assert f.feed(source[:split]) + f.feed(source[split:]) + f.feed('', final=True) == ''
        assert f.actions_removed == 1
    f = CaptionImages()
    assert ''.join(f.feed(c) for c in source) + f.feed('', final=True) == ''


def test_unclosed_fence_does_not_hide_later_action_at_eof():
    source = 'Here ```unfinished then {"action":"render","action_input":{}}'
    expected = 'Here ```unfinished then '
    for split in range(len(source) + 1):
        f = CaptionImages()
        assert f.feed(source[:split]) + f.feed(source[split:]) + f.feed('', final=True) == expected
        assert f.actions_removed == 1
