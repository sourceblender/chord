"""In chat, the model the user is talking to writes the image prompt (specialists/image.py)."""
import asyncio
from types import SimpleNamespace

import pytest

from chord.config import Settings
from chord.contract import Job
from chord.specialists import SpecialistContext
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from chord.specialists.image import PROMPT_ONLY, PROMPT_WRITER, write_prompt
from chord.trace import Trace


class Reply:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class Model:
    def __init__(self, behaviour):
        self.behaviour, self.seen, self.bound = behaviour, None, None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    async def ainvoke(self, messages):
        self.seen = messages
        return await self.behaviour() if asyncio.iscoroutinefunction(self.behaviour) else self.behaviour()


def ctx(model, *, persona_model="", timeout=5.0, thinking="qwen_chat_template"):
    names = []
    settings = Settings(persona_model="chat-model", router_model="router-model",
                        image_prompt_timeout_s=timeout, persona_thinking_mode=thinking)

    def factory(name):
        names.append(name)
        return model
    c = SpecialistContext(settings=settings, artifacts=SimpleNamespace(), trace=Trace(persona_id="generic", model_id_requested="chord-1-poly"), model=factory,
                          persona_model=persona_model)
    return c, names


def job(*turns):
    return Job(job_id="j", persona_id="generic", intent="draw that", conversation=list(turns))


LONG = job({"role": "system", "text": "You are Ada. Pictures are watercolour."},
           {"role": "developer", "text": "Keep scenes outdoors."},
           {"role": "user", "text": "My old red fire truck has a dented ladder."},
           *[{"role": r, "text": f"chat {i}"} for i in range(5) for r in ("assistant", "user")],
           {"role": "user", "text": "Create that picture now."})


def run(coro):
    return asyncio.run(coro)


def test_the_chat_model_writes_the_prompt_from_the_whole_request():
    m = Model(lambda: Reply("A watercolour of a red fire truck with a dented ladder, outdoors"))
    c, names = ctx(m)
    prompt, source = run(write_prompt(LONG, c))
    assert (prompt, source) == ("A watercolour of a red fire truck with a dented ladder, outdoors", "chat_model")
    system, rest = m.seen[0], m.seen[1:]
    # Instructions keep their authority: folded into the one system message, never a user turn.
    assert isinstance(system, SystemMessage)
    assert system.content.startswith(PROMPT_WRITER) and system.content.rstrip().endswith(PROMPT_ONLY)
    assert "Pictures are watercolour" in system.content
    assert "Developer instruction" in system.content and "Keep scenes outdoors" in system.content
    assert not any("watercolour" in x.content or "outdoors" in x.content for x in rest)
    # Detail from more than eight messages back reaches the writer as conversation.
    assert any(isinstance(x, HumanMessage) and "dented ladder" in x.content for x in rest)
    assert isinstance(rest[-1], HumanMessage) and rest[-1].content == "Create that picture now."
    assert any(isinstance(x, AIMessage) for x in rest)
    assert names == ["chat-model"] and c.trace.fields["image_prompt_model"] == "chat-model"
    assert m.bound == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_the_effective_persona_writes_it_when_a_service_tier_overrides_the_model():
    c, names = ctx(Model(lambda: Reply("a red fire truck")), persona_model="router-model")
    assert run(write_prompt(LONG, c)) == ("a red fire truck", "chat_model")
    assert names == ["router-model"] and c.trace.fields["image_prompt_model"] == "router-model"


def test_passthrough_thinking_binds_nothing():
    m = Model(lambda: Reply("a red fire truck"))
    c, _ = ctx(m, thinking="passthrough")
    run(write_prompt(LONG, c))
    assert m.bound is None


@pytest.mark.parametrize("reply", [
    Reply(""), Reply("   "),
    Reply('{"route": "image", "intent": "a red fire truck"}'),
    Reply('[{"name": "image_generate"}]'),
    Reply('<tool_call>{"name": "image_generate"}</tool_call>'),
    Reply("a red fire truck", tool_calls=[{"name": "image_generate", "args": {}}]),
])
def test_a_reply_that_is_not_a_prompt_falls_back_to_the_users_own_words(reply):
    c, _ = ctx(Model(lambda: reply))
    assert run(write_prompt(LONG, c)) == ("Create that picture now.", "user_words_fallback")
    assert "image_prompt_error" not in c.trace.fields and "image_prompt_model" not in c.trace.fields


def test_a_failing_writer_falls_back_and_is_traced():
    def boom():
        raise ConnectionError("katy down")
    c, _ = ctx(Model(boom))
    assert run(write_prompt(LONG, c)) == ("Create that picture now.", "user_words_fallback")
    assert c.trace.fields["image_prompt_error"] == "ConnectionError"


def test_a_slow_writer_times_out_to_the_users_own_words():
    async def slow():
        await asyncio.sleep(1)
        return Reply("too late")
    c, _ = ctx(Model(slow), timeout=0.01)
    assert run(write_prompt(LONG, c)) == ("Create that picture now.", "user_words_fallback")
    assert c.trace.fields["image_prompt_error"] == "timed out"


def test_cancellation_propagates():
    async def cancelled():
        raise asyncio.CancelledError
    c, _ = ctx(Model(cancelled))
    with pytest.raises(asyncio.CancelledError):
        run(write_prompt(LONG, c))


def test_quotes_and_fences_around_a_prompt_are_removed():
    c, _ = ctx(Model(lambda: Reply('"a red fire truck at dusk"')))
    assert run(write_prompt(LONG, c))[0] == "a red fire truck at dusk"
    c, _ = ctx(Model(lambda: Reply("```\na red fire truck at dusk\n```")))
    assert run(write_prompt(LONG, c))[0] == "a red fire truck at dusk"


def test_an_over_long_reply_falls_back_instead_of_being_cut():
    import json as _json
    long_json = _json.dumps({"route": "image", "intent": "x" * 4480})      # > limit, valid JSON
    assert len(long_json) > 4000
    c, _ = ctx(Model(lambda: Reply(long_json)))
    assert run(write_prompt(LONG, c)) == ("Create that picture now.", "user_words_fallback")
    assert c.trace.fields["image_prompt_error"] == "PromptTooLong"
    c, _ = ctx(Model(lambda: Reply("a red fire truck, " * 330)))             # ~5900 plain chars
    assert run(write_prompt(LONG, c)) == ("Create that picture now.", "user_words_fallback")
    assert c.trace.fields["image_prompt_error"] == "PromptTooLong"
