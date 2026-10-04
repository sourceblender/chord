"""Image specialist through the operator's configured ComfyUI workflow."""
from __future__ import annotations

from io import BytesIO

from PIL import Image

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage

from ..contract import Job, Outcome, Result
from . import SpecialistContext, specialist

# In chat, the model the user is talking to writes the render prompt, as a
# plain-text reply: no tool call, so nothing to repair. The direct images
# endpoint never comes here; its caller's prompt is rendered as given.
PROMPT_WRITER = (
    "You write the text prompt for an image generator, on behalf of the assistant in the conversation below. "
    "Write one prompt for the picture the user is asking for now. Keep every detail the user gave: subject, "
    "setting, style, mood, composition, lighting, and anything they said to include or avoid. Add only what "
    "makes the picture concrete. Reply with the prompt text only: no preamble, no quotes, no markdown, no "
    "explanation."
)
PROMPT_MAX_CHARS = 4000


def _transcript(conversation: list[dict]) -> str:
    return "\n\n".join(f"{m.get('role', 'user')}: {m.get('text', '')}" for m in conversation if m.get("text"))


def _last_user_text(job: Job) -> str:
    return next((m.get("text", "") for m in reversed(job.conversation) if m.get("role") == "user"), "") or job.intent


def _usable(reply) -> str:
    """The reply's prompt text, or "" when it is not a prompt: tool calls, a
    JSON object, or tool-call markup are never rendered as a picture."""
    if getattr(reply, "tool_calls", None) or getattr(reply, "invalid_tool_calls", None):
        return ""
    text = _clean(reply.content if isinstance(getattr(reply, "content", None), str) else "")
    if "<tool_call>" in text or "</tool_call>" in text or '"tool_calls"' in text:
        return ""
    if text.startswith(("{", "[")):
        try:
            json.loads(text)
            return ""
        except ValueError:
            pass
    return text


def _clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").split("\n", 1)[-1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text[:PROMPT_MAX_CHARS]


async def write_prompt(job: Job, ctx: SpecialistContext) -> tuple[str, str]:
    """(prompt, source). source is `chat_model`, or `user_words_fallback` when the
    chat model fails, times out or returns nothing: then the user's own last
    message is rendered verbatim, never a summary."""
    fallback = _last_user_text(job).strip()
    model_id = ctx.persona_model or ctx.settings.persona_model
    try:
        llm = ctx.model(model_id)
        if ctx.settings.persona_thinking_mode == "qwen_chat_template" and hasattr(llm, "bind"):
            llm = llm.bind(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        reply = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(PROMPT_WRITER), HumanMessage(_transcript(job.conversation))]),
            ctx.settings.image_prompt_timeout_s,
        )
        text = _usable(reply)
    except Exception as exc:  # cancellation (BaseException) still propagates
        ctx.trace.set(image_prompt_error=("timed out" if isinstance(exc, TimeoutError) else type(exc).__name__))
        text = ""
    if text:
        ctx.trace.set(image_prompt_model=model_id)
        return text, "chat_model"
    return fallback, "user_words_fallback"


@specialist("image")
async def run(job: Job, ctx: SpecialistContext) -> Result:
    if ctx.image_backend is None:
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.failed,
                      summary="No image provider is configured.")
    return await _configured_run(job, ctx)


async def _configured_run(job: Job, ctx: SpecialistContext) -> Result:
    """Render a generic scene through the operator's workflow, without identity logic.

    The chat model the user is talking to writes the prompt from the whole
    conversation (write_prompt); Chord submits it unchanged, then validates and
    stores exactly one PNG.
    """
    backend = ctx.image_backend
    assert backend is not None
    ctx.progress("preparing")
    prompt, source = await write_prompt(job, ctx)
    if not prompt:
        return Result(job_id=job.job_id, revision=job.revision,
                      status=Outcome.needs_clarification,
                      question="What should the picture show?",
                      summary="No image prompt was submitted.")
    ctx.trace.set(image_backend="comfyui", image_prompt_source=source, image_prompt=prompt,
                  visual_inspection="not_performed")
    try:
        png, prompt_id = await backend.render(prompt)
        with Image.open(BytesIO(png)) as image:
            image.verify()
            if image.format != "PNG" or image.size != (1024, 1024):
                raise ValueError("image workflow must return a 1024x1024 PNG")
        descriptor = ctx.artifacts.register(png, "image/png")
    except Exception as exc:
        ctx.trace.set(image_backend_error=type(exc).__name__)
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.failed,
                      summary="Image generation did not return a usable image; this job was not automatically retried.")
    ctx.trace.set(comfy_prompt_id=prompt_id)
    ctx.trace.artifacts.append(descriptor.model_dump())
    return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                  artifacts=[descriptor],
                  summary="An image was generated from the requested scene. It has not been visually reviewed.",
                  provenance={"comfy_prompt_id": prompt_id, "prompt_source": source})
