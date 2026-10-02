"""Image specialist through the operator's configured ComfyUI workflow."""
from __future__ import annotations

from io import BytesIO

from PIL import Image

from ..contract import Job, Outcome, Result
from . import SpecialistContext, specialist


@specialist("image")
async def run(job: Job, ctx: SpecialistContext) -> Result:
    if ctx.image_backend is None:
        return Result(job_id=job.job_id, revision=job.revision, status=Outcome.failed,
                      summary="No image provider is configured.")
    return await _configured_run(job, ctx)


async def _configured_run(job: Job, ctx: SpecialistContext) -> Result:
    """Route a generic scene to the operator's workflow, without identity logic.

    The provider owns any enhancement of the prompt. Chord passes the router's
    request intent and constraints, then validates and stores exactly one PNG.
    """
    backend = ctx.image_backend
    assert backend is not None
    prompt = job.intent.strip()
    if not prompt:
        return Result(job_id=job.job_id, revision=job.revision,
                      status=Outcome.needs_clarification,
                      question="What should the picture show?",
                      summary="No image prompt was submitted.")
    if job.constraints:
        prompt += "\nConstraints: " + "; ".join(job.constraints)
    ctx.trace.set(image_backend="comfyui", image_prompt_source="router_intent",
                  visual_inspection="not_performed")
    ctx.progress("preparing")
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
                  provenance={"comfy_prompt_id": prompt_id, "prompt_source": "router_intent"})
