"""Voice messages (#82b).

2026-09-13: asked for a clip, got "(audio clip playing)" and no clip. A caller
that declares its own tools (OpenClaw offers `tts`) is never routed at all (S04).
For a caller that asked for audio output (modalities), this marks
the turn as a voice message: the assistant writes only the words it will say
(graph._outcome_note) and the server returns them as message.audio. A caller
that didn't ask for audio can't receive a voice message inside the spec, so the
graph never reaches this for them: the assistant says it can't send one here.
Nothing is made here; the work happens after the words exist.
"""
from __future__ import annotations

from ..contract import Job, Outcome, Result
from . import SpecialistContext, specialist


@specialist("audio")
async def run(job: Job, ctx: SpecialistContext) -> Result:
    return Result(job_id=job.job_id, revision=job.revision, status=Outcome.completed,
                  summary=job.intent, provenance={"kind": "voice_message"})
