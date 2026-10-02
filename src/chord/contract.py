"""Types exchanged between the router and specialist jobs."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Outcome(str, Enum):
    """Machine-readable specialist outcome; user-facing text is separate."""

    chat = "chat"
    completed = "completed"
    needs_clarification = "needs_clarification"
    failed = "failed"
    cancelled = "cancelled"


class ArtifactDescriptor(BaseModel):
    id: str
    type: str  # "image" | "audio"
    mime: str
    sha256: str


class JobInputs(BaseModel):
    images: list[str] = Field(default_factory=list)  # data: or https: URLs
    transcript: str | None = None


class Job(BaseModel):
    """A specialist job. persona_id comes from the model id, never from content."""

    job_id: str
    revision: int = 1
    persona_id: str
    intent: str
    constraints: list[str] = Field(default_factory=list)
    latitude: str = ""
    inputs: JobInputs = Field(default_factory=JobInputs)
    conversation: list[dict] = Field(default_factory=list)
    web_search_options: dict | None = None


class Result(BaseModel):
    """A specialist result; `completed` means ready, not delivered."""

    job_id: str
    revision: int
    status: Outcome
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)
    summary: str = ""
    question: str | None = None
    provenance: dict = Field(default_factory=dict)
