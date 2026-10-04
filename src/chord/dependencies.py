"""Production dependency construction for the public and internal applications."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from . import audio, comfy, registry
from .artifacts import ArtifactStore
from .comfy_workflow import ComfyImageBackend, ComfyImageInputBackend
from .config import Settings
from .specialists import SPECIALISTS
from .trace import TraceSink
from .upstream import Upstream
from .video import VideoBackend


# ChatOpenAI will not construct without an API key. An uncredentialed direct
# backend gets an explicitly empty Authorization header; this inert value only
# satisfies the client constructor and is never sent as a bearer credential.
UNCREDENTIALED = "chord-no-credential-configured"

# A specialist model call is planning, not rendering: five minutes without an
# answer is a hung backend. Callers that own a tighter deadline (router,
# delivery check) wrap their call in wait_for regardless. The openai
# defaults this replaces -- 600 s and TWO silent retries on 5xx -- meant a hung
# backend could hold a chat turn for half an hour, and the invisible retries
# doubled GPU load on exactly the turns that were already failing. A blind
# retry can duplicate work (review 2026-09-22).
MODEL_TIMEOUT_S = 300.0
SPECIALIST_ENTRY_MODULES = ("image", "search", "voice_message")


def load_specialists() -> list[str]:
    """Import registered specialist entry points."""
    from . import specialists as package

    for name in SPECIALIST_ENTRY_MODULES:
        importlib.import_module(f"{package.__name__}.{name}")
    return sorted(SPECIALISTS)


class Deps:
    """Application-owned dependencies shared with API-family registrations."""

    def __init__(
        self,
        settings: Settings,
        upstream: Upstream | None = None,
        model: Callable[[str], BaseChatModel] | None = None,
        video_backend: VideoBackend | None = None,
        image_backend: ComfyImageBackend | None = None,
        edit_backend: ComfyImageInputBackend | None = None,
        variation_backend: ComfyImageInputBackend | None = None,
    ):
        self.settings = settings
        if upstream is None:
            persona_model, persona_url = settings.slot_target("persona")
            router_model, router_url = settings.slot_target("router")
            if persona_model == router_model and persona_url.rstrip("/") != router_url.rstrip("/"):
                raise ValueError(
                    f"chat model {persona_model!r} cannot identify two different direct routes"
                )
            # STT/TTS addresses must be resolved from their explicit *_BASE_URL,
            # not via base_url_for(model). The latter resolves by model name and
            # would silently return the persona URL if STT_MODEL equals
            # PERSONA_MODEL and STT_BASE_URL is unset, defeating the
            # refuse_unset_audio guarantee (Copilot review of #341, 2026-09-27).
            # An unset audio URL leaves the client absent and the handler
            # refuses with 503.
            stt_url = settings.stt_base_url
            tts_url = settings.tts_base_url
            embeddings_url = settings.embeddings_base_url

            keys = {}
            for url in (persona_url, router_url, stt_url, tts_url, embeddings_url):
                if url:
                    keys[url.rstrip("/")] = settings.credential_for(url)

            upstream = Upstream(
                persona_url,
                "",
                stt_base_url=stt_url,
                tts_base_url=tts_url,
                embeddings_base_url=embeddings_url,
                embeddings_auth=settings.embeddings_auth_header(),
                chat_routes={persona_model: persona_url, router_model: router_url},
                keys=keys,
                refuse_unset_audio=True,
            )
        self.upstream = upstream
        # One GPU lock per ComfyUI process: every route that points at the same
        # address (video, image, edit, variation) waits on the same lock, and
        # routes on different processes never wait on each other.
        locks: dict[str, asyncio.Lock] = {}

        def render_lock(url: str) -> asyncio.Lock | None:
            return locks.setdefault(url.rstrip("/"), asyncio.Lock()) if url else None

        self.video_backend = (
            video_backend if video_backend is not None
            else VideoBackend.from_settings(settings, render_lock=render_lock(settings.comfy_base_url))
        )
        self.image_backend = image_backend if image_backend is not None else (
            ComfyImageBackend(
                settings.image_workflow.load(),
                comfy.ComfyClient(settings.image_comfy_base_url, timeout=settings.image_deadline_s),
                timeout_s=settings.image_deadline_s,
                render_lock=render_lock(settings.image_comfy_base_url),
            ) if settings.image_workflow is not None else None
        )

        def input_backend(workflow, url: str) -> ComfyImageInputBackend | None:
            if workflow is None:
                return None
            return ComfyImageInputBackend(
                workflow.load(), comfy.ComfyClient(url, timeout=settings.image_deadline_s),
                timeout_s=settings.image_deadline_s, render_lock=render_lock(url),
            )

        self.edit_backend = edit_backend if edit_backend is not None else input_backend(
            settings.image_edit_workflow, settings.image_edit_comfy_base_url)
        self.variation_backend = variation_backend if variation_backend is not None else input_backend(
            settings.image_variation_workflow, settings.image_variation_comfy_base_url)

        # One client per model name, built on first use. A fresh ChatOpenAI per
        # call rebuilt the httpx pool -- TCP/TLS setup on every router decision,
        # specialist attempt -- and left the old pool to
        # GC without a close (review 2026-09-22).
        clients: dict[str, ChatOpenAI] = {}

        def model_client(name: str) -> ChatOpenAI:
            cached = clients.get(name)
            if cached is not None:
                return cached
            url = settings.base_url_for(name)
            credential = settings.credential_for(url)
            # The router model never thinks, whoever calls it. #345 switched it
            # off at the four graph.py call sites, and search -- which calls the
            # same model by its registry name -- kept thinking on (review
            # 2026-09-27, #6). Keyed on the model here, every caller inherits it.
            # Not when the router IS the persona: persona thinking is the
            # client's to ask for (reasoning_effort).
            thinking_off = ({"chat_template_kwargs": {"enable_thinking": False}}
                            if (settings.config_version != 2
                                and name == settings.router_model and name != settings.persona_model
                                and settings.router_thinking_mode == "qwen_chat_template") else None)
            # Version 2 binds thinking at each call by its role (router_client for the
            # helper's calls, the image writer by the selected reply), so a client
            # cached for one role never carries another role's switch.
            if credential:
                client = ChatOpenAI(
                    model=name,
                    base_url=url,
                    api_key=SecretStr(credential),
                    temperature=0,
                    timeout=MODEL_TIMEOUT_S,
                    max_retries=0,
                    extra_body=thinking_off,
                )
            else:
                client = ChatOpenAI(
                    model=name,
                    base_url=url,
                    temperature=0,
                    api_key=SecretStr(UNCREDENTIALED),
                    default_headers={"Authorization": ""},
                    timeout=MODEL_TIMEOUT_S,
                    max_retries=0,
                    extra_body=thinking_off,
                )
            clients[name] = client
            return client

        self.model = model or model_client
        self.artifacts = ArtifactStore(settings.artifact_dir)
        self.traces = TraceSink(settings.trace_dir, data_dir=settings.data_dir,
                                retention_days=settings.retention_days)
        self.capabilities = registry.load()
        self.specialists = load_specialists()
        self.transcriber = audio.Transcriber(self.upstream, settings.stt_model)
        self.speaker = audio.Speaker(self.upstream, settings.tts_model, self.artifacts)
