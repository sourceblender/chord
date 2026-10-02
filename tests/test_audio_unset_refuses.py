"""With no STT/TTS address and no gateway, audio refuses instead of hitting chat.

Review of the sanitisation PR (2026-09-25): once LITELLM_BASE_URL stopped
defaulting to a real gateway, a persona/router-only deployment resolved the audio
models to "" and Upstream aliased both audio clients to the chat client.
"""

from __future__ import annotations

import httpx
import pytest

from chord.config import Settings
from chord.upstream import Upstream, UpstreamError


def _persona_only() -> Settings:
    return Settings(
        service_api_key="service-key",
        persona_model="example-persona",
        router_model="example-router",
        persona_base_url="http://persona.internal/v1",
        router_base_url="http://router.internal/v1",
    )


def _stt_model_reuses_persona_model() -> Settings:
    """STT_MODEL equals PERSONA_MODEL and STT_BASE_URL is unset. The previous
    resolution went through Settings.base_url_for(model), which matched the
    persona URL on the model name and produced a live chat client for _stt,
    defeating refuse_unset_audio. The fix routes STT/TTS URLs from the
    explicit *_BASE_URL only, so this configuration still refuses."""
    return Settings(
        service_api_key="service-key",
        persona_model="shared-model",
        persona_base_url="http://persona.internal/v1",
        router_model="router-model",
        router_base_url="http://router.internal/v1",
        stt_model="shared-model",   # collides with persona_model
        stt_base_url="",            # but the explicit URL is empty
        tts_model="shared-model",   # same collision
        tts_base_url="",
    )


@pytest.mark.asyncio
async def test_unset_audio_refuses_with_503_and_never_calls_chat() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"text": "wrong backend"})

    up = Upstream("http://persona.internal/v1", "", refuse_unset_audio=True)
    up._client = httpx.AsyncClient(base_url="http://persona.internal/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(UpstreamError) as stt:
        await up.transcribe(b"RIFF", "wav", "audio/wav", "stt-model")
    with pytest.raises(UpstreamError) as tts:
        await up.speak("hi", "voice", "wav", "tts-model")

    assert stt.value.status == 503 and "STT_BASE_URL" in str(stt.value)
    assert tts.value.status == 503 and "TTS_BASE_URL" in str(tts.value)
    assert calls == []
    await up.aclose()


def test_deps_wires_refusal_for_a_persona_only_deployment() -> None:
    from chord.dependencies import Deps

    settings = _persona_only()
    settings.validate_startup()  # still a valid, supported configuration
    deps = Deps(settings)
    assert deps.upstream._stt is None
    assert deps.upstream._tts is None


def test_deps_wires_refusal_when_stt_model_reuses_persona_model() -> None:
    """Copilot review of #341 (2026-09-27): STT_MODEL=PERSONA_MODEL with
    STT_BASE_URL unset used to alias the chat client via base_url_for(model).
    The explicit *_BASE_URL must win, so the audio client stays None and the
    handler refuses with 503 rather than POSTing audio to a chat backend."""
    from chord.dependencies import Deps

    settings = _stt_model_reuses_persona_model()
    settings.validate_startup()
    deps = Deps(settings)
    assert deps.upstream._stt is None
    assert deps.upstream._tts is None


def test_direct_unit_construction_keeps_the_single_client_fallback() -> None:
    up = Upstream("http://persona.internal/v1", "")
    assert up._stt is up._client and up._tts is up._client
