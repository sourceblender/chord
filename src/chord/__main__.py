"""`python -m chord` — the public app and the loopback eval app."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
from urllib.parse import urlsplit

import uvicorn

from .config import ConfigurationError, Settings, load_settings
from .dependencies import Deps
from . import registry
from .server import create_app, create_internal_app
from .videos import video_tools_available


def effective_config_view(settings: Settings, *, yaml_mode: bool) -> dict:
    """An explicit, credential-free view of the routes this process would use.

    Show only URL origins. A URL can contain userinfo, a secret-bearing path,
    query, or fragment, so printing the resolved URL is not safe even when the
    separate auth field is hidden.
    """
    def origin(url: str) -> str | None:
        if not url:
            return None
        try:
            parsed = urlsplit(url)
            host = parsed.hostname
            if not host or parsed.scheme not in {"http", "https"}:
                return "invalid"
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            return "invalid"
        if ":" in host:  # IPv6 authority
            host = f"[{host}]"
        return f"{parsed.scheme}://{host}{port}"

    def route(model: str, url: str, auth: str = "") -> dict:
        return {"configured": bool(model and url), "model": model or None,
                "origin": origin(url), "auth": "bearer" if auth else "none"}

    routes = {
        "chat": route(settings.persona_model, settings.persona_base_url, settings.persona_api_key),
        "router": {**route(settings.router_model, settings.router_base_url, settings.router_api_key),
                   "backend": settings.router_backend,
                   "thinking": settings.router_thinking_mode,
                   "json_mode": settings.router_json_mode,
                   "classifier_origin": origin(settings.router_classifier_url)},
        "stt": route(settings.stt_model, settings.stt_base_url, settings.stt_api_key),
        "tts": route(settings.tts_model, settings.tts_base_url, settings.tts_api_key),
        "embeddings": {**route(settings.embeddings_model, settings.embeddings_base_url,
                                settings.embeddings_api_key),
                       "auth": ("bearer" if settings.embeddings_api_key else
                                "basic" if settings.embeddings_basic_auth else "none")},
        "video": {"configured": settings.video_workflows is not None,
                  "origin": origin(settings.comfy_base_url),
                  "kinds": sorted(settings.video_workflows or {}),
                  "tools_available": video_tools_available(),
                  "serving": settings.video_workflows is not None and video_tools_available()},
        "image": {"configured": settings.image_workflow is not None,
                  "origin": origin(settings.image_comfy_base_url),
                  "workflow_configured": settings.image_workflow is not None,
                  "serving": settings.image_workflow is not None,
                  "chat_routable": settings.image_workflow is not None and settings.router_enabled
                  and "image" in registry.load()},
        "image_edit": {"configured": settings.image_edit_workflow is not None,
                       "origin": origin(settings.image_edit_comfy_base_url)},
        "image_variation": {"configured": settings.image_variation_workflow is not None,
                            "origin": origin(settings.image_variation_comfy_base_url),
                            "max_n": (None if settings.image_variation_workflow is None else
                                      10 if settings.image_variation_workflow.seed_node_id else 1)},
    }
    routes["chat"]["thinking"] = settings.persona_thinking_mode
    return {"mode": "yaml-v1" if yaml_mode else "legacy-environment",
            "router_enabled": settings.router_enabled, "routes": routes}


async def main(settings: Settings | None = None) -> None:
    # Keep the direct async entry point used by operators and tests while the
    # CLI can pass an already validated configuration.
    if settings is None:
        settings = load_settings()
    deps = Deps(settings)
    public = uvicorn.Server(uvicorn.Config(create_app(deps), host=settings.public_host, port=settings.public_port, log_level="info"))
    # The internal server binds to loopback regardless of PUBLIC_HOST.
    internal = uvicorn.Server(uvicorn.Config(create_internal_app(deps), host="127.0.0.1", port=settings.internal_port, log_level="info"))
    try:
        await asyncio.gather(public.serve(), internal.serve())
    finally:
        await deps.upstream.aclose()
        # The ComfyUI client is a process-lifetime singleton; its aclose()
        # existed with no caller, so shutdown dropped its connections instead
        # of draining them (review 2026-09-22).
        if deps.video_backend is not None:
            await deps.video_backend.aclose()
        if deps.image_backend is not None:
            await deps.image_backend.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chord or validate its configuration")
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--check-config", action="store_true",
                         help="validate configuration without starting the server or contacting a backend")
    offline.add_argument("--show-config", action="store_true",
                         help="print resolved routes with credentials and URL paths redacted; contact no backend")
    args = parser.parse_args()
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        parser.exit(2, f"chord: {exc}\n")
    if args.check_config:
        print("Chord configuration OK")
    elif args.show_config:
        print(json.dumps(effective_config_view(settings, yaml_mode=bool(os.environ.get("CHORD_CONFIG"))),
                         indent=2, sort_keys=True))
    else:
        asyncio.run(main(settings))
