"""Runtime configuration, from the environment. Secrets are never defaulted."""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .client_keys import parse_client_keys
from .comfy_workflow import ComfyImageInputWorkflow, ComfyImageWorkflow, WorkflowError
from .comfy_video_workflow import VideoWorkflow, VideoWorkflowError, parse_bindings


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"{name} is not set")
    return value


def _strict_bool_env(name: str, default: str) -> bool:
    value = _env(name, default).lower()
    if value not in {"true", "false"}:
        raise ConfigurationError(f"{name} must be true or false")
    return value == "true"


class ConfigurationError(ValueError):
    """All actionable startup configuration failures, reported together."""


@dataclass(frozen=True)
class ImageWorkflowConfig:
    path: Path
    directory: Path
    prompt_node_id: str
    output_node_id: str
    prompt_input_name: str = "text"
    seed_node_id: str | None = None
    seed_input_name: str = "seed"

    def load(self) -> ComfyImageWorkflow:
        _contained_file(self.path, self.directory)
        return ComfyImageWorkflow.load(
            self.path, prompt_node_id=self.prompt_node_id,
            output_node_id=self.output_node_id,
            prompt_input_name=self.prompt_input_name, seed_node_id=self.seed_node_id,
            seed_input_name=self.seed_input_name,
        )


def _contained_file(path: Path, directory: Path) -> None:
    # Resolve symlinks too: an operator-mounted workflow cannot escape the
    # directory declared for it, even through a link inside that directory.
    if not path.resolve().is_relative_to(directory.resolve()):
        raise WorkflowError("image workflow escapes image_workflows_dir")
    if not path.is_file():
        raise WorkflowError("image workflow must be a regular file")


@dataclass(frozen=True)
class ImageInputWorkflowConfig:
    """routing.image_edit or routing.image_variation: a workflow that takes the caller's image."""
    path: Path
    directory: Path
    image_node_id: str
    output_node_id: str
    image_input_name: str = "image"
    prompt_node_id: str | None = None
    prompt_input_name: str = "text"
    seed_node_id: str | None = None
    seed_input_name: str = "seed"

    def load(self) -> ComfyImageInputWorkflow:
        _contained_file(self.path, self.directory)
        return ComfyImageInputWorkflow.load(
            self.path, image_node_id=self.image_node_id, output_node_id=self.output_node_id,
            image_input_name=self.image_input_name, prompt_node_id=self.prompt_node_id,
            prompt_input_name=self.prompt_input_name, seed_node_id=self.seed_node_id,
            seed_input_name=self.seed_input_name,
        )
@dataclass(frozen=True)
class VideoWorkflowConfig:
    path: Path
    directory: Path
    output_node_id: str
    bindings: dict[str, tuple[str, str]]
    duration_unit: str = "seconds"
    fps: int | None = None

    def load(self) -> VideoWorkflow:
        try:
            _contained_file(self.path, self.directory)
        except WorkflowError as exc:
            raise VideoWorkflowError(str(exc)) from None
        return VideoWorkflow.load(self.path, output_node_id=self.output_node_id,
                                  bindings=self.bindings, duration_unit=self.duration_unit,
                                  fps=self.fps)



def _positive_finite(value: float) -> bool:
    """A usable duration. `value <= 0` alone let NaN (every comparison is
    False) and inf through, and neither bounds anything (Copilot on #338)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


# Manifest service-tier slots. `router` is the pre-version-2 spelling of `fast`.
TIER_SLOTS = frozenset({"persona", "fast", "router"})


def _dispatch(spec: object, interpolate) -> dict:
    """A version 2 `dispatch` block as Settings fields. Absent means nobody picks a
    lane: every turn is chat. The environment's ROUTER_ENABLED, ROUTER_BACKEND and
    ROUTER_CLASSIFIER_* never apply to a version 2 file."""
    off = {"router_enabled": False, "router_backend": "llm", "router_classifier_url": "",
           "router_classifier_timeout_s": 1.0}
    if spec is None:
        return off
    if not isinstance(spec, dict) or "by" not in spec or set(spec) - {"by", "classifier_url", "classifier_timeout_s"}:
        raise ConfigurationError("dispatch needs by, and may set classifier_url and classifier_timeout_s")
    by = spec["by"]
    if by not in ("classifier", "helper", "none"):
        raise ConfigurationError("dispatch.by must be classifier, helper or none")
    if by != "classifier" and set(spec) - {"by"}:
        raise ConfigurationError("dispatch.classifier_url and classifier_timeout_s need by: classifier")
    if by == "none":
        return off
    if by == "helper":
        return {**off, "router_enabled": True}
    if "classifier_url" not in spec:
        raise ConfigurationError("dispatch.by: classifier needs classifier_url")
    url = interpolate(spec["classifier_url"], "dispatch.classifier_url")
    timeout = spec.get("classifier_timeout_s", 1.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not _positive_finite(float(timeout)):
        raise ConfigurationError("dispatch.classifier_timeout_s must be greater than zero and finite")
    return {"router_enabled": True, "router_backend": "classifier", "router_classifier_url": url,
            "router_classifier_timeout_s": float(timeout)}


@dataclass(frozen=True)
class Settings:
    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        """Load endpoint choices from one operator-owned file.

        The environment remains the source for non-model server settings and
        for explicitly interpolated secrets. Endpoint values never fall back
        to an unrelated environment route when a YAML file is supplied.
        """
        try:
            raw = yaml.safe_load(path.read_text())
        except OSError as exc:
            raise ConfigurationError(f"cannot read CHORD_CONFIG {path}: {exc.strerror}") from None
        except yaml.YAMLError as exc:
            # PyYAML's exception text includes a source excerpt. That could
            # echo a literal credential in an operator's config.
            mark = getattr(exc, "problem_mark", None)
            location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            raise ConfigurationError(f"CHORD_CONFIG {path} is invalid YAML{location}") from None
        if not isinstance(raw, dict) or set(raw) - {"version", "endpoints", "routing", "image_workflows_dir",
                                                      "enabled_routes", "dispatch"}:
            raise ConfigurationError(
                "CHORD_CONFIG supports version, endpoints, routing, dispatch, image_workflows_dir and enabled_routes")
        enabled = raw.get("enabled_routes", [])
        if not isinstance(enabled, list) or not all(isinstance(r, str) and r.strip() for r in enabled):
            raise ConfigurationError("enabled_routes must be a list of route names")
        version = raw.get("version", 1)  # versionless files from #379 are v1
        if type(version) is not int or version not in (1, 2):
            raise ConfigurationError("CHORD_CONFIG version must be 1 or 2")
        endpoints = raw.get("endpoints")
        routing = raw.get("routing", {})
        if not isinstance(endpoints, dict) or not isinstance(routing, dict):
            raise ConfigurationError("endpoints and routing must be mappings")

        def interpolate(value: object, label: str) -> str:
            if not isinstance(value, str):
                raise ConfigurationError(f"{label} must be a string")
            def lookup(match: re.Match[str]) -> str:
                name = match.group(1)
                if name not in os.environ:
                    raise ConfigurationError(f"{label} needs environment variable {name}")
                return os.environ[name]
            expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", lookup, value)
            if "${" in expanded:
                raise ConfigurationError(f"{label} has unsupported environment interpolation")
            return expanded

        resolved: dict[str, dict] = {}
        types = {"openai-chat", "openai-audio", "openai-embeddings", "tei-embeddings", "comfyui"}
        for name, item in endpoints.items():
            if not isinstance(name, str) or not isinstance(item, dict):
                raise ConfigurationError("each endpoint must be a named mapping")
            if set(item) - {"type", "url", "model", "auth", "thinking", "json_mode"}:
                raise ConfigurationError(f"endpoint {name!r} has unsupported fields")
            kind = item.get("type")
            if kind not in types:
                raise ConfigurationError(f"endpoint {name!r} needs a supported type: {sorted(types)}")
            url = interpolate(item.get("url", ""), f"endpoints.{name}.url")
            model = interpolate(item.get("model", ""), f"endpoints.{name}.model")
            if not url.strip() or (kind != "comfyui" and not model.strip()):
                raise ConfigurationError(f"endpoint {name!r} needs a nonblank URL and model")
            auth = item.get("auth", "")
            if auth is None:
                auth = ""
            basic = ""
            if isinstance(auth, dict):
                if set(auth) != {"basic"} or kind not in {"openai-embeddings", "tei-embeddings"}:
                    raise ConfigurationError(f"endpoint {name!r} has unsupported auth")
                basic = interpolate(auth["basic"], f"endpoints.{name}.auth.basic")
                auth = ""
            auth = interpolate(auth, f"endpoints.{name}.auth")
            thinking = item.get("thinking", "passthrough")
            if thinking not in {"passthrough", "qwen_chat_template"}:
                raise ConfigurationError(f"endpoint {name!r} thinking must be passthrough or qwen_chat_template")
            json_mode = item.get("json_mode", False)
            if not isinstance(json_mode, bool):
                raise ConfigurationError(f"endpoint {name!r} json_mode must be true or false")
            if kind != "openai-chat" and ("thinking" in item or "json_mode" in item):
                raise ConfigurationError(f"endpoint {name!r} cannot set chat backend options on {kind}")
            resolved[name] = dict(type=kind, url=url, model=model, auth=auth,
                                  basic=basic, thinking=thinking, json_mode=json_mode)

        def selected(route: str, default: str, allowed: set[str]) -> dict | None:
            spec = routing.get(route, {"endpoint": default})
            if not isinstance(spec, dict) or set(spec) != {"endpoint"}:
                raise ConfigurationError(f"routing.{route} must contain only endpoint")
            name = spec["endpoint"]
            if name is None:
                return None
            endpoint = resolved.get(name)
            if endpoint is None:
                raise ConfigurationError(f"routing.{route} names unknown endpoint {name!r}")
            if endpoint["type"] not in allowed:
                raise ConfigurationError(f"routing.{route} cannot use endpoint type {endpoint['type']!r}")
            return endpoint

        # One form per file (contract: docs/configure.md). Version 1 is the original
        # form, loaded exactly as before: `router` is the helper and also answers the
        # fast tier, and dispatch comes from the environment. Version 2 names each job:
        # `helper` and `fast` each default to `chat` on their own, and dispatch is the
        # file's `dispatch` block. Neither form accepts the other's names.
        writers = {"chat", "router"} if version == 1 else {"chat", "helper", "fast"}
        for name in ({"helper", "fast"} if version == 1 else {"router"}) & set(routing):
            raise ConfigurationError(f"routing.{name} needs version: 2" if version == 1 else
                                     "routing.router is the version 1 name; version 2 uses routing.helper")
        if version == 1 and "dispatch" in raw:
            raise ConfigurationError("dispatch needs version: 2")
        if set(routing) - writers - {"stt", "tts", "embeddings", "video", "image",
                                     "image_edit", "image_variation"}:
            raise ConfigurationError("routing has unsupported route names")
        chat = selected("chat", "chat", {"openai-chat"})
        chat_name = routing.get("chat", {"endpoint": "chat"})["endpoint"]
        helper_key = "router" if version == 1 else "helper"
        router = selected(helper_key, chat_name, {"openai-chat"})
        fast = selected("fast", chat_name, {"openai-chat"}) if version == 2 else router
        if chat is None or router is None or fast is None:
            raise ConfigurationError(f"chat and {helper_key} routes must have endpoints"
                                     if version == 1 else "chat, helper and fast routes must have endpoints")
        dispatch = _dispatch(raw.get("dispatch"), interpolate) if version == 2 else None
        stt = selected("stt", "stt", {"openai-audio"}) if "stt" in routing or "stt" in resolved else None
        tts = selected("tts", "tts", {"openai-audio"}) if "tts" in routing or "tts" in resolved else None
        embeddings = (selected("embeddings", "embeddings", {"openai-embeddings", "tei-embeddings"})
                      if "embeddings" in routing or "embeddings" in resolved else None)
        video = None
        video_workflows: dict[str, VideoWorkflowConfig] | None = None
        def workflow_route(name: str, spec: object, required: tuple[str, ...],
                           optional: tuple[str, ...], *,
                           endpoint_name: str | None = None,
                           metadata: tuple[str, ...] = ()) -> tuple[dict, Path, dict]:
            """The checks every operator-workflow ComfyUI route shares.

            `name` is the dotted label after `routing.` (e.g. "image", or a
            nested map such as "video.t2v"); `spec` is that mapping. Returns
            the comfyui endpoint, the resolved image_workflows_dir and the
            mapping, after the endpoint, auth, path and binding checks.
            `endpoint_name` is an endpoint inherited from an enclosing map
            (routing.video.endpoint for its t2v and r2v maps); a spec that is
            given one must not name its own."""
            needs = ", ".join(required[:-1]) + " and " + required[-1]
            if (not isinstance(spec, dict) or not set(required) <= set(spec)
                    or set(spec) - set(required) - set(optional) - set(metadata)):
                raise ConfigurationError(f"routing.{name} needs {needs}")
            if endpoint_name is not None:
                if "endpoint" in spec:
                    raise ConfigurationError(f"routing.{name} inherits its endpoint and must not set one")
            else:
                endpoint_name = spec["endpoint"]
            if not isinstance(endpoint_name, str):
                raise ConfigurationError(f"routing.{name}.endpoint must name a comfyui endpoint")
            endpoint = resolved.get(endpoint_name)
            if endpoint is None or endpoint["type"] != "comfyui":
                raise ConfigurationError(f"routing.{name} needs a configured comfyui endpoint")
            if endpoint["auth"] or endpoint["basic"]:
                raise ConfigurationError(f"routing.{name} ComfyUI endpoint does not support auth")
            try:
                has_userinfo = urlparse(endpoint["url"]).username is not None
            except ValueError:
                has_userinfo = False  # URL validation below reports the malformed authority.
            if has_userinfo:
                raise ConfigurationError(f"routing.{name} ComfyUI endpoint URL must not contain userinfo")
            directory_value = raw.get("image_workflows_dir")
            if not isinstance(directory_value, str) or not directory_value.strip():
                raise ConfigurationError(f"routing.{name} needs image_workflows_dir")
            directory = Path(interpolate(directory_value, "image_workflows_dir"))
            if not directory.is_absolute():
                directory = path.parent / directory
            workflow_value = spec["workflow"]
            if not isinstance(workflow_value, str) or not workflow_value or Path(workflow_value).is_absolute():
                raise ConfigurationError(f"routing.{name}.workflow must be a relative path")
            bindings = [key for key in (*required, *optional) if key not in ("endpoint", "workflow")]
            if any(not isinstance(spec[key], str) or not spec[key] for key in required
                   if key not in ("endpoint", "workflow")) or any(
                       spec.get(key) is not None and (not isinstance(spec.get(key), str) or not spec.get(key))
                       for key in bindings
                   ):
                raise ConfigurationError(f"routing.{name} node and input bindings must be nonblank strings")
            return endpoint, directory, spec

        if "video" in routing:
            route = routing["video"]
            if (not isinstance(route, dict) or not {"endpoint", "t2v"} <= set(route)
                    or set(route) - {"endpoint", "t2v", "r2v"}):
                raise ConfigurationError("routing.video needs endpoint and t2v workflow; r2v is optional")
            endpoint_name = route["endpoint"]
            if not isinstance(endpoint_name, str):
                raise ConfigurationError("routing.video.endpoint must name a comfyui endpoint")
            video_workflows = {}
            for kind in ("t2v", "r2v"):
                if kind not in route:
                    continue
                required = ("workflow", "output_node_id", "prompt_node_id", "width_node_id",
                            "height_node_id", "duration_node_id", "seed_node_id")
                if kind == "r2v":
                    required += ("reference_node_id",)
                optional = tuple(f"{name}_input_name" for name in
                                 ("prompt", "width", "height", "duration", "seed"))
                if kind == "r2v":
                    optional += ("reference_input_name",)
                video, directory, spec = workflow_route(
                    f"video.{kind}", route[kind], required, optional,
                    endpoint_name=endpoint_name, metadata=("duration_unit", "fps"))
                try:
                    bindings, output, unit, fps = parse_bindings(spec, reference=kind == "r2v")
                except VideoWorkflowError as exc:
                    raise ConfigurationError(f"routing.video.{kind}: {exc}") from None
                video_workflows[kind] = VideoWorkflowConfig(
                    path=directory / spec["workflow"], directory=directory,
                    output_node_id=output, bindings=bindings, duration_unit=unit, fps=fps,
                )

        image = None
        image_workflow = None
        if "image" in routing:
            image, directory, spec = workflow_route(
                "image", routing["image"], ("endpoint", "workflow", "prompt_node_id", "output_node_id"),
                ("prompt_input_name", "seed_node_id", "seed_input_name"))
            image_workflow = ImageWorkflowConfig(
                path=directory / spec["workflow"], directory=directory,
                prompt_node_id=spec["prompt_node_id"], output_node_id=spec["output_node_id"],
                prompt_input_name=spec.get("prompt_input_name", "text"),
                seed_node_id=spec.get("seed_node_id"),
                seed_input_name=spec.get("seed_input_name", "seed"),
            )
        input_routes: dict[str, tuple[dict, ImageInputWorkflowConfig]] = {}
        for name, required, optional in (
            ("image_edit", ("endpoint", "workflow", "image_node_id", "prompt_node_id", "output_node_id"),
             ("image_input_name", "prompt_input_name", "seed_node_id", "seed_input_name")),
            # A variation has no prompt in the OpenAI contract, so it has no prompt binding.
            ("image_variation", ("endpoint", "workflow", "image_node_id", "output_node_id"),
             ("image_input_name", "seed_node_id", "seed_input_name")),
        ):
            if name not in routing:
                continue
            endpoint, directory, spec = workflow_route(name, routing[name], required, optional)
            input_routes[name] = (endpoint, ImageInputWorkflowConfig(
                path=directory / spec["workflow"], directory=directory,
                image_node_id=spec["image_node_id"], output_node_id=spec["output_node_id"],
                image_input_name=spec.get("image_input_name", "image"),
                prompt_node_id=spec.get("prompt_node_id"),
                prompt_input_name=spec.get("prompt_input_name", "text"),
                seed_node_id=spec.get("seed_node_id"),
                seed_input_name=spec.get("seed_input_name", "seed"),
            ))
        if "image_workflows_dir" in raw and image is None and not input_routes and video_workflows is None:
            raise ConfigurationError("image_workflows_dir requires a ComfyUI workflow route")
        base = cls()
        return replace(base,
            # A YAML install is self-contained: stale env must not enable a
            # specialist or override endpoint auth. The file enables its own.
            enabled_routes=frozenset(r.strip() for r in enabled),
            persona_model=chat["model"], persona_base_url=chat["url"], persona_api_key=chat["auth"],
            persona_thinking_mode=chat["thinking"],
            router_model=router["model"], router_base_url=router["url"], router_api_key=router["auth"],
            router_thinking_mode=router["thinking"], router_json_mode=router["json_mode"],
            config_version=version,
            helper_explicit=helper_key in routing,
            **({"fast_model": fast["model"], "fast_base_url": fast["url"], "fast_api_key": fast["auth"],
                "fast_thinking_mode": fast["thinking"], "fast_explicit": "fast" in routing,
                **dispatch} if version == 2 else {}),
            stt_model=stt["model"] if stt else "", stt_base_url=stt["url"] if stt else "",
            stt_api_key=stt["auth"] if stt else "",
            tts_model=tts["model"] if tts else "", tts_base_url=tts["url"] if tts else "",
            tts_api_key=tts["auth"] if tts else "",
            embeddings_model=embeddings["model"] if embeddings else "",
            embeddings_base_url=embeddings["url"] if embeddings else "",
            embeddings_basic_auth=embeddings["basic"] if embeddings else "",
            embeddings_api_key=embeddings["auth"] if embeddings else "",
            comfy_base_url=video["url"] if video else "",
            video_workflows=video_workflows,
            image_comfy_base_url=image["url"] if image else "",
            image_workflow=image_workflow,
            image_edit_comfy_base_url=input_routes["image_edit"][0]["url"] if "image_edit" in input_routes else "",
            image_edit_workflow=input_routes["image_edit"][1] if "image_edit" in input_routes else None,
            image_variation_comfy_base_url=(input_routes["image_variation"][0]["url"]
                                            if "image_variation" in input_routes else ""),
            image_variation_workflow=(input_routes["image_variation"][1]
                                      if "image_variation" in input_routes else None),
        )

    def validate_startup(self) -> None:
        """Validate the production process boundary before opening sockets.

        Unit tests may construct deliberately partial settings while injecting
        fake dependencies. The executable calls this method before constructing
        real clients, so a deployment never discovers configuration defects on
        its first user request.
        """
        errors: list[str] = []

        from . import registry  # registry imports nothing from config; deferred to keep import order flat
        unknown = sorted(self.enabled_routes - set(registry.load()))
        if unknown:
            errors.append(f"enabled routes are not in the registry: {', '.join(unknown)}")

        clients: tuple[tuple[str, str], ...] = ()
        try:
            clients = parse_client_keys(self.client_keys_json, self.service_api_key)
        except ValueError as exc:
            errors.append(str(exc))
        has_auth = bool(clients or (self.service_api_key and self.accept_legacy_client_key))
        if not has_auth and self.public_host not in {"127.0.0.1", "localhost", "::1"}:
            errors.append("CHORD_API_KEY or CHORD_CLIENT_KEYS_JSON is required when PUBLIC_HOST is not loopback")
        if self.public_artifact_base and not self.artifact_signer_key:
            errors.append("PUBLIC_ARTIFACT_BASE requires CHORD_ARTIFACT_SIGNING_KEY or CHORD_API_KEY")
        if self.artifact_signing_key:
            if len(self.artifact_signing_key) < 32:
                errors.append("CHORD_ARTIFACT_SIGNING_KEY must have at least 32 characters")
            if self.artifact_signing_key == self.service_api_key or any(
                self.artifact_signing_key == token for _, token in clients
            ):
                errors.append("CHORD_ARTIFACT_SIGNING_KEY must differ from every bearer key")
        if self.legacy_artifact_verify_until < 0:
            errors.append("CHORD_LEGACY_ARTIFACT_VERIFY_UNTIL must be a nonnegative Unix timestamp")
        if self.legacy_artifact_verify_until > time.time() + self.artifact_url_ttl_s:
            errors.append("CHORD_LEGACY_ARTIFACT_VERIFY_UNTIL exceeds one artifact link lifetime")
        if self.legacy_artifact_verify_until and not (self.artifact_signing_key and self.service_api_key):
            errors.append("CHORD_LEGACY_ARTIFACT_VERIFY_UNTIL needs both signing and legacy keys")

        from . import manifest  # deferred like registry above
        bad = sorted({str(r.get("slot")) for r in manifest.load()["service_tier_routes"].values()} - TIER_SLOTS)
        if bad:
            errors.append(f"manifest service_tier_routes has unknown slots: {', '.join(bad)}; "
                          f"expected {', '.join(sorted(TIER_SLOTS))}")
        for slot in ("persona", "router", "fast"):
            try:
                self.slot_target(slot)
            except ValueError as exc:
                errors.append(str(exc))
        seen: dict[str, str] = {}
        for model, url in ((self.persona_model, self.persona_base_url), (self.router_model, self.router_base_url),
                           (self.fast_model, self.fast_base_url)):
            if model and url and seen.setdefault(model, url.rstrip("/")) != url.rstrip("/"):
                errors.append(f"chat model {model!r} cannot identify two different direct routes")
                break

        urls = {
            "PERSONA_BASE_URL": self.persona_base_url,
            "ROUTER_BASE_URL": self.router_base_url,
            "ROUTER_CLASSIFIER_URL": self.router_classifier_url,
            "STT_BASE_URL": self.stt_base_url,
            "TTS_BASE_URL": self.tts_base_url,
            "EMBEDDINGS_BASE_URL": self.embeddings_base_url,
            "COMFY_BASE_URL": self.comfy_base_url,
            "IMAGE_COMFY_BASE_URL": self.image_comfy_base_url,
            "routing.image_edit endpoint": self.image_edit_comfy_base_url,
            "routing.image_variation endpoint": self.image_variation_comfy_base_url,
            "PUBLIC_ARTIFACT_BASE": self.public_artifact_base,
        }
        for name, value in urls.items():
            if not value:
                continue
            try:
                parsed = urlparse(value)
                valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                # Accessing .port validates the spelling and range; a netloc
                # like host:notaport otherwise passes and reaches the client.
                if valid:
                    _ = parsed.port
            except ValueError:
                valid = False
            if not valid:
                errors.append(f"{name} must be an absolute http(s) URL")

        for name, value in (("PUBLIC_PORT", self.public_port), ("INTERNAL_PORT", self.internal_port)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                errors.append(f"{name} must be an integer from 1 through 65535")
        if self.public_port == self.internal_port:
            errors.append("PUBLIC_PORT and INTERNAL_PORT must differ")
        if not _positive_finite(self.router_timeout_s):
            errors.append("ROUTER_TIMEOUT_S must be greater than zero and finite")
        if self.router_backend not in {"llm", "classifier"}:
            errors.append("ROUTER_BACKEND must be llm or classifier")
        for name, mode in (("PERSONA_THINKING_MODE", self.persona_thinking_mode),
                           ("ROUTER_THINKING_MODE", self.router_thinking_mode),
                           *((("routing.fast thinking", self.fast_thinking_mode),) if self.fast_thinking_mode else ())):
            if mode not in {"passthrough", "qwen_chat_template"}:
                errors.append(f"{name} must be passthrough or qwen_chat_template")
        if self.router_backend == "classifier" and not self.router_classifier_url:
            errors.append("ROUTER_BACKEND=classifier needs ROUTER_CLASSIFIER_URL")
        if not _positive_finite(self.router_classifier_timeout_s):
            errors.append("ROUTER_CLASSIFIER_TIMEOUT_S must be greater than zero and finite")
        if not _positive_finite(self.video_timeout_s):
            errors.append("VIDEO_TIMEOUT_S must be greater than zero and finite")
        if not _positive_finite(self.image_deadline_s):
            errors.append("IMAGE_DEADLINE_S must be greater than zero and finite")
        if not _positive_finite(self.image_prompt_timeout_s):
            errors.append("IMAGE_PROMPT_TIMEOUT_S must be greater than zero and finite")
        if not _positive_finite(self.artifact_url_ttl_s):
            errors.append("ARTIFACT_URL_TTL_S must be greater than zero and finite")
        if self.retention_days < 0:
            errors.append("CHORD_RETENTION_DAYS must be 0 (keep forever) or greater")
        elif self.retention_days > 0 and self.retention_days * 86400 < self.artifact_url_ttl_s:
            # Signed links must not outlive the bytes they point to: sweeping
            # artifacts while a live link still names one turns a delivered
            # URL into a 404 inside its own window (batch 2).
            errors.append("CHORD_RETENTION_DAYS must cover ARTIFACT_URL_TTL_S, or signed links outlive the bytes")
        if self.max_json_body_bytes < 0:
            errors.append("MAX_JSON_BODY_BYTES must be 0 (no cap) or greater")
        for name, value in (
            ("CHORD_MAX_GLOBAL_INFLIGHT", self.max_global_inflight),
            ("CHORD_MAX_CLIENT_INFLIGHT", self.max_client_inflight),
            ("CHORD_MAX_GLOBAL_PER_MINUTE", self.max_global_per_minute),
            ("CHORD_MAX_CLIENT_PER_MINUTE", self.max_client_per_minute),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{name} must be a nonnegative integer")
        for host in self.image_url_allowed_hosts:
            if "://" in host or "/" in host:
                errors.append(
                    f"IMAGE_URL_ALLOWED_HOSTS entries must be host or host:port, got {host!r}")

        try:
            voices = json.loads(self.tts_voices)
            if not isinstance(voices, dict) or not all(
                isinstance(name, str) and name and isinstance(voice, str) and voice
                for name, voice in voices.items()
            ):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            errors.append("TTS_VOICES must be a JSON object of non-empty string names to voices")

        if self.embeddings_basic_auth:
            if not self.embeddings_base_url:
                errors.append("EMBEDDINGS_BASIC_AUTH requires EMBEDDINGS_BASE_URL")
            if ":" not in self.embeddings_basic_auth:
                errors.append("EMBEDDINGS_BASIC_AUTH must use user:password form")
        if self.embeddings_api_key:
            if not self.embeddings_base_url:
                errors.append("EMBEDDINGS_API_KEY requires EMBEDDINGS_BASE_URL")
            if self.embeddings_basic_auth:
                errors.append("EMBEDDINGS_API_KEY and EMBEDDINGS_BASIC_AUTH cannot both be set")

        for url in {value for value in urls.values() if value}:
            try:
                self.credential_for(url)
            except ValueError as exc:
                errors.append(str(exc))

        if self.image_workflow is not None:
            if not self.image_comfy_base_url:
                errors.append("image workflow requires an image ComfyUI endpoint")
            try:
                self.image_workflow.load()
            except WorkflowError as exc:
                errors.append(str(exc))
        for label, workflow, url in (
            ("image edit", self.image_edit_workflow, self.image_edit_comfy_base_url),
            ("image variation", self.image_variation_workflow, self.image_variation_comfy_base_url),
        ):
            if workflow is None:
                continue
            if not url:
                errors.append(f"{label} workflow requires a ComfyUI endpoint")
            try:
                workflow.load()
            except WorkflowError as exc:
                errors.append(f"{label}: {exc}")

        if self.video_workflows is not None:
            if not self.comfy_base_url:
                errors.append("video workflows require a ComfyUI endpoint")
            for workflow in self.video_workflows.values():
                try:
                    workflow.load()
                except VideoWorkflowError as exc:
                    errors.append(str(exc))

        if errors:
            detail = "\n".join(f"- {message}" for message in dict.fromkeys(errors))
            raise ConfigurationError(f"invalid startup configuration:\n{detail}")

    def slot_target(self, slot: str) -> tuple[str, str]:
        """Resolve a logical chat slot to its configured direct backend.

        Service-tier policy names roles, never deployment model literals.  Both
        halves are required: a missing model or URL fails before serving
        requests instead of silently choosing another endpoint.
        """
        targets = {
            "persona": (self.persona_model, self.persona_base_url),
            "router": (self.router_model, self.router_base_url),
            "fast": (self.fast_model or self.router_model, self.fast_base_url or self.router_base_url),
        }
        if slot not in targets:
            raise ValueError(f"unknown chat slot {slot!r}; expected one of {sorted(targets)}")
        model, base_url = targets[slot]
        if not model.strip() or not base_url.strip():
            raise ValueError(f"chat slot {slot!r} requires a nonblank model and direct base URL")
        return model, base_url

    def tier_model(self, slot: str) -> str:
        """The model a manifest service tier selects. `fast` is the fast writer; a
        manifest written before version 2 says `router` for the same thing."""
        if slot not in TIER_SLOTS:
            raise ValueError(f"service tier slot {slot!r}; expected one of {sorted(TIER_SLOTS)}")
        return self.slot_target("persona" if slot == "persona" else "fast")[0]

    def reply_thinking_mode(self, model: str) -> str:
        """The thinking mode for a reply written by `model`. In a version 2 config the
        fast endpoint owns its own; everywhere else every reply uses the persona's,
        as before."""
        if (self.config_version == 2 and self.fast_thinking_mode and model == self.fast_model
                and model != self.persona_model):
            return self.fast_thinking_mode
        return self.persona_thinking_mode

    def base_url_for(self, model: str) -> str:
        """The explicitly configured endpoint that serves `model`.

        Resolved by model NAME rather than by call site, because the router and the
        persona are different models on different hosts and the same factory builds
        clients for both."""
        if model == self.persona_model and self.persona_base_url:
            return self.persona_base_url
        if model == self.router_model and self.router_base_url:
            return self.router_base_url
        if model == self.fast_model and self.fast_base_url:
            return self.fast_base_url
        if model == self.stt_model and self.stt_base_url:
            return self.stt_base_url
        if model == self.tts_model and self.tts_base_url:
            return self.tts_base_url
        if model == self.embeddings_model and self.embeddings_base_url:
            return self.embeddings_base_url
        raise ValueError(f"model {model!r} has no configured backend")

    def credential_for(self, base_url: str) -> str:
        """The credential to present to `base_url`, or "" for none at all.

        Resolved by ADDRESS, not by call site. Each backend receives only its
        own configured credential; an uncredentialed backend gets no
        Authorization header.
        """
        if not base_url:
            return ""
        url = base_url.rstrip("/")
        matched = {key for configured, key in ((self.persona_base_url, self.persona_api_key),
                                               (self.router_base_url, self.router_api_key),
                                               (self.fast_base_url, self.fast_api_key),
                                               (self.stt_base_url, self.stt_api_key),
                                               (self.tts_base_url, self.tts_api_key))
                   if configured and configured.rstrip("/") == url}
        # Two slots may share an address; they may not disagree about its credential.
        # Returning the first match made the answer depend on the order of this tuple,
        # which is not a decision -- and the caller could not tell it had been made.
        distinct = {k for k in matched if k}
        if len(distinct) > 1:
            raise ValueError(f"address {url} is configured with two different credentials")
        return next(iter(distinct), "")

    def voice_for(self, persona_id: str) -> str | None:
        return json.loads(self.tts_voices).get(persona_id)

    # Bearer key a client must present to Chord.
    # Empty means "no auth", which is allowed only on a loopback bind (checked at startup).
    service_api_key: str = field(default_factory=lambda: _env("CHORD_API_KEY", ""), repr=False)
    # Additional direct-client keys, id -> random bearer token. They identify
    # callers for revocation and admission, while stored data remains shared.
    # Empty keeps the single-key deployment's behavior during rollout.
    client_keys_json: str = field(default_factory=lambda: _env("CHORD_CLIENT_KEYS_JSON", ""), repr=False)
    # New browser-link signatures use a key separate from bearer authentication.
    # The old service key may verify pre-cutover links until an absolute deadline,
    # at most one link lifetime ahead when configured.
    artifact_signing_key: str = field(default_factory=lambda: _env("CHORD_ARTIFACT_SIGNING_KEY", ""), repr=False)
    accept_legacy_client_key: bool = field(default_factory=lambda: _strict_bool_env("CHORD_ACCEPT_LEGACY_CLIENT_KEY", "true"))
    legacy_artifact_verify_until: int = field(default_factory=lambda: int(_env("CHORD_LEGACY_ARTIFACT_VERIFY_UNTIL", "0")))

    @property
    def artifact_signer_key(self) -> str:
        """The key used for newly minted links during and after migration."""
        return self.artifact_signing_key or self.service_api_key
    persona_model: str = field(default_factory=lambda: _env("PERSONA_MODEL", ""))
    router_model: str = field(default_factory=lambda: _env("ROUTER_MODEL", ""))
    # Both env-only and YAML installs use portable OpenAI-style passthrough.
    # Qwen chat-template thinking and router JSON mode are explicit options.
    persona_thinking_mode: str = field(default_factory=lambda: _env("PERSONA_THINKING_MODE", "passthrough"))
    router_thinking_mode: str = field(default_factory=lambda: _env("ROUTER_THINKING_MODE", "passthrough"))
    router_json_mode: bool = field(default_factory=lambda: _strict_bool_env("ROUTER_JSON_MODE", "false"))
    # Where each upstream model actually lives. Endpoints and their credentials
    # are selected per backend; no intermediate gateway is required.
    # Persona and router have no fallback: `slot_target` refuses a blank model or base
    # URL, so a deployment without PERSONA_BASE_URL or ROUTER_BASE_URL fails at startup.
    # STT and TTS without direct URLs refuse with 503; embeddings without a URL are
    # not served.
    #
    # A credential belongs to one backend, not to the whole service. Reusing a
    # gateway credential across unrelated direct backends can expose it to those
    # servers' request logs.
    # Each defaults to empty, which means NO Authorization header at all for that
    # backend -- not an empty bearer, no header. The key follows the URL.

    persona_api_key: str = field(default_factory=lambda: _env("PERSONA_API_KEY", ""))
    # Version 2 config only (no environment variables). The model that answers a
    # `service_tier: fast`/`priority` turn and writes that turn's image prompt.
    # Blank means the router model, which is the version 1 and env-only behaviour;
    # a blank thinking mode means persona_thinking_mode, also as before.
    fast_model: str = ""
    fast_base_url: str = ""
    fast_api_key: str = ""
    fast_thinking_mode: str = ""
    # 0: env-only install; 1 or 2: the CHORD_CONFIG form. Role policy that
    # differs by form (search query writer, endpoint-owned thinking) reads this.
    config_version: int = 0
    helper_explicit: bool = False
    fast_explicit: bool = False
    router_api_key: str = field(default_factory=lambda: _env("ROUTER_API_KEY", ""))
    stt_api_key: str = field(default_factory=lambda: _env("STT_API_KEY", ""))
    tts_api_key: str = field(default_factory=lambda: _env("TTS_API_KEY", ""))
    persona_base_url: str = field(default_factory=lambda: _env("PERSONA_BASE_URL", ""))
    router_base_url: str = field(default_factory=lambda: _env("ROUTER_BASE_URL", ""))
    stt_base_url: str = field(default_factory=lambda: _env("STT_BASE_URL", ""))
    tts_base_url: str = field(default_factory=lambda: _env("TTS_BASE_URL", ""))
    # Embeddings is a separate service in this fleet (text-embeddings-inference),
    # not the chat model. Whether this points at the authenticated shared-inference
    # gateway or straight at TEI is a DEPLOYMENT decision, not an architectural one:
    # both are a base URL here, and neither needs a code change. The URL may
    # end at the service root or /v1; the embeddings client sends one /v1 only.
    embeddings_base_url: str = field(default_factory=lambda: _env("EMBEDDINGS_BASE_URL", ""))
    # The shared-inference ingress authenticates with Basic, not the LiteLLM
    # Bearer. Supplied as user:pass and encoded once at startup; empty means the
    # default Bearer, which is correct for a deployment that fronts TEI itself.
    embeddings_basic_auth: str = field(default_factory=lambda: _env("EMBEDDINGS_BASIC_AUTH", ""))
    embeddings_api_key: str = field(default_factory=lambda: _env("EMBEDDINGS_API_KEY", ""), repr=False)

    # ComfyUI renders video, and will render music next. It is a graph executor
    # rather than an OpenAI-compatible backend, so it resolves by ADDRESS and
    # never through `base_url_for`, which answers questions about model names.
    # Empty means video create returns 503.
    # List, retrieve, delete, and content stay up, because those read the data
    # volume and do not need a GPU.
    comfy_base_url: str = field(default_factory=lambda: _env("COMFY_BASE_URL", ""))
    image_comfy_base_url: str = ""
    image_workflow: ImageWorkflowConfig | None = None
    # Edits and variations run an operator's workflow too; none ships with Chord.
    # Unset (and always in environment mode) means those two doors answer 503.
    image_edit_comfy_base_url: str = ""
    image_edit_workflow: ImageInputWorkflowConfig | None = None
    image_variation_comfy_base_url: str = ""
    image_variation_workflow: ImageInputWorkflowConfig | None = None
    video_workflows: dict[str, VideoWorkflowConfig] | None = None
    # A render occupies the GPU for minutes, not seconds; the default HTTP
    # timeout would abandon a job that is going to succeed and leave it running.
    video_timeout_s: float = field(default_factory=lambda: float(_env("VIDEO_TIMEOUT_S", "1800")))
    # The image provider deadline covers lock wait, submission and rendering.
    # A render that exceeds it fails before an HTTP caller waits indefinitely.
    image_deadline_s: float = field(default_factory=lambda: float(_env("IMAGE_DEADLINE_S", "1500")))
    # Bound on the chat model writing an image prompt. Past it, the user's own
    # last message is the prompt.
    image_prompt_timeout_s: float = field(default_factory=lambda: float(_env("IMAGE_PROMPT_TIMEOUT_S", "60")))

    def embeddings_auth_header(self) -> str:
        """`Authorization` for the embeddings upstream, or "" to inherit the Bearer."""
        if self.embeddings_api_key:
            return "Bearer " + self.embeddings_api_key
        if not self.embeddings_basic_auth:
            return ""
        import base64
        return "Basic " + base64.b64encode(self.embeddings_basic_auth.encode()).decode()
    # Speech-to-text for input_audio parts.
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", ""))
    # Text-to-speech for modalities ["text","audio"]. The configured persona's
    # voice comes from this map, never from an untrusted request override.
    # The portable default maps generic to alloy; operators may replace it.
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", ""))
    # The one model /v1/embeddings serves. The OpenAI request still has to name it.
    embeddings_model: str = field(default_factory=lambda: _env("EMBEDDINGS_MODEL", ""))
    tts_voices: str = field(default_factory=lambda: _env("TTS_VOICES", '{"generic": "alloy"}'))
    # Web search (#81): Brave's API when keyed, DuckDuckGo's HTML page otherwise.
    brave_api_key: str = field(default_factory=lambda: _env("BRAVE_API_KEY", ""))
    # M1 proves the plain pipe first: with the router off, every turn is chat.
    router_enabled: bool = field(default_factory=lambda: _env("ROUTER_ENABLED", "false").lower() == "true")
    # The most a turn waits for the router before it falls back to plain chat.
    # The router normally answers in 1-2 s.
    router_timeout_s: float = field(default_factory=lambda: float(_env("ROUTER_TIMEOUT_S", "10")))
    # Who picks the lane. "llm": the router model. "classifier": an operator-supplied HTTP
    # classifier at ROUTER_CLASSIFIER_URL (contract: docs/architecture.md, Router backends).
    # It returns a lane only; on a specialist lane the router model still writes the brief.
    router_backend: str = field(default_factory=lambda: _env("ROUTER_BACKEND", "llm").strip().lower())
    router_classifier_url: str = field(default_factory=lambda: _env("ROUTER_CLASSIFIER_URL", ""))
    # Past this the turn is routed by the router model instead.
    router_classifier_timeout_s: float = field(default_factory=lambda: float(_env("ROUTER_CLASSIFIER_TIMEOUT_S", "1")))
    # Specialist routes the operator turns on (search, audio, video). A route
    # runs when it is enabled here and its backend is configured; a registry
    # `certified` record is reported, never required. Image is enabled by
    # configuring an image workflow. EXPERIMENTAL_ROUTES is the earlier name.
    enabled_routes: frozenset = field(default_factory=lambda: frozenset(
        r.strip() for r in (_env("ENABLED_ROUTES") if "ENABLED_ROUTES" in os.environ
                            else _env("EXPERIMENTAL_ROUTES", "")).split(",")
        if r.strip()))  # an explicitly empty ENABLED_ROUTES enables nothing
    # The http(s) hosts a chat image_url part may name (review 2026-09-22, #7).
    # The persona BACKEND fetches image_url parts itself, from its own position
    # on the VLAN, so every http URL a caller can name is a fetch primitive
    # aimed at the GPU host's network. data:image URIs are always allowed --
    # that is what vision input was certified with (evidence/vision-input) --
    # and http(s) is allowed only against this exact-host list. Entries are
    # `host` or `host:port`, never a scheme or a path; a bare host answers only
    # URLs with no explicit port, because a host entry that allowed every port
    # would allow every service on that host. Default empty: data-only.
    image_url_allowed_hosts: frozenset = field(default_factory=lambda: frozenset(
        h.strip().lower() for h in _env("IMAGE_URL_ALLOWED_HOSTS", "").split(",") if h.strip()))
    # Where a browser can load a generated image (a public route to this
    # service's /v1/artifacts). Unset: images are delivered as data: URIs.
    public_artifact_base: str = field(default_factory=lambda: _env("PUBLIC_ARTIFACT_BASE", ""))
    artifact_url_ttl_s: int = field(default_factory=lambda: int(_env("ARTIFACT_URL_TTL_S", str(7 * 24 * 3600))))
    data_dir: Path = field(default_factory=lambda: Path(_env("CHORD_DATA_DIR", "./var")))
    # One window bounds volume growth (review 2026-09-22). The sqlite stores
    # keep their own retention (30-day chats/responses, 24h spec videos,
    # expires_at files); this sweeps what nothing swept: the daily trace JSONL,
    # the artifact store, and failed renders' attempt directories. Triggered by
    # the first trace write of each UTC day. 0 keeps all three, for an
    # evidence-preservation deployment; videos still expire on the spec's 24h
    # whatever this is (review 2026-09-24 B22).
    retention_days: int = field(default_factory=lambda: int(_env("CHORD_RETENTION_DAYS", "30")))
    # Cap on any non-multipart request body, refused BEFORE parsing and before
    # auth. Multipart routes own their own per-part caps, and /v1/files
    # legitimately needs 512 MB. 64 MiB admits one max-size image part (~34 MB
    # base64) with headroom; 0 disables the cap.
    max_json_body_bytes: int = field(default_factory=lambda: int(_env("MAX_JSON_BODY_BYTES", str(64 * 1024 * 1024))))
    # Zero leaves each cap disabled until the client-key rollout chooses
    # measured budgets. One public API worker owns these in-memory counters.
    max_global_inflight: int = field(default_factory=lambda: int(_env("CHORD_MAX_GLOBAL_INFLIGHT", "0")))
    max_client_inflight: int = field(default_factory=lambda: int(_env("CHORD_MAX_CLIENT_INFLIGHT", "0")))
    max_global_per_minute: int = field(default_factory=lambda: int(_env("CHORD_MAX_GLOBAL_PER_MINUTE", "0")))
    max_client_per_minute: int = field(default_factory=lambda: int(_env("CHORD_MAX_CLIENT_PER_MINUTE", "0")))
    public_host: str = field(default_factory=lambda: _env("PUBLIC_HOST", "127.0.0.1"))
    public_port: int = field(default_factory=lambda: int(_env("PUBLIC_PORT", "8710")))
    # The internal server is loopback-only, on its own port.
    internal_port: int = field(default_factory=lambda: int(_env("INTERNAL_PORT", "8711")))

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "traces"


def load_settings() -> Settings:
    """Resolve the same configuration for the server and deploy preflights."""
    config_path = os.environ.get("CHORD_CONFIG")
    settings = Settings.from_yaml(Path(config_path)) if config_path else Settings()
    settings.validate_startup()
    return settings
