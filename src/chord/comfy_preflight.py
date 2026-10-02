"""Deploy-time proof that the configured ComfyUI is actually there.

`VideoBackend.from_settings` treats a configured video workflow and endpoint as
available, which is all a settings object can know. It cannot tell a working
host from a decommissioned one, so an unreachable ComfyUI produces a deployment
that passes every check we have: the container is healthy, the revision matches,
the route registers, `/v1/videos` is advertised — and every call fails.  That is
strictly worse than not serving the route, because the caller was told it was
there.

This runs INSIDE the container, on purpose.  The question is not whether the
operator's laptop can reach ComfyUI; it is whether the process that will make
the call can.  Those differ for exactly the reasons deployments go wrong:
container networking, a VLAN the host is not on, an address that resolves
differently in two places.

Exit 0 means checked and reachable, or deliberately not configured.  Exit 1
means configured and unreachable, and the deploy should stop.

    docker exec chord python -m chord.comfy_preflight
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from .config import ConfigurationError, load_settings
from .videos import video_tools_available

PROBE = "/system_stats"
TIMEOUT = 10.0


def check(base_url: str, timeout: float = TIMEOUT) -> tuple[int, str]:
    """(exit code, message) for `base_url`.

    `/system_stats` rather than `/` because it answers with a JSON body a
    ComfyUI wrote — a reverse proxy, a parked page or a different service on
    that port all return 200 for `/` and would read as success.
    """
    if not base_url.strip():
        return 0, "COMFY_BASE_URL is empty: this deployment serves no video, route not advertised"

    url = base_url.rstrip("/") + PROBE
    parts = urllib.parse.urlsplit(base_url)
    display = f"{parts.scheme}://{parts.netloc.rsplit('@', 1)[-1]}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return 1, f"{display} answered {response.status}"
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return 1, f"{display} answered {exc.code}"
    except (urllib.error.URLError, OSError):
        return 1, f"{display} unreachable from inside the container"
    except (ValueError, UnicodeDecodeError):
        return 1, f"{display} answered 200 but not ComfyUI JSON"

    version = (body.get("system") or {}).get("comfyui_version")
    if not version:
        # A 200 with a JSON body is not enough: something else could be
        # listening on that port. The version field is ComfyUI identifying
        # itself, which is what we actually came to confirm.
        return 1, f"{display} answered JSON with no comfyui_version; this may not be ComfyUI"
    return 0, f"ComfyUI {version} reachable at {display}"


def main(argv: list[str] | None = None) -> int:
    try:
        settings = load_settings()
    except (ConfigurationError, RuntimeError, ValueError):
        print("comfy-preflight: invalid configuration", file=sys.stderr)
        return 1
    if settings.video_workflows is not None and not video_tools_available():
        print("comfy-preflight: video needs ffmpeg and ffprobe on PATH", file=sys.stderr)
        return 1
    urls = [settings.comfy_base_url] if settings.video_workflows is not None else []
    if settings.image_workflow is not None:
        urls.append(settings.image_comfy_base_url)
    if settings.image_edit_workflow is not None:
        urls.append(settings.image_edit_comfy_base_url)
    if settings.image_variation_workflow is not None:
        urls.append(settings.image_variation_comfy_base_url)
    if not urls:
        print("comfy-preflight: no ComfyUI workflow configured")
        return 0
    failed = False
    for base_url in dict.fromkeys(urls):
        code, message = check(base_url)
        print(f"comfy-preflight: {message}", file=sys.stderr if code else sys.stdout)
        failed |= bool(code)
    return int(failed)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
