# Chord

Chord serves an OpenAI-compatible API through one model name, `chord-1-poly`. Its text profile sends chat to an operator-chosen backend. Optional image, audio, search, and video routes need their own services and certification; the bundled configuration does not advertise them as ready.

Chord is currently a single-tenant service: one installation is one trust boundary for stored data. The [configuration example](chord.example.yaml), [architecture](docs/architecture.md), [pinned API spec](qa/conformance/spec/README.md), and [source manifest](src/chord/manifest.yaml) are the starting points for running and extending it. The manifest says what a configured instance advertises; `GET /health` reports its actual revision to a caller with a valid key.

## Text-only quickstart

You need [uv](https://docs.astral.sh/uv/), network access to PyPI for the first install, Python 3.12 (which uv can fetch), and an OpenAI-compatible chat endpoint running first. The example assumes Ollama at `http://127.0.0.1:11434/v1` with a `llama3.1` model. Change the URL and model in `chord.yaml` to match your server. Other backends may need different thinking and JSON settings; validate the configuration before serving traffic. No image, audio, search, or ComfyUI service is needed for this profile.

```sh
cp chord.example.yaml chord.yaml
uv sync --frozen --no-dev
CHORD_CONFIG=./chord.yaml uv run --no-sync python -m chord --check-config
CHORD_CONFIG=./chord.yaml uv run --no-sync python -m chord
```

Chord binds to loopback port 8710 (`PUBLIC_PORT`) by default, with a loopback-only internal port 8711 (`INTERNAL_PORT`). No API key is needed while it listens only on loopback. In another terminal:

```sh
curl http://127.0.0.1:8710/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chord-1-poly","messages":[{"role":"user","content":"hello"}]}'
```

The root [Dockerfile](Dockerfile) builds the same text-only package. In a container, the backend URL must be reachable *from inside the container*. Mount `chord.yaml` at `/app/chord.yaml`, set `CHORD_CONFIG` to that path, and set `CHORD_API_KEY` before binding beyond loopback.

Video generation requires operator-provided `ffmpeg` and `ffprobe` executables on the service's `PATH`. The public image does not bundle them. Chord refuses video creation with a 503 when either is absent, and never stores an unscrubbed render. `--show-config` and `/internal/health` report whether video can serve. The same tools make video thumbnails and spritesheets.

## Configuration

`chord.yaml` uses version 1. Its named endpoints specify a type, URL, model, and optional backend features. `routing.chat` and `routing.router` may select different endpoints. Endpoint settings such as `thinking: qwen_chat_template` and `json_mode: true` should be enabled only when the backend supports them; the default is portable passthrough. `python -m chord --check-config` validates the file without contacting a backend, and `--show-config` prints resolved routes without credentials. Lane routing can use the router model or your own HTTP classifier; see [Router backends](docs/architecture.md#router-backends).

`CHORD_MANIFEST` and `CHORD_REGISTRY` can select overlays for additional capabilities. Client keys, storage, and bind addresses are environment settings. A separate proxy may front Chord as a client routing choice; Chord does not require one.

For embeddings, `EMBEDDINGS_BASE_URL` accepts either the service root or a URL ending in `/v1`. Chord sends the request to `/v1/embeddings` once in either case.

### Turning on specialist routes

Chat needs nothing extra. Search, audio and video are specialist routes: Chord's router can hand a turn to them once
you turn them on and give them a backend. Pictures turn on by configuring an image workflow (next section).

```yaml
routing:
  chat: {endpoint: chat}
  router: {endpoint: chat}
enabled_routes: [search]
```

Without `CHORD_CONFIG`, the same list is the `ENABLED_ROUTES` environment setting (`ENABLED_ROUTES=search,audio`;
`EXPERIMENTAL_ROUTES` is the earlier name and still works). A name Chord doesn't know stops startup. Each route still
needs its backend: search uses Brave when `BRAVE_API_KEY` is set and falls back to DuckDuckGo, audio needs a `tts`
endpoint, and the router must be on (`ROUTER_ENABLED=true`). `/internal/health` lists what is enabled.

A capability in the registry may carry a `certified` record (who tested which model and prompt version, and when).
It is reported for reference and never turns a route on or off.

### ComfyUI workflows

Chord ships no image model or graph. Image generation, edits and variations each run a ComfyUI workflow you supply, saved in ComfyUI's API format under `image_workflows_dir`. For each route you name the node inputs Chord fills; the rest of the graph (model, sampler, size) is yours. Every binding is checked against the graph at startup.

```yaml
image_workflows_dir: workflows        # relative to chord.yaml, or absolute
endpoints:
  comfy: {type: comfyui, url: http://127.0.0.1:8188}
routing:
  image:                              # POST /v1/images/generations
    endpoint: comfy
    workflow: generate.json
    prompt_node_id: "6"               # prompt_input_name defaults to text
    seed_node_id: "3"                 # optional; seed_input_name defaults to seed
    output_node_id: "9"               # must save exactly one PNG
  image_edit:                         # POST /v1/images/edits
    endpoint: comfy
    workflow: edit.json
    image_node_id: "10"               # a LoadImage; image_input_name defaults to image
    prompt_node_id: "20"
    seed_node_id: "30"                # optional
    output_node_id: "40"
  image_variation:                    # POST /v1/images/variations
    endpoint: comfy
    workflow: vary.json
    image_node_id: "10"
    seed_node_id: "30"                # needed for n > 1
    output_node_id: "40"
```

A variation has no prompt, so it has no prompt binding; put any instruction in the graph. Each requested variation is one submit of the workflow with a fresh seed, so without `seed_node_id` the route returns one image and refuses `n` above 1. A route that isn't configured answers 503. Routes on the same ComfyUI address share one render lock, so they queue rather than compete for its GPU.

### Video workflows

Video uses operator-owned ComfyUI API-format workflows under `image_workflows_dir`: text-to-video (`t2v`) is required, and reference-to-video (`r2v`) is optional. A reference request refuses with 503 if `r2v` is absent. Chord ships neither a checkpoint nor a video graph. Bind each request input to a node and input name in the exported graph; the output node must save an MP4. For example:

```yaml
image_workflows_dir: /config/workflows
endpoints:
  comfy: {type: comfyui, url: 'http://comfyui:8188'}
routing:
  video:
    endpoint: comfy
    t2v:
      workflow: text-to-video.json
      output_node_id: save
      prompt_node_id: sampler
      width_node_id: sampler
      height_node_id: sampler
      duration_node_id: sampler
      seed_node_id: sampler
    r2v:
      workflow: reference-to-video.json
      output_node_id: save
      prompt_node_id: sampler
      width_node_id: sampler
      height_node_id: sampler
      duration_node_id: sampler
      seed_node_id: sampler
      reference_node_id: image_loader
```

The example is a routing excerpt; keep your chat endpoint and route alongside it. Input names default to `text`, `width`, `height`, `seconds`, `seed`, and `image`, respectively; set `<input>_input_name` beside any node ID when your graph uses another name. Chord sends the requested width, height, and seconds unchanged. To pass frame count instead, set `duration_unit: frames` and a positive integer `fps` in that workflow mapping. Model-specific size or frame-grid transforms belong in your graph. Both files are validated at startup and cannot escape the configured directory through a symlink. Video also needs operator-provided `ffmpeg` and `ffprobe` on the service `PATH` to scrub metadata and make thumbnails.

## Trust boundary

Several trusted agents and their separate conversations can share an installation. A conversation ID identifies a conversation; it is not an access boundary. Every valid client key can access stored chat completions, conversations, responses, files, and videos created through another key. List endpoints for files, videos, and stored chat completions also expose IDs across keys. Use separate installations when callers must not access each other's stored objects.

`CHORD_CLIENT_KEYS_JSON` provides distinct credentials for revocation and per-client admission limits, without partitioning storage. If eight agents all reach Chord through one gateway credential, Chord sees one client for those limits. Per-agent attribution requires each agent to call with its own key or the gateway to forward a distinct credential. Chord does not yet provide a complete per-client audit trail.

## Quality gates

[Public CI](.github/workflows/public-ci.yml) runs the exported test suite, ruff, pyright, and compileall, walks the quickstart above, builds a text-only container and serves a chat turn, and scans locked dependencies for known vulnerabilities. To run the export and container gates locally:

```sh
bash scripts/test_public_export.sh
bash scripts/smoke_text_container.sh
```

They test the checkout they run in. The container smoke uses Docker host networking and runs on Linux CI. The tests establish the source and packaging behavior; they do not certify an operator's live backend or answer quality.

The package version is in `pyproject.toml`. It is separate from a deployment's release label and from the name a gateway gives the service.

Chord is available under the [Apache License 2.0](LICENSE).
