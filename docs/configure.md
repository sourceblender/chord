# Configure Chord

Start with one running OpenAI-compatible chat model. Save this as `chord.yaml`, replacing the URL and model with your server's values:

```yaml
version: 2
endpoints:
  main:
    type: openai-chat
    url: http://127.0.0.1:11434/v1
    model: llama3.1
    auth: null
routing:
  chat: {endpoint: main}
```

If your backend requires a bearer key, set `auth: "${MODEL_API_KEY}"` and supply `MODEL_API_KEY` in the environment that starts Chord. `auth: null` is for a backend that needs no key.

This is a complete text-only configuration. The main model handles normal chat, helper work, and fast/priority replies. No dispatcher is enabled: chat requests remain chat. You do not need a small model, classifier, manifest override, or certification record to start.

From your Chord checkout, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --frozen --no-dev
CHORD_CONFIG=./chord.yaml uv run --no-sync python -m chord --check-config
CHORD_CONFIG=./chord.yaml uv run --no-sync python -m chord --show-config
CHORD_CONFIG=./chord.yaml uv run --no-sync python -m chord
```

The first two Chord commands validate and print your resolved choices without making inference requests or printing credentials. A valid config does not establish that your backend is available. After starting Chord, confirm a real reply:

```sh
curl http://127.0.0.1:8710/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chord-1-poly","messages":[{"role":"user","content":"hello"}]}'
```

Chord listens on loopback by default. Set `CHORD_API_KEY` before binding beyond loopback. Backend URLs must be reachable from wherever Chord runs; inside a container, `127.0.0.1` means that container. Ports, storage and client keys remain environment settings.

## Choose writers and a dispatcher separately

| Choice | What it does | When omitted |
|---|---|---|
| `routing.chat` | Normal replies; image prompts in chat/Responses; compaction and translation | Required |
| `routing.helper` | Search queries, search/audio/video hand-off notes, failed-delivery checks, and text dispatch when selected | Main model |
| `routing.fast` | Replies and chat image prompts when a client explicitly sends `service_tier: fast` or `priority` | Main model |
| `dispatch` | Chooses chat, image, search, audio or video | No dispatch; every turn is chat |

Helper and fast are independent. Adding a helper does not change fast replies. The tier is an API field sent by the client, not something Chord infers from “answer quickly.” An HTTP classifier such as ModernBERT only chooses a lane; it never answers chat or writes prompts.

To add a small writer and a classifier, extend your file with this shape:

```yaml
version: 2
endpoints:
  main: {type: openai-chat, url: 'http://main-server:8000/v1', model: my-main-model, auth: null}
  small: {type: openai-chat, url: 'http://small-server:8101/v1', model: my-small-model, auth: null}
routing:
  chat: {endpoint: main}
  helper: {endpoint: small}
  # fast omitted: main still answers fast/priority requests.
dispatch:
  by: classifier
  classifier_url: http://classifier-server:8088/v1/route
  classifier_timeout_s: 1
```

Replace the example hostnames and model IDs with your own. Add `fast: {endpoint: small}` under `routing` only if you also want the small model answering fast/priority requests. Set endpoint `thinking: qwen_chat_template` or `json_mode: true` only when your backend supports those features; the defaults are passthrough thinking and no forced JSON.

For text-model dispatch without a classifier, use `dispatch: {by: helper}`. With no helper configured, the main model then picks lanes. To disable dispatch explicitly, use `dispatch: {by: none}`.

If a classifier fails, Chord asks the helper to choose the lane. If that text dispatch fails or returns an unusable decision, the turn becomes chat. Writer defaults apply when settings are omitted; they do not switch an explicitly chosen offline writer to the main model. `--show-config` shows defaults and classifier-failure fallback separately.

## Turn on services as you add them

For search, add `enabled_routes: [search]` at the top level and choose `dispatch.by: helper` or `classifier`. Search uses Brave with `BRAVE_API_KEY` set, otherwise DuckDuckGo. In version 2, the helper writes the query; you do not choose its model in a registry file. No certification record is required.

For images, export your ComfyUI workflow in API format to `workflows/generate.json`. Add a ComfyUI endpoint and an image route alongside your existing chat endpoint and route:

```yaml
image_workflows_dir: workflows
endpoints:
  # Keep main here too.
  comfy: {type: comfyui, url: 'http://comfy-server:8188'}
routing:
  # Keep chat here too.
  image:
    endpoint: comfy
    workflow: generate.json
    prompt_node_id: "6"
    seed_node_id: "3"
    output_node_id: "9"
```

This is an excerpt: merge it into the existing `endpoints` and `routing` mappings, rather than adding duplicate YAML keys. Replace the node IDs with those in your graph. The output node must save one PNG; Chord ships no image checkpoint or workflow. Image configuration enables the image backend; dispatch is also needed for chat to select the image lane. See [ComfyUI workflows](../README.md#comfyui-workflows) for edits, variations and other bindings.

Through chat/Responses, the model answering that request writes the image prompt using the supplied conversation. Through `/v1/images/generations`, your prompt goes to the workflow unchanged, without a chat writer. The direct Images API does not need a dispatcher.

## Existing installs

Version 1 files, files without a version, and environment-only installs keep their earlier behavior. In that form, `routing.router` is a text-model slot, fast follows that slot, search's writer comes from the registry, and dispatcher settings come from `ROUTER_*` environment variables.

Use `version: 2` for the names in this guide. It rejects `routing.router`; version 1 rejects `helper`, `fast` and `dispatch`. Version 2 reads dispatch from the YAML file and ignores old dispatcher environment settings. When migrating, choose dispatch explicitly and inspect `--show-config` before restarting: copying a version 1 router binding into `helper` does not automatically select that helper for fast replies or turn dispatch on.
