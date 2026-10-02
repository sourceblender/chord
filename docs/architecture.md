# Chord architecture boundaries

This document records the boundaries enforced by code and CI. It is not a
future-state diagram.

## Runtime composition

`src/chord/server.py` is the application composition root. OpenAI API families
expose a `register(app, deps)` function from their own module. Chat Completions,
Images, Legacy Completions, Models, public system/artifact routes, Audio,
Embeddings, Files, Moderations, Responses/Conversations, stored Chat, and
Videos all follow that pattern. Production dependency construction lives
separately in `dependencies.py`; the ComfyUI graph client for video and image
edit (`comfy.py`, `video.py`) is wired the same way through `ImagesDeps` and
the videos family.

The loopback-only evaluation application is separately composed in
`internal_api.py`; the public app never mounts it.

New API families must not add handlers directly to `server.py`. Chat validation,
tool-policy enforcement, and response orchestration live together in
`chat_api.py`; shared error handling, authentication, artifact signing,
streaming transport, fingerprinting, and wire shaping live in focused modules.

Dependencies passed to an API-family module should use the smallest practical
`Protocol`; modules must not import `Deps` from `server.py`. This keeps the
composition direction one-way and prevents circular imports.

## Capability authorities

Three files answer different questions and must not be merged:

- `qa/conformance/endpoint-profile.json` is the authority for the public OpenAI
  operation surface and its exact response schemas. All consumers load it
  through `qa/conformance/profile.py`.
- `src/chord/manifest.yaml` is the authority for behavior advertised by the
  virtual model, including supported Chat parameters and input/output modes.
- `src/chord/registry.yaml` is the authority for internal specialist routing and
  certification state.

CI proves that the endpoint profile and actual FastAPI routes match in both
directions and that claimed schemas belong to their operations. Operators
certify their own backend configurations separately.

## Router backends

When `ROUTER_ENABLED=true`, something must choose each turn's lane: `chat`, `image`, `search`, `audio` or `video`. `ROUTER_BACKEND` picks who does it.

- `llm` (the default): the router model selected by `routing.router` in `chord.yaml` reads the recent turns and replies with a JSON route. It may also answer `clarify`.
- `classifier`: an HTTP service you run at `ROUTER_CLASSIFIER_URL` picks the lane. Chord ships no classifier model. Any service that speaks this contract works:

  ```
  POST <ROUTER_CLASSIFIER_URL>
  {"text": "user: ...\nassistant: ...\nuser: ..."}
  ```

  `text` is built from the last eight messages of the conversation: of those, each user and assistant message becomes one `role: text` line, oldest first. System and tool messages in that window are dropped, so `text` can hold fewer than eight lines.

  ```
  200 {"route": "image", "probabilities": {"image": 0.97, ...}, "engine_sha256": "..."}
  ```

  `route` is required and must be one of the five lanes; a classifier has no `clarify` lane. `probabilities` and `engine_sha256` are optional. When `probabilities` is an object whose value for the chosen route is a number, Chord traces only that value, rounded to four decimal places, as `classifier_confidence`. When `engine_sha256` is a string, Chord traces its first 64 characters. Any other shape of either field is ignored.

  A non-2xx reply, a timeout (`ROUTER_CLASSIFIER_TIMEOUT_S`, default 1 second), malformed JSON or a route outside the five lanes never fails the turn: Chord routes that turn with the router model instead and traces why. A valid lane whose capability is not registered on this install is different: the turn goes straight to chat, without asking the router model, and the trace records `classifier_route_not_registered`. The classifier returns a lane only: on a specialist lane the router model still writes the brief (intent and constraints), so `routing.router` must stay configured.

## Configuration lifecycle

`Settings` parses environment values. `Settings.validate_startup()` validates
the complete production configuration and reports all actionable errors in one
exception. `python -m chord` calls it before constructing real clients or
opening sockets.

Tests may construct partial `Settings` while injecting fake dependencies. That
is an explicit test seam, not production behavior.

An operator's version-1 `chord.yaml`, selected by `CHORD_CONFIG` (or
`./chord.yaml`), defines named backend endpoints and routes. The loader checks
the complete file before creating backend clients; `python -m chord
--check-config` validates it offline, and `--show-config` reports resolved
routes without credentials. Authentication, storage, and bind addresses remain
environment settings. The portable example is [`chord.example.yaml`](../chord.example.yaml).

## Static analysis adoption

Ruff initially gates Python correctness rules on runtime source. Pyright gates
the stabilized configuration and composition infrastructure, the extracted API
families, the shared Chat contract and wire-shaping modules, and the
endpoint-profile loader.

Both sets are explicit lists in `pyproject.toml`, so widening or narrowing them
is a deliberate edit, visible in the diff and reviewable. Widening is welcome and
narrowing needs a reason, and errors must not be replaced with a blanket
baseline. The point of an explicit list is that somebody chose every name on it.

## Package and evidence identity

The distribution name, import package, repository, and service identity are all
`chord`.

Active operational identifiers follow the same boundary: runtime configuration
uses `CHORD_API_KEY`, `CHORD_DATA_DIR`, `CHORD_REVISION`, and `CHORD_MANIFEST`.
The image API sends plain prompts to its configured ComfyUI workflow and returns
`x-chord-trace-id` for request tracing.

## Direct-client keys

`CHORD_CLIENT_KEYS_JSON` can add direct-client credentials without changing the
stored-object policy. It is a JSON object mapping a stable client id to a
random bearer token of at least 32 characters, for example
`{"gateway":"<random-token>","test-client":"<different-random-token>"}`. The
tokens come from the secret store and must never enter the repository or a
command argument. Startup refuses malformed, duplicate and legacy-key values
without printing any token. The old `CHORD_API_KEY` remains accepted during
this compatibility phase and still signs artifact links.

All recognized keys currently access the same stored chat completions,
conversations, responses, files and videos. Several list endpoints enumerate
objects across keys. Distinct client ids support revocation and configured
admission limits; Chord does not yet produce a complete per-client audit trail.
The ids do not assert ownership of objects or distinguish people behind a
gateway. Owner isolation needs a separate storage migration and a decision for
rows created before owners were recorded.

This makes an installation single-tenant in the storage sense: one trusted
owner or group may run many agents, sessions, conversations and direct-client
keys against the same store. An object ID is a lookup key, not an authorization
scope. A caller holding any valid client key and an object ID can access that
object regardless of which key created it. Deploy separate installations when
callers need separate stored-object boundaries. A gateway using one credential
for many agents appears as one immediate client to Chord; its downstream agents
do not gain distinct admission identities or per-agent attribution. Future owner isolation must
specify owner identity, historical rows, every stored-object endpoint, artifact
links and cross-owner authorization tests before changing this contract.

`CHORD_ARTIFACT_SIGNING_KEY` signs new browser artifact links independently of
bearer authentication. When unset, the legacy `CHORD_API_KEY` still signs links
for compatibility. During rotation, `CHORD_LEGACY_ARTIFACT_VERIFY_UNTIL` is an
absolute Unix timestamp permitting old signatures for at most one configured
link lifetime; startup refuses a farther deadline. Once that time passes, old
signatures refuse even if their embedded expiry is later. Set
`CHORD_ACCEPT_LEGACY_CLIENT_KEY=false` separately after direct clients move to
their own keys. A public deployment can then remove `CHORD_API_KEY` entirely,
provided client keys and the separate artifact signing key are configured.

Chord's repository holds no evidence archive. Certification records, red-team
runs and their captured evidence describe a particular deployment, so they live
with that deployment's operator, not with the product. Records that cite an
evidence path keep resolving there: the move kept every path unchanged.
Maintained executable checks of the product belong under `tests/`.
