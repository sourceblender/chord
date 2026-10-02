"""A credential belongs to an ADDRESS, not to the service.

One `api_key` used to be handed to `Upstream` and applied to every client it built, so
once chat moved direct the LiteLLM gateway key was presented -- in cleartext, over plain
HTTP -- to persona, router, STT and TTS, none of which is LiteLLM and none of which
checks it. Nothing leaked it back; the cost is that the gateway credential lands in four
services' request logs, so any one of them becoming readable becomes a gateway problem.

These cells assert the boundary by reading the header each client would actually send,
not by reading the settings that were meant to produce it.
"""
import httpx
import pytest

from chord.config import Settings
from chord.server import Deps, create_app  # noqa: F401  (create_app: import-time wiring)
from chord.upstream import Upstream
from test_skeleton import FakeUpstream  # noqa: F401

PERSONA = "http://persona-direct:8113/v1"
ROUTER = "http://router-direct:8101/v1"
STT = "http://stt-direct:5057/v1"
TTS = "http://tts-direct:5056/v1"


def direct(tmp_path, **over):
    base = dict(data_dir=tmp_path,
                persona_model="persona-model", persona_base_url=PERSONA,
                router_model="router-model", router_base_url=ROUTER,
                stt_base_url=STT, tts_base_url=TTS)
    # merge rather than **over on top of literals: a caller overriding one of the
    # defaults got "multiple values for keyword argument" instead of an override.
    return Settings(**{**base, **over})


def auth_of(client: httpx.AsyncClient) -> str | None:
    return client.headers.get("Authorization")


def clients(upstream: Upstream) -> dict[str, httpx.AsyncClient]:
    """Every client the object actually holds, keyed by its address.

    Enumerated from the instance rather than from a list this file maintains: a client
    added later is covered without anyone remembering to add it here.
    """
    found = {}

    def add(c):
        # httpx normalises base_url with a trailing slash. Keying on the raw value made
        # every lookup here miss and read as "no client was built" -- a wrong instrument
        # reporting a missing feature.
        found[str(c.base_url).rstrip("/")] = c

    for value in vars(upstream).values():
        if isinstance(value, httpx.AsyncClient):
            add(value)
        elif isinstance(value, dict):
            for inner in value.values():
                if isinstance(inner, httpx.AsyncClient):
                    add(inner)
    return found


def test_a_direct_backend_gets_no_authorization_header_at_all(tmp_path):
    """Not an empty bearer -- no header. An empty `Bearer ` is still a credential claim."""
    s = direct(tmp_path)
    deps = Deps(s)
    for url, client in clients(deps.upstream).items():
        assert auth_of(client) is None, f"{url} was handed {auth_of(client)!r}"


def test_direct_backends_do_not_inherit_another_key(tmp_path):
    deps = Deps(direct(tmp_path))
    for url, client in clients(deps.upstream).items():
        assert auth_of(client) is None, f"unexpected key reached {url}"


def test_chat_without_a_direct_url_refuses_to_start_rather_than_using_the_gateway(tmp_path):
    """The chat slots are fail-closed, so there is no unconfigured-deployment path left.

    This cell asserted the opposite first, because `config.py` still documents a gateway
    fallback for unset routes. That stopped being true when service_tier's fail-closed
    slots landed: `Deps` calls `slot_target("persona")` unconditionally. The comment
    outlived its behaviour by one merge and I inherited the claim from it.
    """
    s = Settings(data_dir=tmp_path)
    with pytest.raises(ValueError, match="requires a nonblank model and direct base URL"):
        Deps(s)


@pytest.mark.parametrize("field,url", [("persona_api_key", PERSONA), ("router_api_key", ROUTER),
                                       ("stt_api_key", STT), ("tts_api_key", TTS)])
def test_an_explicitly_configured_backend_key_is_presented(tmp_path, field, url):
    deps = Deps(direct(tmp_path, **{field: "OWN-KEY"}))
    client = clients(deps.upstream).get(url.rstrip("/"))
    assert client is not None, f"no client was built for {url}"
    assert auth_of(client) == "Bearer OWN-KEY"


def test_credential_for_resolves_by_address_not_by_purpose(tmp_path):
    s = direct(tmp_path, persona_api_key="P")
    assert s.credential_for(PERSONA) == "P"
    assert s.credential_for(ROUTER) == ""
    assert s.credential_for("http://never-configured:9/v1") == ""
    assert s.credential_for("") == ""


def test_every_upstream_client_is_covered_by_the_key_map(tmp_path):
    """Scope is in the NAME. This enumerates `Upstream` only.

    Its first version was called "every client production builds" while examining one of
    the two client factories, so it read as whole-system coverage and was not. The model
    factory is covered by the cells below; the name now says which half this one is.
    """
    s = direct(tmp_path)
    addresses = {u.rstrip("/") for u in clients(Deps(s).upstream)}
    decided = {u.rstrip("/") for u in (PERSONA, ROUTER, STT, TTS)}
    assert addresses <= decided, f"client at an address nobody decided about: {addresses - decided}"


def wire_auth(chat_model) -> tuple[bool, str | None]:
    """What this client actually puts on the wire, read from a server that received it.

    Constructed-client introspection is not enough here: `auth_headers` is computed from
    `api_key` and still shows the key even when `default_headers` overrides it at request
    time. Only the receiving end settles it.
    """
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen: list[tuple[bool, str | None]] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(("Authorization" in self.headers, self.headers.get("Authorization")))
            b = _json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                          "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        chat_model(srv.server_address[1]).invoke("hi")
    finally:
        srv.shutdown()
    assert seen, "no request reached the probe server; this cell would pass vacuously"
    return seen[-1]


def test_the_model_factory_never_sends_the_gateway_key_to_a_direct_backend(tmp_path):
    """The second client factory. Wire-captured, not inferred from the object."""
    def build(port):
        s = direct(tmp_path, persona_base_url=f"http://127.0.0.1:{port}/v1")
        return Deps(s).model(s.persona_model)
    present, value = wire_auth(build)
    assert "GATEWAY-KEY" not in (value or ""), f"gateway key reached a direct backend: {value!r}"
    assert not (value or "").strip(), f"expected no credential, got {value!r}"


def test_the_model_factory_presents_a_configured_backend_key(tmp_path):
    def build(port):
        s = direct(tmp_path, persona_base_url=f"http://127.0.0.1:{port}/v1", persona_api_key="OWN-KEY")
        return Deps(s).model(s.persona_model)
    _, value = wire_auth(build)
    assert value == "Bearer OWN-KEY"


def test_the_model_factory_refuses_an_unconfigured_model(tmp_path):
    with pytest.raises(ValueError, match="no configured backend"):
        Deps(direct(tmp_path)).model("unconfigured-model")


def test_one_address_cannot_carry_two_different_credentials(tmp_path):
    shared = "http://shared-backend:9000/v1"
    with pytest.raises(ValueError, match="two different credentials"):
        Deps(direct(tmp_path, persona_base_url=shared, router_base_url=shared,
                    persona_api_key="A", router_api_key="B"))


def test_the_model_factory_reuses_one_client_and_never_retries_silently(tmp_path):
    """One client per name: a fresh ChatOpenAI per call redid TCP/TLS setup on
    every router decision, identity resolution and specialist attempt and left
    the pool to GC. And openai's default of 2 silent 5xx retries doubled GPU
    load on turns already failing -- the deployment's no-retry policy
    can duplicate work (review 2026-09-22)."""
    from chord.dependencies import MODEL_TIMEOUT_S

    deps = Deps(direct(tmp_path))
    first = deps.model(direct(tmp_path).persona_model)
    assert deps.model("persona-model") is first
    assert first.max_retries == 0
    assert first.request_timeout == MODEL_TIMEOUT_S
