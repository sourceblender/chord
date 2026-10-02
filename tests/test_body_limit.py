"""The global non-multipart body cap: refused before parsing, before auth.

`await request.json()` had no bound at all (review 2026-09-22): a
multi-gigabyte chat body was parsed and held whole in one process. The cap
answers 413 in the API error envelope, exempts multipart (those routes own their
per-part caps), and sits outside the API-key check so a flood is refused
without being parsed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chord.config import Settings
from chord.server import Deps, create_app
from test_skeleton import FakeUpstream

AUTH = {"Authorization": "Bearer k"}


def _client(tmp_path, cap: int) -> TestClient:
    settings = Settings(service_api_key="k", data_dir=tmp_path,
                        max_json_body_bytes=cap)
    return TestClient(create_app(Deps(settings, upstream=FakeUpstream(), model=lambda n: None)))


def _body(n: int) -> dict:
    return {"model": "chord-1-poly", "messages": [{"role": "user", "content": "x" * n}]}


def _assert_cap_envelope(r) -> None:
    assert r.status_code == 413, r.text
    error = r.json()["error"]
    assert error["code"] == "request_too_large"
    assert error["type"] == "invalid_request_error"
    assert error["param"] is None


def test_a_declared_oversized_body_is_refused_without_auth(tmp_path):
    client = _client(tmp_path, 64)
    _assert_cap_envelope(client.post("/v1/chat/completions", json=_body(500)))


def test_a_chunked_oversized_body_is_counted_and_refused(tmp_path):
    # transfer-encoding: chunked carries no Content-Length to pre-check; the
    # counter inside receive is what catches it, mid-parse, as the same 413.
    client = _client(tmp_path, 64)
    import json as _json
    raw = _json.dumps(_body(500)).encode()
    _assert_cap_envelope(client.post("/v1/chat/completions", content=iter([raw[:100], raw[100:]]),
                                     headers={**AUTH, "content-type": "application/json"}))


def test_multipart_is_exempt_because_those_routes_own_their_caps(tmp_path):
    client = _client(tmp_path, 64)
    r = client.post("/v1/files", data={"purpose": "user_data"},
                    files={"file": ("x.txt", b"y" * 500, "text/plain")}, headers=AUTH)
    assert r.status_code == 200, r.text      # 500 bytes through a 64-byte cap


def test_zero_disables_the_cap(tmp_path):
    client = _client(tmp_path, 0)
    r = client.post("/v1/chat/completions", json=_body(5000), headers=AUTH)
    assert r.status_code != 413              # the middleware stands aside


def test_a_small_body_still_reaches_its_route(tmp_path):
    client = _client(tmp_path, 64 * 1024 * 1024)
    r = client.post("/v1/chat/completions", json=_body(4), headers=AUTH)
    assert r.status_code == 200, r.text


def test_a_json_body_wearing_a_multipart_header_is_still_capped(tmp_path):
    """The repro (review 2026-09-22, #2): the exemption used to trust the
    client's Content-Type, and no JSON route checks that header, so a chat
    body declaring multipart/form-data walked past the cap uncapped."""
    client = _client(tmp_path, 64)
    r = client.post("/v1/chat/completions", content=_json_bytes(500),
                    headers={**AUTH, "content-type": "multipart/form-data; boundary=x"})
    _assert_cap_envelope(r)


def test_duplicate_content_type_headers_cannot_pick_the_exempt_tier(tmp_path):
    # Starlette reads the FIRST Content-Type; the old dict comprehension kept
    # the LAST. A request carrying both made the middleware and the parser
    # disagree about what the body was -- json first, multipart last, so the
    # middleware exempted what the route would parse as JSON.
    client = _client(tmp_path, 64)
    r = client.post("/v1/chat/completions", content=_json_bytes(500),
                    headers=[("content-type", "application/json"),
                             ("content-type", "multipart/form-data; boundary=x"),
                             ("authorization", "Bearer k")])
    _assert_cap_envelope(r)


def test_the_multipart_tier_caps_even_exempt_routes(tmp_path, monkeypatch):
    """Exempt means 'the route's own caps own the per-part semantics', not
    'unbounded': Starlette spools the whole body to disk before any per-part
    check runs (review, #3), so the middleware keeps an aggregate tier."""
    from chord import app_core
    monkeypatch.setattr(app_core, "MULTIPART_MAX_BYTES", 64)
    client = _client(tmp_path, 64 * 1024 * 1024)
    r = client.post("/v1/files", data={"purpose": "user_data"},
                    files={"file": ("x.txt", b"y" * 500, "text/plain")}, headers=AUTH)
    _assert_cap_envelope(r)


def test_every_post_route_outside_the_pinned_list_caps_a_lying_multipart_body(tmp_path):
    """The drift pin: a NEW form-parsing route that is not added to
    MULTIPART_PATHS deliberately fails here, because its lying-body probe
    would reach the route instead of the cap. Enumerated from the live app,
    not hand-listed, so it cannot go stale."""
    from chord import app_core

    client = _client(tmp_path, 64)
    tested = 0
    for route in client.app.routes:
        if "POST" not in getattr(route, "methods", set()):
            continue
        path = route.path
        if path in app_core.MULTIPART_PATHS:
            continue
        url = (path.replace("{response_id}", "resp_x").replace("{conversation_id}", "conv_x")
                   .replace("{item_id}", "item_x").replace("{completion_id}", "c"))
        r = client.post(url, content=_json_bytes(500),
                        headers={**AUTH, "content-type": "multipart/form-data; boundary=x"})
        assert r.status_code == 413, (path, r.status_code, r.text[:120])
        assert r.json()["error"]["code"] == "request_too_large", path
        tested += 1
    assert tested >= 10, f"route enumeration found only {tested} POST routes -- the pin went hollow"


def _json_bytes(n: int) -> bytes:
    import json as _json
    return _json.dumps({"model": "chord-1-poly",
                        "messages": [{"role": "user", "content": "x" * n}]}).encode()


@pytest.mark.asyncio
async def test_read_body_within_cap_stops_at_the_cap():
    """The route-level guard for the four multipart doors that used to call
    form() first: Starlette spools the WHOLE body to disk before any per-part
    check, so one authenticated 50 GB POST to a 4 MB route wrote 50 GB
    (review 2026-09-22, #3). The reader must stop early, not measure
    after."""
    from starlette.requests import Request

    from chord.http_transport import read_body_within_cap

    def make(chunks, declared=None):
        state = {"n": 0}

        async def receive():
            if state["n"] < len(chunks):
                c = chunks[state["n"]]
                state["n"] += 1
                return {"type": "http.request", "body": c, "more_body": state["n"] < len(chunks)}
            return {"type": "http.disconnect"}

        headers = [(b"content-length", str(sum(len(c) for c in chunks)).encode())] if declared is None \
            else [(b"content-length", declared)]
        scope = {"type": "http", "method": "POST", "path": "/x", "headers": headers, "query_string": b""}
        return Request(scope, receive), state

    req, state = make([b"x" * 100] * 10)
    assert await read_body_within_cap(req, 250) is None
    assert state["n"] <= 3, f"read {state['n']} chunks; must stop at the cap, not drain the body"

    req, state = make([b"x" * 100] * 10, declared=b"99999999")
    assert await read_body_within_cap(req, 250) is None
    assert state["n"] == 0, "an honest oversize declaration must read zero bytes"

    req, _ = make([b"small body"])
    assert await read_body_within_cap(req, 250) == b"small body"
