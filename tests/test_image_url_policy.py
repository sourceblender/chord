"""The chat image_url policy (review 2026-09-22, decision #7): a data:image
URI, or an http(s) URL whose exact host[:port] the deployment allowed.

The persona backend fetches image_url parts itself from inside the VLAN, so
an http URL a caller can name is a fetch primitive aimed at the GPU host's
network. The certified vision evidence is a data: URI, so data:image is what
is allowed by default; the allowlist is exact-match opt-in, and the trace
records which hosts allowed parts named -- the alpha evidence that tunes the
list instead of guesses.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chord.config import Settings
from chord.server import Deps, create_app, create_internal_app
from test_skeleton import FakeUpstream

AUTH = {"Authorization": "Bearer k"}
DATA_PNG = "data:image/png;base64," + "QUJDRA=="


def _client(tmp_path, hosts=frozenset()):
    settings = Settings(service_api_key="k", data_dir=tmp_path,
                        image_url_allowed_hosts=hosts)
    deps = Deps(settings, upstream=FakeUpstream(), model=lambda n: None)
    return TestClient(create_app(deps)), deps


def _post(client, url: str, role: str = "user"):
    return client.post("/v1/chat/completions", headers=AUTH,
                       json={"model": "chord-1-poly", "messages": [
                           {"role": role,
                            "content": [{"type": "image_url", "image_url": {"url": url}}]}]})


def test_a_data_image_uri_still_reaches_the_model_unchanged(tmp_path):
    client, deps = _client(tmp_path)
    r = _post(client, DATA_PNG)
    assert r.status_code == 200, r.text
    sent = deps.upstream.bodies[0]["messages"][-1]["content"][0]
    assert sent["image_url"]["url"] == DATA_PNG


@pytest.mark.parametrize("url", [
    "http://10.9.8.6:5056/secret",                 # a sibling service on a private network
    "http://169.254.169.254/latest/meta-data",      # cloud metadata
    "https://example.com/a.png",                    # an unallowed public host
    "file:///etc/passwd",                           # the backend's own disk, never
    "ftp://host/x",
    "data:text/html;base64,PHA+",                   # data, but not an image
    "http://[::1/a.png",                            # urlparse raises; this was a 500 until #339
    "https://[not-an-ip]/a.png",
])
def test_everything_else_is_refused_by_default(tmp_path, url):
    client, deps = _client(tmp_path)
    r = _post(client, url)
    assert r.status_code == 400, url
    error = r.json()["error"]
    assert error["code"] == "unsupported_value"
    assert error["param"] == "messages[0].content"
    assert deps.upstream.bodies == []               # nothing reached the backend


def test_the_policy_covers_every_role(tmp_path):
    # An assistant-message part is forwarded and fetched just the same.
    client, _ = _client(tmp_path)
    assert _post(client, "http://10.0.0.1/x", role="assistant").status_code == 400


def test_the_allowlist_matches_exact_host_and_port(tmp_path):
    client, deps = _client(tmp_path, hosts=frozenset({"images.example:8080", "plain.example"}))
    assert _post(client, "http://images.example:8080/a.png").status_code == 200
    assert _post(client, "http://images.example:9999/a.png").status_code == 400   # wrong port
    assert _post(client, "https://images.example:8080/a.png").status_code == 200  # scheme-agnostic host match
    assert _post(client, "http://plain.example/a.png").status_code == 200         # bare host, no port
    assert _post(client, "http://plain.example:8080/a.png").status_code == 400    # a bare entry is not every port
    assert _post(client, "http://x@images.example:8080/a.png").status_code == 400  # userinfo never matches
    assert _post(client, "http://images.example:8080x/a.png").status_code == 400   # malformed port: refused
    sent = deps.upstream.bodies[-1]["messages"][-1]["content"][0]
    assert sent["image_url"]["url"] == "http://plain.example/a.png"   # the last ALLOWED url, forwarded intact


def test_allowed_http_hosts_are_traced_for_allowlist_tuning(tmp_path):
    client, deps = _client(tmp_path, hosts=frozenset({"images.example:8080"}))
    r = _post(client, "http://images.example:8080/a.png")
    assert r.status_code == 200
    trace = TestClient(create_internal_app(deps)).get(
        f"/internal/traces/{r.headers['x-request-id']}").json()
    assert trace["image_url_forwarded_hosts"] == ["images.example:8080"]


def test_a_data_only_turn_traces_no_hosts(tmp_path):
    client, deps = _client(tmp_path)
    r = _post(client, DATA_PNG)
    assert r.status_code == 200
    trace = TestClient(create_internal_app(deps)).get(
        f"/internal/traces/{r.headers['x-request-id']}").json()
    assert "image_url_forwarded_hosts" not in trace


def test_authority_tricks_are_refused_even_against_an_allowed_host(tmp_path):
    """urlparse and the backend's fetcher are two different URL parsers, and
    an authority carrying userinfo or a backslash is where they disagree:
    a probe showed evil\\@allowed.example reading as evil.example to
    urllib3.util.parse_url while urlparse says allowed.example. Whatever
    either parser would do with these shapes, the policy refuses them."""
    client, deps = _client(tmp_path, hosts=frozenset({"plain.example", "images.example:8080"}))
    for url in [
        "http://user@plain.example/a.png",              # userinfo against a bare-host entry
        "http://evil.example\\@plain.example/a.png",    # her backslash divergence
        "http://evil.example%40plain.example/a.png",    # percent-encoded @
        "http://plain.example%5C@evil.example/a.png",   # percent-encoded backslash
    ]:
        r = _post(client, url)
        assert r.status_code == 400, url
        assert r.json()["error"]["code"] == "unsupported_value"
    assert deps.upstream.bodies == []                   # nothing reached the backend


@pytest.mark.parametrize("image_url", [
    "http://169.254.169.254/latest",            # the bare-string form some layers expand to {url}
    {"url": ["http://169.254.169.254/latest"]},  # a url that is not a string
    {"url": None},
    {},
])
def test_an_image_url_that_is_not_an_object_with_a_string_url_is_refused(tmp_path, image_url):
    """Review 2026-09-24 A7: the policy only judged a dict with a string url and let
    every other shape through to the backend unchecked, so the bare-string form of a
    metadata URL answered 200 while the object form of the same URL answered 400."""
    client, deps = _client(tmp_path)
    r = client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "chord-1-poly", "messages": [
                        {"role": "user", "content": [{"type": "image_url", "image_url": image_url}]}]})
    assert r.status_code == 400, r.text
    assert deps.upstream.bodies == []
