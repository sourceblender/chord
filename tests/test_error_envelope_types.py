"""review 2026-09-24 B23: one upstream failure, one error `type`. The same condition
answered `upstream_error` on the images door, `api_error` on embeddings and
`server_error` everywhere else; a client branching on `error.type` could not tell
them apart as the same thing. They are all `server_error` now, the API value."""
from chord.upstream import UpstreamError

from test_embeddings import _post, embeddings_client  # noqa: F401  (a fixture)
from test_images_api import URL, configured_comfy_client


def test_a_failed_image_generation_is_a_server_error(tmp_path, monkeypatch):
    client, _, _ = configured_comfy_client(tmp_path, monkeypatch, b"unusable image")
    r = client.post(URL, json={"model": "chord-1-poly", "prompt": "a mug"})
    assert r.status_code == 502 and r.json()["error"]["type"] == "server_error"


def test_an_embeddings_upstream_rejection_is_a_server_error(embeddings_client):  # noqa: F811
    client, upstream = embeddings_client
    upstream.failure = UpstreamError(503, "backend down")
    r = _post(client, {"input": "hello", "model": "example/embeddings"})
    assert r.status_code == 503 and r.json()["error"]["type"] == "server_error"


def test_an_unusable_embeddings_answer_is_a_server_error(embeddings_client):  # noqa: F811
    client, upstream = embeddings_client
    upstream.vectors = [[0.0]]
    r = _post(client, {"input": ["first", "second"], "model": "example/embeddings"})
    assert r.status_code == 502 and r.json()["error"]["type"] == "server_error"
