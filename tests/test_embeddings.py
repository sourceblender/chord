from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
import httpx
from openai import OpenAI
import pytest

from qa.conformance.schema import Spec, validate_payload
from chord.config import Settings
from chord.server import Deps, create_app
from chord.upstream import Upstream, UpstreamError


@pytest.mark.parametrize("base_url", ["https://embeddings.invalid", "https://embeddings.invalid/v1"])
def test_embeddings_base_url_accepts_root_or_v1_without_doubling(base_url):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"object": "list", "data": [], "model": "example"})

    upstream = Upstream("https://chat.invalid/v1", "", embeddings_base_url=base_url)
    upstream._embeddings = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))
    asyncio.run(upstream.embed({"model": "example", "input": ["hello"]}))
    assert paths == ["/v1/embeddings"]


class EmbeddingUpstream:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.failure: UpstreamError | None = None
        self.vectors: list[list[float]] | None = None
        self.models: list[str] = []
        self.formats: list[str] = []
        self.usage: dict = {"prompt_tokens": 7, "total_tokens": 7}
        self.requests: list[dict] = []
        self.response_override: object | None = None
        # What the live service actually answers with, not the alias requested.
        self.upstream_model: str = "BAAI/bge-m3"
        # Tripwire: token ids must never be decoded by any tokenizer here
        # (review 2026-09-24 B1).
        self.decoded: list[list[int]] = []

    async def decode_embedding_tokens(self, ids: list[int]) -> str:
        self.decoded.append(ids)
        return "decoded:" + ",".join(map(str, ids))

    async def embed(self, request: dict) -> tuple[dict, dict]:
        # TEI 1.9.4's envelope, passed through untouched by Chord.
        #
        # `model` is the UPSTREAM's answer, not the alias we asked for: live
        # 1.9.4 returns "BAAI/bge-m3" for a request naming "example/embeddings".
        # Echoing the alias back here would make the fake disagree with the
        # service in exactly the place a passthrough must not be trusted blindly.
        self.calls.append(request["input"])
        self.models.append(request.get("model"))
        self.formats.append(request.get("encoding_format", "float"))
        self.requests.append(request)
        if self.failure:
            raise self.failure
        if self.response_override is not None:
            return self.response_override, {}
        inputs = request["input"]
        vectors = self.vectors if self.vectors is not None else [
            [float(index), 0.5] for index, _ in enumerate(inputs)
        ]
        # base64 is served BY TEI now, so the fake has to serve it too — a fake
        # that returns floats whatever was asked cannot prove the passthrough
        # preserves base64, and that assertion was the whole point of the SDK
        # round-trip.
        fmt = request.get("encoding_format", "float")
        def _as_wire(vec: list[float]):
            if fmt != "base64":
                return vec
            import base64 as _b64, struct
            return _b64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": _as_wire(v)}
                     for i, v in enumerate(vectors)],
            "model": self.upstream_model,
            "usage": self.usage,
        }, {}


@pytest.fixture
def embeddings_client(tmp_path):
    upstream = EmbeddingUpstream()
    settings = Settings(
        service_api_key="service-key",
        data_dir=tmp_path,
        embeddings_model="example/embeddings",
        embeddings_base_url="https://embeddings.invalid/v1",
    )
    client = TestClient(create_app(Deps(settings, upstream=upstream, model=lambda _: None)))
    return client, upstream


def _post(client: TestClient, body: dict, key: str = "service-key"):
    return client.post(
        "/v1/embeddings",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
    )


def test_embeddings_requires_service_authentication(embeddings_client) -> None:
    client, upstream = embeddings_client

    assert client.post("/v1/embeddings", json={"input": "hello"}).status_code == 401
    assert _post(client, {"input": "hello"}, key="wrong").status_code == 401
    assert upstream.calls == []


def test_embeddings_preserves_batch_order(embeddings_client) -> None:
    client, upstream = embeddings_client

    response = _post(client, {"input": ["first", "second"], "model": "example/embeddings"})

    assert response.status_code == 200
    assert upstream.calls == [["first", "second"]]
    assert response.json() == {
        "object": "list",
        "model": "example/embeddings",  # normalised: TEI answers BAAI/bge-m3
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.0, 0.5]},
            {"object": "embedding", "index": 1, "embedding": [1.0, 0.5]},
        ],
        # Real counts, straight from TEI 1.9.4. This asserted zeros while Chord
        # built the envelope over the native /embed route, which reports none —
        # pinning a number that was never a measurement.
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }
    assert validate_payload(response.json(), kind="embedding", spec=Spec(), fields="strict")["verdict"] == "pass"


def test_embeddings_openai_sdk_round_trip(embeddings_client) -> None:
    client, upstream = embeddings_client
    sdk = OpenAI(
        api_key="service-key",
        base_url="http://testserver/v1",
        max_retries=0,
        http_client=client,
    )

    result = sdk.embeddings.create(input="hello", model="example/embeddings")

    assert result.model == "example/embeddings"
    assert result.data[0].embedding == [0.0, 0.5]
    assert upstream.calls == [["hello"]]
    # The SDK sends base64 by ITSELF on an ordinary create() — that is the whole
    # reason this test exists — and decodes it before we see floats above. If
    # the passthrough ever stops forwarding the format, or the upstream stops
    # honouring it, this is the assertion that catches it.
    assert upstream.formats == ["base64"]


def test_embeddings_forwards_optional_request_fields(embeddings_client) -> None:
    """`user` reached TEI only after review caught it being dropped.

    The route once rebuilt three fields by hand instead of forwarding the
    validated request, which silently lost every optional field the spec allows
    and the method forgot. This pins the fix rather than the symptom."""
    client, upstream = embeddings_client
    response = client.post("/v1/embeddings",
                           json={"input": "hello", "model": "example/embeddings",
                                 "user": "seat-example"},
                           headers={"Authorization": "Bearer service-key"})

    assert response.status_code == 200
    assert upstream.requests[0]["user"] == "seat-example"


@pytest.mark.parametrize(
    ("body", "code", "param"),
    [
        ({}, "missing_required_parameter", "input"),
        ({"input": "ok"}, "missing_required_parameter", "model"),
        ({"input": "", "model": "example/embeddings"}, "invalid_value", "input"),
        ({"input": [], "model": "example/embeddings"}, "invalid_value", "input"),
        ({"input": ["ok", ""], "model": "example/embeddings"}, "invalid_value", "input"),
        ({"input": ["ok", 7], "model": "example/embeddings"}, "invalid_value", "input"),
        ({"input": "ok", "surprise": True}, "unknown_parameter", "surprise"),
    ],
)
def test_embeddings_rejects_bad_requests_before_upstream(
    embeddings_client, body: dict, code: str, param: str
) -> None:
    client, upstream = embeddings_client

    response = _post(client, body)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["param"] == param
    assert upstream.calls == []


@pytest.mark.parametrize("status", [400, 404, 429, 503])
def test_embeddings_preserves_upstream_http_status(embeddings_client, status: int) -> None:
    client, upstream = embeddings_client
    upstream.failure = UpstreamError(status, "backend detail that must not leak")

    response = _post(client, {"input": "hello", "model": "example/embeddings"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == "upstream_error"
    assert "backend detail" not in response.text


def test_models_advertises_configured_embedding_model(embeddings_client) -> None:
    client, _ = embeddings_client

    response = client.get("/v1/models", headers={"Authorization": "Bearer service-key"})

    assert response.status_code == 200
    assert "example/embeddings" in {row["id"] for row in response.json()["data"]}


def test_models_does_not_advertise_unconfigured_embedding_model(tmp_path) -> None:
    settings = Settings(service_api_key="service-key", data_dir=tmp_path)
    client = TestClient(create_app(Deps(settings, upstream=EmbeddingUpstream(), model=lambda _: None)))

    response = client.get("/v1/models", headers={"Authorization": "Bearer service-key"})

    assert "example/embeddings" not in {row["id"] for row in response.json()["data"]}


def test_embeddings_rejects_mismatched_upstream_batch(embeddings_client) -> None:
    client, upstream = embeddings_client
    upstream.vectors = [[0.0]]

    response = _post(client, {"input": ["first", "second"], "model": "example/embeddings"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


@pytest.mark.parametrize(
    ("body", "param", "status"),
    [
        ({"input": "hello", "model": "not-the-served-model"}, "model", 404),
        ({"input": "hello", "dimensions": 512}, "dimensions", 400),
    ],
)
def test_embeddings_refuses_semantics_the_backend_cannot_honor(
    embeddings_client, body: dict, param: str, status: int
) -> None:
    client, upstream = embeddings_client

    response = _post(client, body)

    assert response.status_code == status
    assert response.json()["error"]["param"] == param
    assert upstream.calls == []


def test_embeddings_basic_auth_is_scoped_to_embedding_client() -> None:
    settings = Settings(
        embeddings_base_url="https://embeddings.invalid/v1",
        embeddings_basic_auth="tei-user:tei-password",
    )
    upstream = Upstream(
        "https://chat.invalid/v1",
        "",
        embeddings_base_url=settings.embeddings_base_url,
        embeddings_auth=settings.embeddings_auth_header(),
    )

    assert "Authorization" not in upstream._client.headers
    assert upstream._embeddings.headers["Authorization"] == "Basic dGVpLXVzZXI6dGVpLXBhc3N3b3Jk"


@pytest.mark.parametrize("value", [[1212, 318, 13], [[1212, 13], [42]], [1] * 2049, [[1]] * 2049],
                         ids=["ids", "id-arrays", "ids-over-2048", "id-arrays-over-2048"])
def test_embeddings_refuses_token_array_input_by_name(embeddings_client, value) -> None:
    """Token ids belong to the tokenizer that produced them. LangChain's
    OpenAIEmbeddings sends cl100k arrays by default; decoding those with the
    embedding model's own tokenizer (bge-m3) embedded unrelated text and
    answered 200. Ids from an unknown tokenizer cannot be mapped, so the form is
    refused by name (review 2026-09-24 B1)."""
    client, upstream = embeddings_client

    response = _post(client, {"input": value, "model": "example/embeddings"})

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "input"
    assert "token arrays are not supported" in error["message"]
    assert "strings" in error["message"]
    assert upstream.decoded == [] and upstream.calls == []


@pytest.mark.parametrize("value", [[], [[1], []], [[1, "bad"]], [True, 1], ["text", [1]]])
def test_embeddings_rejects_malformed_token_arrays(embeddings_client, value) -> None:
    client, upstream = embeddings_client

    response = _post(client, {"input": value, "model": "example/embeddings"})

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "input"
    assert upstream.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"object": "wrong", "data": [], "usage": {"prompt_tokens": 1, "total_tokens": 1}},
        {"object": "list", "data": [
            {"object": "embedding", "index": 1, "embedding": [0.1]},
        ], "usage": {"prompt_tokens": 1, "total_tokens": 1}},
        {"object": "list", "data": [
            {"object": "embedding", "index": 0, "embedding": [float("nan")]},
        ], "usage": {"prompt_tokens": 1, "total_tokens": 1}},
        {"object": "list", "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1]},
        ], "usage": {"prompt_tokens": 2, "total_tokens": 1}},
    ],
)
def test_embeddings_refuses_malformed_upstream_responses(embeddings_client, payload) -> None:
    client, upstream = embeddings_client
    upstream.response_override = payload

    response = _post(client, {"input": "hello", "model": "example/embeddings"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_a_down_backend_is_a_502_unreachable_not_an_unhandled_500(embeddings_client) -> None:
    """httpx.ConnectError used to escape the UpstreamError-only handler into
    the generic 500: 'the service could not complete the request' for a
    backend that is simply down, while /v1/completions answers 502
    upstream_unreachable for the identical condition (review 2026-09-22)."""
    import httpx

    client, upstream = embeddings_client

    async def down(request):
        raise httpx.ConnectError("connection refused")

    upstream.embed = down
    r = _post(client, {"model": "example/embeddings", "input": "hello"})
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "upstream_unreachable"


def test_an_upstream_credential_failure_is_our_502_not_the_callers_401(embeddings_client) -> None:
    """A backend refusing OUR credentials passed through as the caller's 401,
    so their SDK blamed their Chord key for a server-side config problem
    (review 2026-09-22)."""
    from chord.upstream import UpstreamError

    client, upstream = embeddings_client

    async def denied(request):
        raise UpstreamError(401, "bad tei credentials")

    upstream.embed = denied
    r = _post(client, {"model": "example/embeddings", "input": "hello"})
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "upstream_error"


def test_embeddings_is_not_served_when_no_embeddings_backend_is_configured(tmp_path) -> None:
    """With EMBEDDINGS_BASE_URL unset, /v1/models omits the embeddings model and
    retrieving it is a 404, but this route used to accept it and send it to the
    persona backend. The door answers the same 404 the models route does
    (review 2026-09-24 B19)."""
    upstream = EmbeddingUpstream()
    settings = Settings(
        service_api_key="service-key",
        data_dir=tmp_path,
        embeddings_model="example/embeddings",
        embeddings_base_url="",
    )
    client = TestClient(create_app(Deps(settings, upstream=upstream, model=lambda _: None)))
    headers = {"Authorization": "Bearer service-key"}

    listed = client.get("/v1/models", headers=headers).json()["data"]
    assert "example/embeddings" not in [m["id"] for m in listed]
    retrieved = client.get("/v1/models/example/embeddings", headers=headers)
    assert retrieved.status_code == 404

    response = _post(client, {"input": "hello", "model": "example/embeddings"})

    assert response.status_code == 404, response.text
    assert response.json() == retrieved.json()
    assert upstream.calls == []
