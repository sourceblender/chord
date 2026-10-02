"""Unexpected failures keep the service wire and its diagnostic handle."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from chord.config import Settings
from chord.server import Deps, create_app
from test_skeleton import FakeUpstream


class EmptyChoices(FakeUpstream):
    async def complete(self, body):
        self.bodies.append(body)
        return {"choices": [], "usage": {}}, {"model-api-base": "http://direct.example/v1"}


def _last_trace(settings: Settings) -> dict:
    lines = [
        line
        for path in sorted(Path(settings.trace_dir).glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    return json.loads(lines[-1])


def test_unhandled_exception_is_a_traced_service_envelope(tmp_path):
    settings = Settings(data_dir=tmp_path)
    client = TestClient(
        create_app(Deps(settings, upstream=EmptyChoices(), model=lambda _: None)),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "chord-1-poly", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"] == response.headers["x-chord-trace-id"]
    assert response.json() == {
        "error": {
            "message": "The service could not complete the request.",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }
    assert "IndexError" not in response.text

    trace = _last_trace(settings)
    assert trace["trace_id"] == response.headers["x-request-id"]
    assert trace["endpoint"] == "POST /v1/chat/completions"
    assert trace["result_status"] == "failed"
    assert trace["unhandled_exception"].startswith("IndexError(")
