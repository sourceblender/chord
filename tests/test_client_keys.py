"""Direct clients get distinct credentials while the object pool stays shared."""

import json

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from chord.config import ConfigurationError, Settings
from chord.server import Deps, create_app


KEY_A = "a" * 48
KEY_B = "b" * 48
LEGACY = "legacy-secret-for-tests"


def _client(tmp_path, keys: dict[str, str]) -> TestClient:
    settings = Settings(data_dir=tmp_path, service_api_key=LEGACY,
                        client_keys_json=json.dumps(keys),
                        persona_model="example-persona", router_model="example-router",
                        persona_base_url="http://persona.test/v1",
                        router_base_url="http://router.test/v1")
    app = create_app(Deps(settings, model=lambda name: None))

    @app.get("/_test/caller")
    async def caller(request: Request):
        return {"client_id": request.state.client_id}

    return TestClient(app)


def test_distinct_keys_identify_direct_clients_and_keep_legacy_during_cutover(tmp_path):
    client = _client(tmp_path, {"client-a": KEY_A, "qa-example": KEY_B})
    for key, expected in ((KEY_A, "client-a"), (KEY_B, "qa-example"), (LEGACY, "legacy")):
        response = client.get("/_test/caller", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200
        assert response.json() == {"client_id": expected}
    assert client.get("/_test/caller").status_code == 401
    assert client.get("/_test/caller", headers={"Authorization": "Bearer unknown"}).status_code == 401


def test_new_keys_can_read_the_same_shared_endpoint(tmp_path):
    client = _client(tmp_path, {"client-a": KEY_A, "qa-example": KEY_B})
    for key in (KEY_A, KEY_B):
        response = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200
        assert response.json()["data"]


def test_distinct_client_keys_share_stored_conversations_by_id(tmp_path):
    client = _client(tmp_path, {"agent-a": KEY_A, "agent-b": KEY_B})
    a = {"Authorization": f"Bearer {KEY_A}"}
    b = {"Authorization": f"Bearer {KEY_B}"}

    created = client.post("/v1/conversations", json={"metadata": {"owner-note": "agent-a"}}, headers=a)
    assert created.status_code == 200
    conversation_id = created.json()["id"]

    read_by_b = client.get(f"/v1/conversations/{conversation_id}", headers=b)
    assert read_by_b.status_code == 200
    assert read_by_b.json()["metadata"] == {"owner-note": "agent-a"}

    changed_by_b = client.post(f"/v1/conversations/{conversation_id}",
                               json={"metadata": {"owner-note": "agent-b"}}, headers=b)
    assert changed_by_b.status_code == 200
    read_by_a = client.get(f"/v1/conversations/{conversation_id}", headers=a)
    assert read_by_a.json()["metadata"] == {"owner-note": "agent-b"}


def test_a_second_client_can_enumerate_and_read_the_first_clients_file(tmp_path):
    client = _client(tmp_path, {"agent-a": KEY_A, "agent-b": KEY_B})
    a = {"Authorization": f"Bearer {KEY_A}"}
    b = {"Authorization": f"Bearer {KEY_B}"}

    created = client.post("/v1/files", data={"purpose": "user_data"},
                          files={"file": ("note.txt", b"shared-store-proof")}, headers=a)
    assert created.status_code == 200
    file_id = created.json()["id"]

    listed = client.get("/v1/files", headers=b)
    assert listed.status_code == 200
    assert file_id in {item["id"] for item in listed.json()["data"]}
    content = client.get(f"/v1/files/{file_id}/content", headers=b)
    assert content.status_code == 200
    assert content.content == b"shared-store-proof"


def test_removing_one_client_key_revokes_only_that_client(tmp_path):
    client = _client(tmp_path, {"qa-example": KEY_B})
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_A}"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {KEY_B}"}).status_code == 200


def test_new_key_sees_health_revision_but_unauthenticated_probe_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORD_REVISION", "revision-under-test")
    client = _client(tmp_path, {"client-a": KEY_A})
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health", headers={"Authorization": f"Bearer {KEY_A}"}).json() == {
        "status": "ok", "revision": "revision-under-test"}


def test_repeated_authorization_header_cannot_choose_a_different_client(tmp_path):
    client = _client(tmp_path, {"client-a": KEY_A})
    response = client.get("/_test/caller", headers=[
        ("Authorization", f"Bearer {KEY_A}"), ("Authorization", "Bearer wrong")])
    assert response.status_code == 401


@pytest.mark.parametrize("mapping", [
    '{',
    '[]',
    '{"client-a": "short"}',
    '{"bad name": "' + KEY_A + '"}',
    '{"legacy": "' + KEY_A + '"}',
    '{"client-a": "' + KEY_A + '", "qa-example": "' + KEY_A + '"}',
    '{"client-a": "' + KEY_A + '", "client-a": "' + KEY_B + '"}',
    '{"client-a": "' + LEGACY + '"}',
])
def test_invalid_key_maps_refuse_at_startup_without_echoing_secrets(mapping):
    settings = Settings(service_api_key=LEGACY, client_keys_json=mapping,
                        persona_base_url="http://persona.test/v1",
                        router_base_url="http://router.test/v1")
    with pytest.raises(ConfigurationError) as exc:
        settings.validate_startup()
    assert "CHORD_CLIENT_KEYS_JSON" in str(exc.value)
    assert KEY_A not in str(exc.value)
    assert LEGACY not in str(exc.value)
    assert "client_keys_json=" not in repr(settings)
    assert "service_api_key=" not in repr(settings)
