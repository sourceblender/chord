"""Artifact link signing can rotate independently of direct-client authentication."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from chord.artifact_links import signature
from chord.config import ConfigurationError, Settings
from chord.server import Deps, create_app


OLD = "legacy-service-secret-for-tests"
NEW = "new-artifact-signing-secret-for-tests"
CLIENT = "c" * 48
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
       b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xdf\xc0\x00\x00\x04\x01\x01\x80\xc5*\x18]\x00\x00\x00\x00IEND\xaeB`\x82")


def _settings(tmp_path, **overrides):
    options = dict(data_dir=tmp_path, service_api_key=OLD,
                   client_keys_json=json.dumps({"client-a": CLIENT}),
                   artifact_signing_key=NEW, public_artifact_base="https://chord.test",
                   legacy_artifact_verify_until=int(time.time()) + 60,
                   persona_model="example-persona", router_model="example-router",
                   persona_base_url="http://persona.test/v1",
                   router_base_url="http://router.test/v1")
    options.update(overrides)
    return Settings(**options)


def _artifact_client(settings):
    deps = Deps(settings, model=lambda name: None)
    artifact = deps.artifacts.register(PNG, "image/png")
    return TestClient(create_app(deps)), artifact.id


def _url(artifact_id, key, expires):
    return f"/v1/artifacts/{artifact_id}?expires={expires}&sig={signature(key, artifact_id, expires)}"


def test_new_signatures_work_and_old_signatures_survive_the_short_rotation_window(tmp_path):
    settings = _settings(tmp_path)
    client, artifact_id = _artifact_client(settings)
    expires = int(time.time()) + 60
    assert client.get(_url(artifact_id, NEW, expires)).status_code == 200
    assert client.get(_url(artifact_id, OLD, expires)).status_code == 200
    far_future = int(time.time()) + settings.artifact_url_ttl_s + 60
    assert client.get(_url(artifact_id, NEW, far_future)).status_code == 401
    assert client.get(_url(artifact_id, OLD, far_future)).status_code == 401


def test_legacy_auth_and_signatures_can_be_retired_independently(tmp_path):
    settings = _settings(tmp_path, accept_legacy_client_key=False)
    client, artifact_id = _artifact_client(settings)
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {OLD}"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {CLIENT}"}).status_code == 200
    expires = int(time.time()) + 60
    assert client.get(_url(artifact_id, OLD, expires)).status_code == 200

    no_old_signatures = _settings(tmp_path / "next", accept_legacy_client_key=False,
                                  legacy_artifact_verify_until=0)
    next_client, next_id = _artifact_client(no_old_signatures)
    assert next_client.get(_url(next_id, OLD, expires)).status_code == 401
    assert next_client.get(_url(next_id, NEW, expires)).status_code == 200


def test_legacy_signature_window_closes_without_another_deploy(tmp_path, monkeypatch):
    now = int(time.time())
    settings = _settings(tmp_path, legacy_artifact_verify_until=now + 60)
    client, artifact_id = _artifact_client(settings)
    old_url = _url(artifact_id, OLD, now + 120)
    new_url = _url(artifact_id, NEW, now + 120)
    assert client.get(old_url).status_code == 200
    monkeypatch.setattr("chord.app_core.time.time", lambda: now + 61)
    assert client.get(old_url).status_code == 401
    assert client.get(new_url).status_code == 200


def test_new_client_keys_and_signer_can_run_without_the_old_service_key(tmp_path):
    settings = _settings(tmp_path, service_api_key="", accept_legacy_client_key=False,
                         legacy_artifact_verify_until=0, public_host="192.0.2.25")
    settings.validate_startup()
    client, artifact_id = _artifact_client(settings)
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {CLIENT}"}).status_code == 200
    assert client.get(_url(artifact_id, NEW, int(time.time()) + 60)).status_code == 200


@pytest.mark.parametrize("value", ["1", "yes", "True ", "off", ""])
def test_legacy_client_auth_flag_refuses_typos(monkeypatch, value):
    monkeypatch.setenv("CHORD_ACCEPT_LEGACY_CLIENT_KEY", value)
    with pytest.raises(ConfigurationError, match="CHORD_ACCEPT_LEGACY_CLIENT_KEY must be true or false"):
        Settings()


@pytest.mark.parametrize("overrides", [
    {"service_api_key": "", "artifact_signing_key": ""},
    {"accept_legacy_client_key": False, "client_keys_json": ""},
    {"legacy_artifact_verify_until": int(time.time()) + 604860},
    {"artifact_signing_key": "short"},
    {"artifact_signing_key": OLD},
    {"artifact_signing_key": CLIENT},
])
def test_public_startup_refuses_an_unusable_key_configuration(tmp_path, overrides):
    settings = _settings(tmp_path, public_host="192.0.2.25", **overrides)
    with pytest.raises(ConfigurationError) as exc:
        settings.validate_startup()
    assert OLD not in str(exc.value) and NEW not in str(exc.value)
