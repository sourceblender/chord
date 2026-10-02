from __future__ import annotations

import pytest

from chord.config import ConfigurationError, Settings


def valid_settings(**changes) -> Settings:
    values = {
        "service_api_key": "service-key",
        "persona_model": "example-chat",
        "router_model": "example-router",
        "persona_base_url": "http://persona.internal/v1",
        "router_base_url": "http://router.internal/v1",
    }
    values.update(changes)
    return Settings(**values)


def test_valid_startup_configuration_passes() -> None:
    valid_settings().validate_startup()


def test_startup_validation_reports_all_actionable_failures_together() -> None:
    settings = valid_settings(
        service_api_key="",
        public_host="0.0.0.0",
        persona_base_url="not-a-url",
        router_base_url="",
        public_port=8710,
        internal_port=8710,
        router_timeout_s=0,
        artifact_url_ttl_s=0,
        video_timeout_s=0,
        image_deadline_s=0,
        tts_voices="[]",
        embeddings_basic_auth="missing-colon",
    )

    with pytest.raises(ConfigurationError) as raised:
        settings.validate_startup()

    message = str(raised.value)
    for expected in (
        "CHORD_API_KEY or CHORD_CLIENT_KEYS_JSON is required",
        "chat slot 'router' requires",
        "PERSONA_BASE_URL must be an absolute http(s) URL",
        "PUBLIC_PORT and INTERNAL_PORT must differ",
        "ROUTER_TIMEOUT_S must be greater than zero",
        "VIDEO_TIMEOUT_S must be greater than zero",
        "IMAGE_DEADLINE_S must be greater than zero",
        "ARTIFACT_URL_TTL_S must be greater than zero",
        "TTS_VOICES must be a JSON object",
        "EMBEDDINGS_BASIC_AUTH requires EMBEDDINGS_BASE_URL",
        "EMBEDDINGS_BASIC_AUTH must use user:password form",
    ):
        assert expected in message


def test_same_model_cannot_name_two_direct_routes() -> None:
    settings = valid_settings(router_model="example-chat")

    with pytest.raises(ConfigurationError, match="cannot identify two different direct routes"):
        settings.validate_startup()


def test_shared_backend_cannot_have_conflicting_credentials() -> None:
    settings = valid_settings(
        persona_base_url="http://shared.internal/v1",
        router_base_url="http://shared.internal/v1",
        persona_api_key="one",
        router_api_key="two",
    )

    with pytest.raises(ConfigurationError, match="two different credentials"):
        settings.validate_startup()


@pytest.mark.asyncio
async def test_executable_validates_before_constructing_real_dependencies(monkeypatch) -> None:
    from chord import __main__

    class InvalidSettings:
        def validate_startup(self) -> None:
            raise ConfigurationError("planted invalid deployment")

    constructed = False

    def deps(_settings):
        nonlocal constructed
        constructed = True
        raise AssertionError("dependencies must not be constructed")

    monkeypatch.setattr("chord.config.Settings", InvalidSettings)
    monkeypatch.setattr(__main__, "Deps", deps)

    with pytest.raises(ConfigurationError, match="planted invalid deployment"):
        await __main__.main()
    assert not constructed


def test_retention_and_body_cap_windows_are_validated() -> None:
    with pytest.raises(ConfigurationError, match="CHORD_RETENTION_DAYS"):
        valid_settings(retention_days=-1).validate_startup()
    with pytest.raises(ConfigurationError, match="MAX_JSON_BODY_BYTES"):
        valid_settings(max_json_body_bytes=-1).validate_startup()
    # 0 is each knob's honest "off": keep forever, and no cap.
    valid_settings(retention_days=0, max_json_body_bytes=0).validate_startup()


def test_image_url_allowlist_entries_are_hosts_not_urls() -> None:
    with pytest.raises(ConfigurationError, match="IMAGE_URL_ALLOWED_HOSTS"):
        valid_settings(image_url_allowed_hosts=frozenset({"https://example.com"})).validate_startup()
    with pytest.raises(ConfigurationError, match="IMAGE_URL_ALLOWED_HOSTS"):
        valid_settings(image_url_allowed_hosts=frozenset({"example.com/path"})).validate_startup()
    valid_settings(image_url_allowed_hosts=frozenset({"example.com", "example.com:8080"})).validate_startup()


def test_retention_must_cover_the_signed_link_ttl() -> None:
    """A swept artifact behind a live signed link is a delivered URL that 404s
    inside its own window (review, batch 2)."""
    with pytest.raises(ConfigurationError, match="ARTIFACT_URL_TTL_S"):
        valid_settings(retention_days=1, artifact_url_ttl_s=7 * 24 * 3600).validate_startup()
    valid_settings(retention_days=0, artifact_url_ttl_s=7 * 24 * 3600).validate_startup()  # never swept
    valid_settings(retention_days=30).validate_startup()                                   # covers the default TTL


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("field_name,env_name", [
    ("router_timeout_s", "ROUTER_TIMEOUT_S"),
    ("router_classifier_timeout_s", "ROUTER_CLASSIFIER_TIMEOUT_S"),
    ("video_timeout_s", "VIDEO_TIMEOUT_S"),
    ("image_deadline_s", "IMAGE_DEADLINE_S"),
    ("artifact_url_ttl_s", "ARTIFACT_URL_TTL_S"),
])
def test_a_non_finite_duration_is_refused(field_name, env_name, bad) -> None:
    """Copilot on #338: `<= 0` is False for NaN, and inf passes it too, so
    IMAGE_DEADLINE_S=nan started with a deadline that is no bound at all. Every
    duration setting shares the check, so every one had the hole."""
    with pytest.raises(ConfigurationError) as raised:
        valid_settings(**{field_name: bad}).validate_startup()
    assert f"{env_name} must be greater than zero" in str(raised.value)
