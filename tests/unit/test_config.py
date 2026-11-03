import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_load_default_values() -> None:
    settings = Settings(_env_file=None)

    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert settings.qdrant_collection_name == "prod_rag_chunks"
    assert settings.qdrant_timeout_seconds == 10.0
    assert settings.qdrant_api_key is None


def test_settings_load_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", "20")

    settings = Settings(_env_file=None)

    assert str(settings.qdrant_url) == "https://example.qdrant.io/"
    assert settings.qdrant_collection_name == "test_collection"
    assert settings.qdrant_timeout_seconds == 20.0
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "test-secret"


def test_settings_reject_invalid_url() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_url="not-a-valid-url",
        )


def test_settings_reject_invalid_timeout() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_timeout_seconds=0,
        )


def test_settings_reject_timeout_above_maximum() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_timeout_seconds=150,
        )


def test_settings_reject_empty_collection_name() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_collection_name="",
        )
