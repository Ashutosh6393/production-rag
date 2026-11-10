import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def test_settings_load_default_values() -> None:
    settings = Settings(_env_file=None)

    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert settings.qdrant_collection_name == "prod_rag_chunks"
    assert settings.qdrant_timeout_seconds == 10
    assert settings.qdrant_api_key is None
    assert settings.database_url is None
    assert settings.database_pool_size == 5
    assert settings.database_max_overflow == 10
    assert settings.database_pool_timeout_seconds == 30
    assert settings.database_pool_recycle_seconds == 1800
    assert settings.database_pool_pre_ping is True


def test_settings_load_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/prod_rag")
    monkeypatch.setenv("DATABASE_POOL_SIZE", "8")
    monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "12")
    monkeypatch.setenv("DATABASE_POOL_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("DATABASE_POOL_RECYCLE_SECONDS", "3600")
    monkeypatch.setenv("DATABASE_POOL_PRE_PING", "false")

    settings = Settings(_env_file=None)

    assert str(settings.qdrant_url) == "https://example.qdrant.io/"
    assert settings.qdrant_collection_name == "test_collection"
    assert settings.qdrant_timeout_seconds == 20
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "test-secret"
    assert isinstance(settings.database_url, SecretStr)
    assert settings.database_url.get_secret_value().endswith("/prod_rag")
    assert settings.database_pool_size == 8
    assert settings.database_max_overflow == 12
    assert settings.database_pool_timeout_seconds == 45
    assert settings.database_pool_recycle_seconds == 3600
    assert settings.database_pool_pre_ping is False


def test_settings_reject_invalid_url() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_url="not-a-valid-url")


def test_settings_reject_invalid_timeout() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_timeout_seconds=0)


def test_settings_reject_timeout_above_maximum() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_timeout_seconds=150)


def test_settings_reject_empty_collection_name() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_collection_name="")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("database_pool_size", 0),
        ("database_pool_size", 101),
        ("database_max_overflow", -1),
        ("database_max_overflow", 101),
        ("database_pool_timeout_seconds", 0),
        ("database_pool_timeout_seconds", 301),
        ("database_pool_recycle_seconds", 0),
        ("database_pool_recycle_seconds", 86401),
    ],
)
def test_settings_reject_invalid_database_pool_configuration(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
