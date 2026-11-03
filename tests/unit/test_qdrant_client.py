from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.vector_store.exceptions import (
    VectorStoreConnectionError,
    VectorStoreOperationError,
)
from app.vector_store.qdrant.client import QdrantClientManager


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def mock_client() -> Mock:
    return Mock(spec=QdrantClient)


def test_client_initializes_with_settings(settings: Settings) -> None:
    with patch("app.vector_store.qdrant.client.QdrantClient") as qdrant_client_class:
        manager = QdrantClientManager(settings)

        qdrant_client_class.assert_called_once_with(
            url=str(settings.qdrant_url),
            api_key=None,
            timeout=settings.qdrant_timeout_seconds,
        )

        assert manager.get_client() is qdrant_client_class.return_value


def test_client_initializes_with_api_key() -> None:
    settings = Settings(
        _env_file=None,
        qdrant_api_key="test-secret",
    )

    with patch("app.vector_store.qdrant.client.QdrantClient") as qdrant_client_class:
        QdrantClientManager(settings)

        qdrant_client_class.assert_called_once_with(
            url=str(settings.qdrant_url),
            api_key="test-secret",
            timeout=settings.qdrant_timeout_seconds,
        )


def test_client_can_be_injected(
    settings: Settings,
    mock_client: Mock,
) -> None:
    with patch("app.vector_store.qdrant.client.QdrantClient") as qdrant_client_class:
        manager = QdrantClientManager(
            settings,
            client=mock_client,
        )

        assert manager.get_client() is mock_client
        qdrant_client_class.assert_not_called()


def test_health_check_returns_true_when_connection_succeeds(
    settings: Settings,
    mock_client: Mock,
) -> None:
    manager = QdrantClientManager(settings, client=mock_client)

    assert manager.health_check() is True

    mock_client.get_collections.assert_called_once_with()


def test_health_check_wraps_connection_errors(
    settings: Settings,
    mock_client: Mock,
) -> None:
    mock_client.get_collections.side_effect = RuntimeError("Connection refused")

    manager = QdrantClientManager(settings, client=mock_client)

    with pytest.raises(VectorStoreConnectionError):
        manager.health_check()


def test_close_releases_client_resources(
    settings: Settings,
    mock_client: Mock,
) -> None:
    manager = QdrantClientManager(settings, client=mock_client)

    manager.close()

    mock_client.close.assert_called_once_with()

    with pytest.raises(VectorStoreOperationError):
        manager.get_client()


def test_close_is_idempotent(
    settings: Settings,
    mock_client: Mock,
) -> None:
    manager = QdrantClientManager(settings, client=mock_client)

    manager.close()
    manager.close()

    mock_client.close.assert_called_once_with()


def test_context_manager_closes_client(
    settings: Settings,
    mock_client: Mock,
) -> None:
    manager = QdrantClientManager(settings, client=mock_client)

    with manager as active_manager:
        assert active_manager is manager
        assert active_manager.get_client() is mock_client

    mock_client.close.assert_called_once_with()


def test_client_initialization_failure_is_wrapped(
    settings: Settings,
) -> None:
    with (
        patch(
            "app.vector_store.qdrant.client.QdrantClient",
            side_effect=RuntimeError("Invalid connection configuration"),
        ),
        pytest.raises(VectorStoreOperationError),
    ):
        QdrantClientManager(settings)


def test_health_check_fails_after_close(
    settings: Settings,
    mock_client: Mock,
) -> None:
    manager = QdrantClientManager(settings, client=mock_client)
    manager.close()

    with pytest.raises(VectorStoreOperationError):
        manager.health_check()

    mock_client.get_collections.assert_not_called()
