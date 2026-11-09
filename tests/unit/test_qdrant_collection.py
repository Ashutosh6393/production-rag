
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client.models import Distance, VectorParams

from app.config import Settings
from app.vector_store.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreOperationError,
)
from app.vector_store.qdrant.collection import QdrantCollectionManager


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def client_manager(mock_client: Mock) -> Mock:
    manager = Mock()
    manager.get_client.return_value = mock_client
    return manager


@pytest.fixture
def collection_manager(
    client_manager: Mock,
    settings: Settings,
) -> QdrantCollectionManager:
    return QdrantCollectionManager(
        client_manager=client_manager,
        settings=settings,
    )


def make_collection_info(
    vectors_config: VectorParams | dict[str, VectorParams],
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=vectors_config,
            )
        )
    )


def test_create_collection_when_it_does_not_exist(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
    settings: Settings,
) -> None:
    mock_client.collection_exists.return_value = False

    collection_manager.ensure_collection(vector_size=1536)

    mock_client.create_collection.assert_called_once_with(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
    )


def test_does_not_recreate_existing_compatible_collection(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
    settings: Settings,
) -> None:
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = make_collection_info(
        VectorParams(
            size=1536,
            distance=Distance.COSINE,
        )
    )

    collection_manager.ensure_collection(vector_size=1536)

    mock_client.get_collection.assert_called_once_with(
        collection_name=settings.qdrant_collection_name
    )
    mock_client.create_collection.assert_not_called()


def test_rejects_existing_collection_with_wrong_dimensions(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
) -> None:
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = make_collection_info(
        VectorParams(
            size=768,
            distance=Distance.COSINE,
        )
    )

    with pytest.raises(
        VectorStoreConfigurationError,
        match="vector size mismatch",
    ):
        collection_manager.ensure_collection(vector_size=1536)

    mock_client.create_collection.assert_not_called()


def test_rejects_existing_collection_with_wrong_distance(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
) -> None:
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = make_collection_info(
        VectorParams(
            size=1536,
            distance=Distance.DOT,
        )
    )

    with pytest.raises(
        VectorStoreConfigurationError,
        match="distance mismatch",
    ):
        collection_manager.ensure_collection(vector_size=1536)


def test_rejects_named_vector_configuration(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
) -> None:
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = make_collection_info(
        {
            "dense": VectorParams(
                size=1536,
                distance=Distance.COSINE,
            )
        }
    )

    with pytest.raises(
        VectorStoreConfigurationError,
        match="single unnamed vector",
    ):
        collection_manager.ensure_collection(vector_size=1536)


@pytest.mark.parametrize("vector_size", [0, -1])
def test_rejects_non_positive_vector_size(
    collection_manager: QdrantCollectionManager,
    vector_size: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        collection_manager.ensure_collection(vector_size=vector_size)


@pytest.mark.parametrize("vector_size", [True, 1536.0, "1536"])
def test_rejects_non_integer_vector_size(
    collection_manager: QdrantCollectionManager,
    vector_size: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        collection_manager.ensure_collection(vector_size=vector_size)


def test_wraps_collection_exists_failure(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
) -> None:
    mock_client.collection_exists.side_effect = RuntimeError("Connection failed")

    with pytest.raises(VectorStoreOperationError, match="check whether"):
        collection_manager.ensure_collection(vector_size=1536)


def test_wraps_collection_retrieval_failure(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
) -> None:
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.side_effect = RuntimeError("Request failed")

    with pytest.raises(VectorStoreOperationError, match="retrieve"):
        collection_manager.ensure_collection(vector_size=1536)


def test_wraps_collection_creation_failure(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
) -> None:
    mock_client.collection_exists.side_effect = [False, False]
    mock_client.create_collection.side_effect = RuntimeError("Creation failed")

    with pytest.raises(VectorStoreOperationError, match="create"):
        collection_manager.ensure_collection(vector_size=1536)


def test_handles_collection_created_by_another_instance(
    collection_manager: QdrantCollectionManager,
    mock_client: Mock,
) -> None:
    mock_client.collection_exists.side_effect = [False, True]
    mock_client.create_collection.side_effect = RuntimeError(
        "Collection already exists"
    )
    mock_client.get_collection.return_value = make_collection_info(
        VectorParams(
            size=1536,
            distance=Distance.COSINE,
        )
    )

    collection_manager.ensure_collection(vector_size=1536)

    mock_client.create_collection.assert_called_once()
    mock_client.get_collection.assert_called_once()
