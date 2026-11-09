
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import Settings
from app.vector_store.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreOperationError,
)
from app.vector_store.qdrant.client import QdrantClientManager


class QdrantCollectionManager:
    """Creates Qdrant collections and validates their configuration."""

    def __init__(
        self,
        client_manager: QdrantClientManager,
        settings: Settings,
    ) -> None:
        if client_manager is None:
            raise TypeError("client_manager cannot be None")

        if settings is None:
            raise TypeError("settings cannot be None")

        self._client_manager = client_manager
        self._collection_name = settings.qdrant_collection_name

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if missing, otherwise validate its configuration."""

        if not isinstance(vector_size, int) or isinstance(vector_size, bool):
            raise TypeError("vector_size must be an integer")

        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")

        client = self._client_manager.get_client()

        try:
            exists = client.collection_exists(
                collection_name=self._collection_name
            )
        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to check whether the Qdrant collection exists."
            ) from exc

        if exists:
            self._validate_existing_collection(client, vector_size)
            return

        try:
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return
        except Exception as exc:
            # Another application instance may have created the collection
            # between collection_exists() and create_collection().
            try:
                exists = client.collection_exists(
                    collection_name=self._collection_name
                )
            except Exception:
                exists = False

            if exists:
                self._validate_existing_collection(client, vector_size)
                return

            raise VectorStoreOperationError(
                "Failed to create the Qdrant collection."
            ) from exc

    def _validate_existing_collection(
        self,
        client: QdrantClient,
        expected_vector_size: int,
    ) -> None:
        """Validate the existing collection's vector configuration."""

        try:
            collection = client.get_collection(
                collection_name=self._collection_name
            )
        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to retrieve the Qdrant collection configuration."
            ) from exc

        vectors_config = collection.config.params.vectors

        # This implementation intentionally supports one unnamed/default vector.
        # Named-vector collections require a different storage contract.
        if not isinstance(vectors_config, VectorParams):
            raise VectorStoreConfigurationError(
                "The Qdrant collection must use a single unnamed vector configuration."
            )

        if vectors_config.size != expected_vector_size:
            raise VectorStoreConfigurationError(
                "Qdrant collection vector size mismatch: "
                f"expected {expected_vector_size}, "
                f"found {vectors_config.size}."
            )

        if vectors_config.distance != Distance.COSINE:
            raise VectorStoreConfigurationError(
                "Qdrant collection distance mismatch: "
                f"expected {Distance.COSINE}, "
                f"found {vectors_config.distance}."
            )
