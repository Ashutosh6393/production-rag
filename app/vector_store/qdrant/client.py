from typing import Self

from qdrant_client import QdrantClient

from app.config import Settings
from app.vector_store.exceptions import (
    VectorStoreConnectionError,
    VectorStoreOperationError,
)


class QdrantClientManager:
    """Manages the lifecycle of a Qdrant SDK client."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        if settings is None:
            raise TypeError("settings cannot be None")

        self._closed = False

        if client is not None:
            self._client = client
            return

        try:
            api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None

            self._client = QdrantClient(
                url=str(settings.qdrant_url),
                api_key=api_key,
                timeout=settings.qdrant_timeout_seconds,
            )
        except Exception as exc:
            raise VectorStoreOperationError("Failed to initialize the Qdrant client.") from exc

    def get_client(self) -> QdrantClient:
        """Return the active Qdrant client."""

        if self._closed:
            raise VectorStoreOperationError("Qdrant client manager is already closed.")

        return self._client

    def health_check(self) -> bool:
        """Verify connectivity by requesting the collection list."""

        client = self.get_client()

        try:
            client.get_collections()
            return True
        except Exception as exc:
            raise VectorStoreConnectionError("Unable to communicate with the Qdrant server.") from exc

    def close(self) -> None:
        """Close the Qdrant client. Safe to call multiple times."""

        if self._closed:
            return

        self._closed = True

        try:
            self._client.close()
        except Exception as exc:
            raise VectorStoreOperationError("Failed to close the Qdrant client.") from exc

    def __enter__(self) -> Self:
        if self._closed:
            raise VectorStoreOperationError("Cannot enter a closed Qdrant client manager.")

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
