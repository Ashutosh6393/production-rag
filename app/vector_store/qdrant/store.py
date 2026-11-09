
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
)

from app.config import Settings
from app.ingestion.embeddings.schemas import EmbeddedDocument
from app.vector_store.base import VectorStore
from app.vector_store.exceptions import (
    VectorStoreOperationError,
)
from app.vector_store.qdrant.client import QdrantClientManager
from app.vector_store.qdrant.collection import QdrantCollectionManager
from app.vector_store.qdrant.point import build_point, generate_point_id


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of the vector store contract."""

    def __init__(
        self,
        client_manager: QdrantClientManager,
        collection_manager: QdrantCollectionManager,
        settings: Settings,
    ) -> None:
        if client_manager is None:
            raise TypeError("client_manager cannot be None")

        if collection_manager is None:
            raise TypeError("collection_manager cannot be None")

        if settings is None:
            raise TypeError("settings cannot be None")

        self._client_manager = client_manager
        self._collection_manager = collection_manager
        self._collection_name = settings.qdrant_collection_name

    def upsert_document(
        self,
        embedded_document: EmbeddedDocument,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        """Insert or replace all chunks belonging to a document."""

        if not isinstance(embedded_document, EmbeddedDocument):
            raise TypeError(
                "embedded_document must be an EmbeddedDocument instance"
            )

        if not embedded_document.chunks:
            raise ValueError("embedded_document must contain at least one chunk")

        client = self._client_manager.get_client()

        # Ensure the collection matches the embedding dimensions.
        self._collection_manager.ensure_collection(
            vector_size=embedded_document.dimensions
        )

        # Construct the new points before modifying Qdrant.
        points = [
            build_point(
                embedded_chunk,
                tenant_id=tenant_id,
                document_id=document_id,
                embedding_model=embedded_document.model_name,
            )
            for embedded_chunk in embedded_document.chunks
        ]

        new_point_ids = {
            generate_point_id(
                tenant_id=tenant_id,
                document_id=document_id,
                chunk_id=item.chunk.chunk_id,
            )
            for item in embedded_document.chunks
        }

        try:
            # Capture existing points before upserting.
            existing_point_ids = self._get_document_point_ids(
                client,
                tenant_id=tenant_id,
                document_id=document_id,
            )

            # Upsert new points. Existing IDs are overwritten.
            client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to upsert document points"
            ) from exc

        # Remove chunks that no longer belong to the document.
        stale_point_ids = existing_point_ids - new_point_ids

        if not stale_point_ids:
            return

        try:
            client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(
                    points=list(stale_point_ids)
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreOperationError(
                "stale points could not be deleted"
            ) from exc

    def delete_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        """Delete all chunks belonging to a tenant's document."""

        # Reuse point ID validation for consistent identifier checks.
        generate_point_id(
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_id="validation-only",
        )

        client = self._client_manager.get_client()

        document_filter = self._build_document_filter(
            tenant_id=tenant_id,
            document_id=document_id,
        )

        try:
            client.delete(
                collection_name=self._collection_name,
                points_selector=FilterSelector(filter=document_filter),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to delete document points"
            ) from exc

    def _get_document_point_ids(
        self,
        client: QdrantClient,
        *,
        tenant_id: str,
        document_id: str,
    ) -> set[str]:
        """Retrieve all existing point IDs for a document."""

        document_filter = self._build_document_filter(
            tenant_id=tenant_id,
            document_id=document_id,
        )

        point_ids: set[str] = set()
        offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name=self._collection_name,
                scroll_filter=document_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            point_ids.update(str(record.id) for record in records)

            if next_offset is None:
                break

            offset = next_offset

        return point_ids

    @staticmethod
    def _build_document_filter(
        *,
        tenant_id: str,
        document_id: str,
    ) -> Filter:
        """Build a tenant-scoped document filter."""

        return Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                ),
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                ),
            ]
        )
