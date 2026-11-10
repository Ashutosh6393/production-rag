import math
from collections.abc import Sequence

from pydantic import ValidationError
from qdrant_client import QdrantClient, models
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
from app.vector_store.schemas import IndexVerificationResult, VectorSearchResult


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of the vector store contract."""

    @staticmethod
    def _validate_identifier(value: str, field_name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")

        if not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

        if value != value.strip():
            raise ValueError(f"{field_name} cannot contain leading or trailing whitespace.")

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
            raise TypeError("embedded_document must be an EmbeddedDocument instance")

        self._validate_identifier(tenant_id, "tenant_id")
        self._validate_identifier(document_id, "document_id")

        if not embedded_document.chunks:
            raise ValueError("embedded_document must contain at least one chunk")

        client = self._client_manager.get_client()

        # Ensure the collection matches the embedding dimensions.
        self._collection_manager.ensure_collection(vector_size=embedded_document.dimensions)

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
            raise VectorStoreOperationError("Failed to upsert document points") from exc

        # Remove chunks that no longer belong to the document.
        stale_point_ids = existing_point_ids - new_point_ids

        if not stale_point_ids:
            return

        try:
            client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=list(stale_point_ids)),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreOperationError("stale points could not be deleted") from exc

    def delete_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        """Delete all chunks belonging to a tenant's document."""

        self._validate_identifier(tenant_id, "tenant_id")
        self._validate_identifier(document_id, "document_id")

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
            raise VectorStoreOperationError("Failed to delete document points") from exc

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

    def verify_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        expected_chunk_ids: Sequence[str],
    ) -> IndexVerificationResult:
        """Verify the exact chunk set stored for a tenant's document."""

        self._validate_identifier(tenant_id, "tenant_id")
        self._validate_identifier(document_id, "document_id")

        if isinstance(expected_chunk_ids, (str, bytes)) or not isinstance(
            expected_chunk_ids, Sequence
        ):
            raise TypeError("expected_chunk_ids must be a sequence of strings")

        if not expected_chunk_ids:
            raise ValueError("expected_chunk_ids cannot be empty")

        normalized_chunk_ids: list[str] = []
        for chunk_id in expected_chunk_ids:
            self._validate_identifier(chunk_id, "chunk_id")
            normalized_chunk_ids.append(chunk_id)

        if len(normalized_chunk_ids) != len(set(normalized_chunk_ids)):
            raise ValueError("expected_chunk_ids must not contain duplicates")

        expected_ids = tuple(normalized_chunk_ids)
        expected_set = set(expected_ids)
        document_filter = self._build_document_filter(
            tenant_id=tenant_id,
            document_id=document_id,
        )

        client = self._client_manager.get_client()
        actual_chunk_ids: list[str] = []
        offset = None

        try:
            if not client.collection_exists(collection_name=self._collection_name):
                records = []
            else:
                while True:
                    records, next_offset = client.scroll(
                        collection_name=self._collection_name,
                        scroll_filter=document_filter,
                        limit=256,
                        offset=offset,
                        with_payload=["tenant_id", "document_id", "chunk_id"],
                        with_vectors=False,
                    )

                    for record in records:
                        payload = record.payload

                        if not isinstance(payload, dict):
                            raise VectorStoreOperationError(
                                "Qdrant returned a point with an invalid verification payload."
                            )

                        chunk_id = payload.get("chunk_id")
                        if (
                            payload.get("tenant_id") != tenant_id
                            or payload.get("document_id") != document_id
                            or not isinstance(chunk_id, str)
                            or not chunk_id
                        ):
                            raise VectorStoreOperationError(
                                "Qdrant returned a point whose payload identity is invalid."
                            )

                        expected_point_id = generate_point_id(
                            tenant_id=tenant_id,
                            document_id=document_id,
                            chunk_id=chunk_id,
                        )
                        if str(record.id) != expected_point_id:
                            raise VectorStoreOperationError(
                                "Qdrant returned a point whose ID does not match its payload."
                            )

                        actual_chunk_ids.append(chunk_id)

                    if next_offset is None:
                        break

                    offset = next_offset

        except VectorStoreOperationError:
            raise
        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to verify document points in Qdrant."
            ) from exc

        if len(actual_chunk_ids) != len(set(actual_chunk_ids)):
            raise VectorStoreOperationError(
                "Qdrant contains duplicate chunk identities for the requested document."
            )

        actual_set = set(actual_chunk_ids)
        verified_chunk_ids = tuple(
            chunk_id for chunk_id in expected_ids if chunk_id in actual_set
        )
        missing_chunk_ids = tuple(
            chunk_id for chunk_id in expected_ids if chunk_id not in actual_set
        )
        unexpected_chunk_ids = tuple(
            chunk_id for chunk_id in actual_chunk_ids if chunk_id not in expected_set
        )

        return IndexVerificationResult(
            expected_chunk_ids=expected_ids,
            verified_chunk_ids=verified_chunk_ids,
            missing_chunk_ids=missing_chunk_ids,
            unexpected_chunk_ids=unexpected_chunk_ids,
        )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        tenant_id: str,
        limit: int = 10,
        document_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar chunks within a tenant's documents."""

        self._validate_identifier(tenant_id, "tenant_id")

        if document_id is not None:
            self._validate_identifier(document_id, "document_id")

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100.")

        if isinstance(query_vector, (str, bytes)) or not isinstance(
            query_vector, Sequence
        ):
            raise ValueError("query_vector must be a non-empty sequence of numbers.")

        if not query_vector:
            raise ValueError("query_vector cannot be empty.")

        validated_vector: list[float] = []

        for value in query_vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("query_vector must contain only numeric values.")

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise ValueError("query_vector must contain only finite values.")

            validated_vector.append(numeric_value)

        client = self._client_manager.get_client()

        try:
            if not client.collection_exists(collection_name=self._collection_name):
                return []

            conditions: list[models.Condition] = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ]

            if document_id is not None:
                conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                )

            query_filter = models.Filter(must=conditions)

            response = client.query_points(
                collection_name=self._collection_name,
                query=validated_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to search for document chunks in Qdrant."
            ) from exc

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = point.payload

            if not isinstance(payload, dict):
                raise VectorStoreOperationError(
                    "Qdrant returned a search result with an invalid payload."
                )

            try:
                result = VectorSearchResult.model_validate(
                    {
                        "document_id": payload["document_id"],
                        "chunk_id": payload["chunk_id"],
                        "chunk_index": payload["chunk_index"],
                        "text": payload["text"],
                        "source_name": payload["source_name"],
                        "page_numbers": payload.get("page_numbers", []),
                        "metadata": payload.get("metadata", {}),
                        "score": point.score,
                    }
                )
            except (KeyError, ValidationError, TypeError, ValueError) as exc:
                raise VectorStoreOperationError(
                    "Qdrant returned a search result with an invalid payload structure."
                ) from exc

            results.append(result)

        return results