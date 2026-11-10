
from app.ingestion.embeddings.exceptions import EmbeddingProviderError
from app.ingestion.embeddings.schemas import EmbeddingBatchResult
from app.ingestion.embeddings.service import EmbeddingService
from app.vector_store.base import VectorStore
from app.vector_store.schemas import VectorSearchResult


class RetrievalService:
    """Coordinates query embedding and vector similarity search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        if embedding_service is None:
            raise TypeError("embedding_service cannot be None")

        if vector_store is None:
            raise TypeError("vector_store cannot be None")

        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        limit: int = 10,
        document_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve the most relevant chunks for a query within a tenant."""

        self._validate_query(query)
        self._validate_identifier(tenant_id, "tenant_id")

        if document_id is not None:
            self._validate_identifier(document_id, "document_id")

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100.")

        # Embed only the user's query. The embedding service handles batching
        # and provider-specific details.
        embedding_result = self._embedding_service.embed([query.strip()])

        if not isinstance(embedding_result, EmbeddingBatchResult):
            raise EmbeddingProviderError(
                "Embedding service returned an invalid result type."
            )

        if len(embedding_result.vectors) != 1:
            raise EmbeddingProviderError(
                "Expected exactly one embedding for the retrieval query."
            )

        query_vector = embedding_result.vectors[0]

        return self._vector_store.search(
            query_vector,
            tenant_id=tenant_id,
            limit=limit,
            document_id=document_id,
        )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

    @staticmethod
    def _validate_identifier(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
