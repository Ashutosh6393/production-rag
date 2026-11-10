from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.ingestion.embeddings.schemas import EmbeddedDocument
from app.vector_store.schemas import IndexVerificationResult, VectorSearchResult


class VectorStore(ABC):
    """Contract for persistent vector storage implementations."""

    @abstractmethod
    def upsert_document(
        self,
        embedded_document: EmbeddedDocument,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        """Insert or update the embedded chunks belonging to a document.

        Args:
            embedded_document: Chunks and their corresponding embedding vectors.
            tenant_id: Identifier of the tenant that owns the document.
            document_id: Stable identifier of the source document.
        """

        raise NotImplementedError

    @abstractmethod
    def delete_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        """Delete all stored chunks belonging to a document."""
        raise NotImplementedError
    
    @abstractmethod
    def verify_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        expected_chunk_ids: Sequence[str],
    ) -> IndexVerificationResult:
        """Verify that all expected chunk IDs exist for the specified document."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        tenant_id: str,
        limit: int = 10,
        document_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar chunks within a tenant's documents."""
        raise NotImplementedError

