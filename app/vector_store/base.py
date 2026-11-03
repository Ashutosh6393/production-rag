from abc import ABC, abstractmethod

from app.ingestion.embeddings.schemas import EmbeddedDocument


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
