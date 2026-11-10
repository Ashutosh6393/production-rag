from app.indexing.exceptions import IndexingError, IndexingVerificationError
from app.indexing.schemas import IndexingResult
from app.ingestion.embeddings.schemas import EmbeddedDocument
from app.vector_store.base import VectorStore
from app.vector_store.schemas import IndexVerificationResult


class IndexingService:
    """Coordinates vector upsert and post-write verification."""

    def __init__(self, vector_store: VectorStore) -> None:
        if vector_store is None:
            raise TypeError("vector_store cannot be None")

        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must implement VectorStore")

        self._vector_store = vector_store

    def index(
        self,
        embedded_document: EmbeddedDocument,
        *,
        tenant_id: str,
        document_id: str,
    ) -> IndexingResult:
        """Upsert a document and verify that every expected chunk is searchable by ID."""

        if not isinstance(embedded_document, EmbeddedDocument):
            raise TypeError("embedded_document must be an EmbeddedDocument instance")

        self._validate_identifier(tenant_id, "tenant_id")
        self._validate_identifier(document_id, "document_id")

        expected_chunk_ids = tuple(
            embedded_chunk.chunk.chunk_id
            for embedded_chunk in embedded_document.chunks
        )

        # Let storage failures retain their original classified exception types.
        self._vector_store.upsert_document(
            embedded_document,
            tenant_id=tenant_id,
            document_id=document_id,
        )

        verification = self._vector_store.verify_document(
            tenant_id=tenant_id,
            document_id=document_id,
            expected_chunk_ids=expected_chunk_ids,
        )

        if not isinstance(verification, IndexVerificationResult):
            raise IndexingError(
                "Vector store returned an invalid document verification result."
            )

        if verification.expected_chunk_ids != expected_chunk_ids:
            raise IndexingError(
                "Vector store verification result does not match the requested chunks."
            )

        if not verification.is_verified:
            raise IndexingVerificationError(verification)

        return IndexingResult(
            tenant_id=tenant_id,
            document_id=document_id,
            embedding_model=embedded_document.model_name,
            vector_dimensions=embedded_document.dimensions,
            indexed_chunk_count=len(expected_chunk_ids),
            verified_chunk_count=len(verification.verified_chunk_ids),
        )

    @staticmethod
    def _validate_identifier(value: str, field_name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")

        if not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

        if value != value.strip():
            raise ValueError(
                f"{field_name} cannot contain leading or trailing whitespace."
            )
