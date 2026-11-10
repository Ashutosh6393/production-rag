import pytest

from app.indexing.exceptions import (
    IndexingError,
    IndexingVerificationError,
)
from app.indexing.schemas import IndexingResult
from app.indexing.service import IndexingService
from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddedDocument
from app.vector_store.base import VectorStore
from app.vector_store.schemas import IndexVerificationResult


@pytest.fixture
def embedded_document() -> EmbeddedDocument:
    return EmbeddedDocument(
        model_name="text-embedding-3-small",
        chunks=(
            EmbeddedChunk(
                chunk=DocumentChunk(
                    chunk_id="chunk-1",
                    chunk_index=0,
                    text="First chunk",
                    source_name="policy.pdf",
                    page_numbers=(1,),
                ),
                vector=(0.1, 0.2, 0.3),
            ),
            EmbeddedChunk(
                chunk=DocumentChunk(
                    chunk_id="chunk-2",
                    chunk_index=1,
                    text="Second chunk",
                    source_name="policy.pdf",
                    page_numbers=(2,),
                ),
                vector=(0.4, 0.5, 0.6),
            ),
        ),
    )


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.upsert_error = None
        self.verification_result = None
        self.verification_error = None
        self.upsert_calls = []
        self.verification_calls = []

    def upsert_document(
        self,
        embedded_document: EmbeddedDocument,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        self.upsert_calls.append((embedded_document, tenant_id, document_id))
        if self.upsert_error is not None:
            raise self.upsert_error

    def delete_document(self, *, tenant_id: str, document_id: str) -> None:
        raise NotImplementedError

    def search(
        self,
        query_vector,
        *,
        tenant_id: str,
        limit: int = 10,
        document_id: str | None = None,
    ):
        raise NotImplementedError

    def verify_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        expected_chunk_ids,
    ) -> IndexVerificationResult:
        self.verification_calls.append(
            (tenant_id, document_id, tuple(expected_chunk_ids))
        )
        if self.verification_error is not None:
            raise self.verification_error

        if self.verification_result is not None:
            return self.verification_result

        return IndexVerificationResult(
            expected_chunk_ids=tuple(expected_chunk_ids),
            verified_chunk_ids=tuple(expected_chunk_ids),
            missing_chunk_ids=(),
        )


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def indexing_service(vector_store: FakeVectorStore) -> IndexingService:
    return IndexingService(vector_store)


def test_indexes_and_verifies_document(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
) -> None:
    result = indexing_service.index(
        embedded_document,
        tenant_id="tenant-1",
        document_id="doc-1",
    )

    assert isinstance(result, IndexingResult)
    assert result.tenant_id == "tenant-1"
    assert result.document_id == "doc-1"
    assert result.embedding_model == "text-embedding-3-small"
    assert result.vector_dimensions == 3
    assert result.indexed_chunk_count == 2
    assert result.verified_chunk_count == 2

    assert vector_store.upsert_calls == [
        (embedded_document, "tenant-1", "doc-1")
    ]
    assert vector_store.verification_calls == [
        ("tenant-1", "doc-1", ("chunk-1", "chunk-2"))
    ]


def test_does_not_verify_when_upsert_fails(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
) -> None:
    vector_store.upsert_error = RuntimeError("Qdrant write failed")

    with pytest.raises(RuntimeError, match="Qdrant write failed"):
        indexing_service.index(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )

    assert vector_store.verification_calls == []


def test_raises_when_verification_finds_missing_chunks(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
) -> None:
    vector_store.verification_result = IndexVerificationResult(
        expected_chunk_ids=("chunk-1", "chunk-2"),
        verified_chunk_ids=("chunk-1",),
        missing_chunk_ids=("chunk-2",),
    )

    with pytest.raises(IndexingVerificationError) as exc_info:
        indexing_service.index(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )

    assert exc_info.value.result.missing_chunk_ids == ("chunk-2",)


def test_propagates_verification_storage_errors(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
) -> None:
    vector_store.verification_error = RuntimeError("Qdrant read failed")

    with pytest.raises(RuntimeError, match="Qdrant read failed"):
        indexing_service.index(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )


def test_rejects_invalid_verification_result_type(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
) -> None:
    vector_store.verification_result = object()

    with pytest.raises(IndexingError, match="invalid document verification"):
        indexing_service.index(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )


def test_rejects_verification_for_different_chunk_set(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
) -> None:
    vector_store.verification_result = IndexVerificationResult(
        expected_chunk_ids=("other-chunk",),
        verified_chunk_ids=("other-chunk",),
        missing_chunk_ids=(),
    )

    with pytest.raises(IndexingError, match="does not match"):
        indexing_service.index(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )


@pytest.mark.parametrize(
    ("tenant_id", "document_id"),
    [
        ("", "doc-1"),
        ("tenant-1", ""),
        (" tenant-1", "doc-1"),
        ("tenant-1", "doc-1 "),
    ],
)
def test_rejects_invalid_identifiers_before_storage_calls(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
    tenant_id: str,
    document_id: str,
) -> None:
    with pytest.raises(ValueError):
        indexing_service.index(
            embedded_document,
            tenant_id=tenant_id,
            document_id=document_id,
        )

    assert vector_store.upsert_calls == []


def test_rejects_invalid_embedded_document(
    indexing_service: IndexingService,
) -> None:
    with pytest.raises(TypeError, match="EmbeddedDocument"):
        indexing_service.index(
            object(),
            tenant_id="tenant-1",
            document_id="doc-1",
        )


def test_rejects_missing_vector_store() -> None:
    with pytest.raises(TypeError, match="cannot be None"):
        IndexingService(None)


def test_raises_when_verification_finds_unexpected_stale_chunks(
    indexing_service: IndexingService,
    vector_store: FakeVectorStore,
    embedded_document: EmbeddedDocument,
) -> None:
    vector_store.verification_result = IndexVerificationResult(
        expected_chunk_ids=("chunk-1", "chunk-2"),
        verified_chunk_ids=("chunk-1", "chunk-2"),
        missing_chunk_ids=(),
        unexpected_chunk_ids=("stale-chunk",),
    )

    with pytest.raises(IndexingVerificationError) as exc_info:
        indexing_service.index(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )

    assert exc_info.value.result.unexpected_chunk_ids == ("stale-chunk",)
