
import pytest

from app.ingestion.embeddings.exceptions import EmbeddingProviderError
from app.ingestion.embeddings.schemas import EmbeddingBatchResult
from app.retrieval.service import RetrievalService
from app.vector_store.base import VectorStore
from app.vector_store.schemas import VectorSearchResult


class FakeEmbeddingService:
    def __init__(self, result=None, error=None):
        self.result = result or EmbeddingBatchResult(
            model_name="text-embedding-3-small",
            vectors=((0.1, 0.2, 0.3),),
        )
        self.error = error
        self.received_texts = None

    def embed(self, texts):
        self.received_texts = texts

        if self.error is not None:
            raise self.error

        return self.result


class FakeVectorStore(VectorStore):
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.received_query_vector = None
        self.received_tenant_id = None
        self.received_limit = None
        self.received_document_id = None

    def upsert_document(
        self,
        embedded_document,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        raise NotImplementedError

    def delete_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        query_vector,
        *,
        tenant_id: str,
        limit: int = 10,
        document_id: str | None = None,
    ):
        self.received_query_vector = query_vector
        self.received_tenant_id = tenant_id
        self.received_limit = limit
        self.received_document_id = document_id

        if self.error is not None:
            raise self.error

        return self.results


    def verify_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        expected_chunk_ids,
    ):
        raise NotImplementedError


@pytest.fixture
def search_result() -> VectorSearchResult:
    return VectorSearchResult(
        document_id="document-123",
        chunk_id="chunk-123",
        chunk_index=0,
        text="Redis is an in-memory data store.",
        source_name="redis-guide.pdf",
        page_numbers=(1,),
        metadata={"section": "Introduction"},
        score=0.92,
    )


@pytest.fixture
def embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def retrieval_service(
    embedding_service: FakeEmbeddingService,
    vector_store: FakeVectorStore,
) -> RetrievalService:
    return RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def test_retrieves_relevant_chunks(
    retrieval_service: RetrievalService,
    embedding_service: FakeEmbeddingService,
    vector_store: FakeVectorStore,
    search_result: VectorSearchResult,
) -> None:
    vector_store.results = [search_result]

    results = retrieval_service.retrieve(
        "What is Redis?",
        tenant_id="tenant-123",
    )

    assert results == [search_result]
    assert embedding_service.received_texts == ["What is Redis?"]
    assert vector_store.received_query_vector == (0.1, 0.2, 0.3)
    assert vector_store.received_tenant_id == "tenant-123"
    assert vector_store.received_limit == 10
    assert vector_store.received_document_id is None


def test_strips_query_before_embedding(
    retrieval_service: RetrievalService,
    embedding_service: FakeEmbeddingService,
) -> None:
    retrieval_service.retrieve(
        "  What is Redis?  ",
        tenant_id="tenant-123",
    )

    assert embedding_service.received_texts == ["What is Redis?"]


def test_forwards_document_filter(
    retrieval_service: RetrievalService,
    vector_store: FakeVectorStore,
) -> None:
    retrieval_service.retrieve(
        "What is Redis?",
        tenant_id="tenant-123",
        document_id="document-123",
    )

    assert vector_store.received_document_id == "document-123"


def test_forwards_custom_limit(
    retrieval_service: RetrievalService,
    vector_store: FakeVectorStore,
) -> None:
    retrieval_service.retrieve(
        "What is Redis?",
        tenant_id="tenant-123",
        limit=5,
    )

    assert vector_store.received_limit == 5


def test_returns_empty_results_without_error(
    retrieval_service: RetrievalService,
) -> None:
    results = retrieval_service.retrieve(
        "What is Redis?",
        tenant_id="tenant-123",
    )

    assert results == []


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_rejects_empty_query(
    retrieval_service: RetrievalService,
    query: str,
) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        retrieval_service.retrieve(query, tenant_id="tenant-123")


def test_rejects_non_string_query(
    retrieval_service: RetrievalService,
) -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        retrieval_service.retrieve(123, tenant_id="tenant-123")


@pytest.mark.parametrize(
    ("tenant_id", "expected_exception"),
    [
        ("", ValueError),
        (" ", ValueError),
        (None, ValueError),
    ],
)
def test_rejects_invalid_tenant_id(
    retrieval_service: RetrievalService,
    tenant_id,
    expected_exception,
) -> None:
    with pytest.raises(expected_exception):
        retrieval_service.retrieve(
            "What is Redis?",
            tenant_id=tenant_id,
        )


def test_rejects_invalid_document_id(
    retrieval_service: RetrievalService,
) -> None:
    with pytest.raises(ValueError, match="document_id"):
        retrieval_service.retrieve(
            "What is Redis?",
            tenant_id="tenant-123",
            document_id=" ",
        )


@pytest.mark.parametrize("limit", [0, -1, 101, 1.5, True])
def test_rejects_invalid_limit(
    retrieval_service: RetrievalService,
    limit,
) -> None:
    with pytest.raises(ValueError, match="limit"):
        retrieval_service.retrieve(
            "What is Redis?",
            tenant_id="tenant-123",
            limit=limit,
        )


def test_rejects_invalid_embedding_result_type() -> None:
    embedding_service = FakeEmbeddingService(result=object())
    vector_store = FakeVectorStore()
    service = RetrievalService(embedding_service, vector_store)

    with pytest.raises(EmbeddingProviderError, match="invalid result type"):
        service.retrieve("What is Redis?", tenant_id="tenant-123")

    assert vector_store.received_query_vector is None


def test_rejects_multiple_query_embeddings() -> None:
    embedding_service = FakeEmbeddingService(
        result=EmbeddingBatchResult(
            model_name="text-embedding-3-small",
            vectors=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
        )
    )
    vector_store = FakeVectorStore()
    service = RetrievalService(embedding_service, vector_store)

    with pytest.raises(EmbeddingProviderError, match="exactly one embedding"):
        service.retrieve("What is Redis?", tenant_id="tenant-123")

    assert vector_store.received_query_vector is None


def test_propagates_embedding_errors() -> None:
    expected_error = RuntimeError("Embedding provider unavailable")
    embedding_service = FakeEmbeddingService(error=expected_error)
    vector_store = FakeVectorStore()
    service = RetrievalService(embedding_service, vector_store)

    with pytest.raises(RuntimeError, match="Embedding provider unavailable"):
        service.retrieve("What is Redis?", tenant_id="tenant-123")

    assert vector_store.received_query_vector is None


def test_propagates_vector_store_errors() -> None:
    expected_error = RuntimeError("Qdrant unavailable")
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(error=expected_error)
    service = RetrievalService(embedding_service, vector_store)

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        service.retrieve("What is Redis?", tenant_id="tenant-123")


@pytest.mark.parametrize(
    ("embedding_service", "vector_store"),
    [
        (None, FakeVectorStore()),
        (FakeEmbeddingService(), None),
    ],
)
def test_rejects_missing_dependencies(
    embedding_service,
    vector_store,
) -> None:
    with pytest.raises(TypeError):
        RetrievalService(embedding_service, vector_store)
