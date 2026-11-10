from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import MagicMock

import pytest

from app.vector_store.exceptions import VectorStoreOperationError
from app.vector_store.schemas import VectorSearchResult

import pytest
from qdrant_client.models import Filter, FilterSelector, PointIdsList

from app.config import Settings
from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddedDocument
from app.vector_store.exceptions import VectorStoreOperationError
from app.vector_store.qdrant.point import generate_point_id
from app.vector_store.qdrant.store import QdrantVectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def client() -> Mock:
    return Mock()


@pytest.fixture
def mock_client(client: Mock) -> Mock:
    return client


@pytest.fixture
def client_manager(client: Mock) -> Mock:
    manager = Mock()
    manager.get_client.return_value = client
    return manager


@pytest.fixture
def collection_manager() -> Mock:
    return Mock()


@pytest.fixture
def store(
    client_manager: Mock,
    collection_manager: Mock,
    settings: Settings,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client_manager=client_manager,
        collection_manager=collection_manager,
        settings=settings,
    )


@pytest.fixture
def embedded_document() -> EmbeddedDocument:
    chunks = (
        EmbeddedChunk(
            chunk=DocumentChunk(
                chunk_id="chunk-1",
                chunk_index=0,
                text="First chunk",
                source_name="policy.pdf",
                page_numbers=(1,),
                metadata={},
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
                metadata={},
            ),
            vector=(0.4, 0.5, 0.6),
        ),
    )

    return EmbeddedDocument(
        model_name="text-embedding-3-small",
        chunks=chunks,
    )


def test_upserts_document(
    store: QdrantVectorStore,
    client: Mock,
    collection_manager: Mock,
    embedded_document: EmbeddedDocument,
    settings: Settings,
) -> None:
    client.scroll.return_value = ([], None)

    store.upsert_document(
        embedded_document,
        tenant_id="tenant-1",
        document_id="doc-1",
    )

    collection_manager.ensure_collection.assert_called_once_with(vector_size=embedded_document.dimensions)

    client.upsert.assert_called_once()

    call_kwargs = client.upsert.call_args.kwargs

    assert call_kwargs["collection_name"] == settings.qdrant_collection_name
    assert len(call_kwargs["points"]) == 2
    assert call_kwargs["wait"] is True

    client.delete.assert_not_called()


def test_deletes_stale_chunks_after_upsert(
    store: QdrantVectorStore,
    client: Mock,
    embedded_document: EmbeddedDocument,
) -> None:
    stale_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    client.scroll.return_value = (
        [
            SimpleNamespace(id=stale_id),
        ],
        None,
    )

    store.upsert_document(
        embedded_document,
        tenant_id="tenant-1",
        document_id="doc-1",
    )

    client.upsert.assert_called_once()
    client.delete.assert_called_once()

    selector = client.delete.call_args.kwargs["points_selector"]

    assert isinstance(selector, PointIdsList)
    assert selector.points == [stale_id]


def test_does_not_delete_stale_chunks_when_upsert_fails(
    store: QdrantVectorStore,
    client: Mock,
    embedded_document: EmbeddedDocument,
) -> None:
    client.scroll.return_value = (
        [SimpleNamespace(id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")],
        None,
    )

    client.upsert.side_effect = RuntimeError("Upsert failed")

    with pytest.raises(
        VectorStoreOperationError,
        match="Failed to upsert document points",
    ):
        store.upsert_document(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )

    client.delete.assert_not_called()


def test_deletes_document_using_tenant_scoped_filter(
    store: QdrantVectorStore,
    client: Mock,
    settings: Settings,
) -> None:
    store.delete_document(
        tenant_id="tenant-1",
        document_id="doc-1",
    )

    client.delete.assert_called_once()

    kwargs = client.delete.call_args.kwargs

    assert kwargs["collection_name"] == settings.qdrant_collection_name
    assert kwargs["wait"] is True

    selector = kwargs["points_selector"]

    assert isinstance(selector, FilterSelector)
    assert isinstance(selector.filter, Filter)

    conditions = selector.filter.must

    assert len(conditions) == 2
    assert conditions[0].key == "tenant_id"
    assert conditions[0].match.value == "tenant-1"
    assert conditions[1].key == "document_id"
    assert conditions[1].match.value == "doc-1"


def test_wraps_delete_failure(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    client.delete.side_effect = RuntimeError("Delete failed")

    with pytest.raises(
        VectorStoreOperationError,
        match="Failed to delete document points",
    ):
        store.delete_document(
            tenant_id="tenant-1",
            document_id="doc-1",
        )


def test_retrieves_multiple_scroll_pages(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    client.scroll.side_effect = [
        (
            [
                SimpleNamespace(id="point-1"),
                SimpleNamespace(id="point-2"),
            ],
            "next-page",
        ),
        (
            [
                SimpleNamespace(id="point-3"),
            ],
            None,
        ),
    ]

    point_ids = store._get_document_point_ids(
        client,
        tenant_id="tenant-1",
        document_id="doc-1",
    )

    assert point_ids == {"point-1", "point-2", "point-3"}
    assert client.scroll.call_count == 2

    assert client.scroll.call_args_list[1].kwargs["offset"] == "next-page"


def test_rejects_empty_document(
    store: QdrantVectorStore,
) -> None:
    empty_document = EmbeddedDocument.model_construct(
        model_name="text-embedding-3-small",
        chunks=(),
    )

    with pytest.raises(ValueError, match="at least one chunk"):
        store.upsert_document(
            empty_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )


@pytest.mark.parametrize(
    ("tenant_id", "document_id"),
    [
        ("", "doc-1"),
        ("tenant-1", ""),
        ("   ", "doc-1"),
        ("tenant-1", "   "),
    ],
)
def test_rejects_invalid_identifiers_before_database_operations(
    store: QdrantVectorStore,
    client: Mock,
    tenant_id: str,
    document_id: str,
) -> None:
    with pytest.raises(ValueError):
        store.delete_document(
            tenant_id=tenant_id,
            document_id=document_id,
        )

    client.delete.assert_not_called()


def test_wraps_stale_point_deletion_failure(
    store: QdrantVectorStore,
    client: Mock,
    embedded_document: EmbeddedDocument,
) -> None:
    client.scroll.return_value = (
        [SimpleNamespace(id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")],
        None,
    )

    client.delete.side_effect = RuntimeError("Delete failed")

    with pytest.raises(
        VectorStoreOperationError,
        match="stale points could not be deleted",
    ):
        store.upsert_document(
            embedded_document,
            tenant_id="tenant-1",
            document_id="doc-1",
        )

    # The new points were already upserted successfully.
    client.upsert.assert_called_once()


def make_search_point(
    *,
    document_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    chunk_index: int = 0,
    text: str = "Employee security policy.",
    score: float = 0.95,
) -> MagicMock:
    point = MagicMock()

    point.payload = {
        "tenant_id": "tenant-1",
        "document_id": document_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "text": text,
        "source_name": "policy.pdf",
        "page_numbers": [1],
        "metadata": {"department": "engineering"},
    }

    point.score = score

    return point


def test_search_returns_validated_results(store, mock_client):
    mock_client.collection_exists.return_value = True
    mock_client.query_points.return_value.points = [
        make_search_point(),
        make_search_point(
            chunk_id="chunk-2",
            chunk_index=1,
            text="Password management policy.",
            score=0.87,
        ),
    ]

    results = store.search(
        [0.1, 0.2, 0.3],
        tenant_id="tenant-1",
        limit=5,
    )

    assert len(results) == 2
    assert all(isinstance(item, VectorSearchResult) for item in results)

    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.95
    assert results[1].chunk_id == "chunk-2"

    mock_client.query_points.assert_called_once()


def test_search_always_applies_tenant_filter(store, mock_client):
    mock_client.collection_exists.return_value = True
    mock_client.query_points.return_value.points = []

    store.search(
        [0.1, 0.2],
        tenant_id="tenant-1",
    )

    query_filter = mock_client.query_points.call_args.kwargs["query_filter"]

    assert len(query_filter.must) == 1
    assert query_filter.must[0].key == "tenant_id"
    assert query_filter.must[0].match.value == "tenant-1"


def test_search_applies_optional_document_filter(store, mock_client):
    mock_client.collection_exists.return_value = True
    mock_client.query_points.return_value.points = []

    store.search(
        [0.1, 0.2],
        tenant_id="tenant-1",
        document_id="doc-1",
    )

    query_filter = mock_client.query_points.call_args.kwargs["query_filter"]

    assert len(query_filter.must) == 2

    filter_values = {condition.key: condition.match.value for condition in query_filter.must}

    assert filter_values == {
        "tenant_id": "tenant-1",
        "document_id": "doc-1",
    }


def test_search_returns_empty_when_collection_does_not_exist(
    store,
    mock_client,
):
    mock_client.collection_exists.return_value = False

    results = store.search(
        [0.1, 0.2],
        tenant_id="tenant-1",
    )

    assert results == []
    mock_client.query_points.assert_not_called()


@pytest.mark.parametrize(
    "query_vector",
    [
        [],
        [0.1, float("nan")],
        [0.1, float("inf")],
        [0.1, float("-inf")],
        [0.1, True],
        [0.1, "invalid"],
        "invalid",
        None,
    ],
)
def test_search_rejects_invalid_query_vectors(store, query_vector):
    with pytest.raises((ValueError, TypeError)):
        store.search(
            query_vector,
            tenant_id="tenant-1",
        )


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
def test_search_rejects_invalid_limits(store, limit):
    with pytest.raises(ValueError):
        store.search(
            [0.1, 0.2],
            tenant_id="tenant-1",
            limit=limit,
        )


def test_search_rejects_empty_tenant_id(store):
    with pytest.raises(ValueError):
        store.search(
            [0.1, 0.2],
            tenant_id="",
        )


def test_search_wraps_qdrant_errors(store, mock_client):
    mock_client.collection_exists.return_value = True
    mock_client.query_points.side_effect = RuntimeError("Qdrant unavailable")

    with pytest.raises(VectorStoreOperationError):
        store.search(
            [0.1, 0.2],
            tenant_id="tenant-1",
        )


def test_search_rejects_malformed_payload(store, mock_client):
    mock_client.collection_exists.return_value = True

    point = make_search_point()
    del point.payload["chunk_id"]

    mock_client.query_points.return_value.points = [point]

    with pytest.raises(VectorStoreOperationError):
        store.search(
            [0.1, 0.2],
            tenant_id="tenant-1",
        )


def test_search_passes_query_configuration_to_qdrant(store, mock_client):
    mock_client.collection_exists.return_value = True
    mock_client.query_points.return_value.points = []

    store.search(
        [0.1, 0.2, 0.3],
        tenant_id="tenant-1",
        limit=7,
    )

    kwargs = mock_client.query_points.call_args.kwargs

    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert kwargs["limit"] == 7
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False


def _verification_record(tenant_id: str, document_id: str, chunk_id: str):
    return SimpleNamespace(
        id=generate_point_id(
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_id=chunk_id,
        ),
        payload={
            "tenant_id": tenant_id,
            "document_id": document_id,
            "chunk_id": chunk_id,
        },
    )


def test_verify_document_confirms_exact_chunk_set(
    store: QdrantVectorStore,
    client: Mock,
    embedded_document: EmbeddedDocument,
) -> None:
    chunk_ids = tuple(item.chunk.chunk_id for item in embedded_document.chunks)
    client.collection_exists.return_value = True
    client.scroll.return_value = (
        [_verification_record("tenant-1", "doc-1", chunk_id) for chunk_id in chunk_ids],
        None,
    )

    result = store.verify_document(
        tenant_id="tenant-1",
        document_id="doc-1",
        expected_chunk_ids=chunk_ids,
    )

    assert result.is_verified is True
    assert result.expected_chunk_ids == chunk_ids
    assert result.verified_chunk_ids == chunk_ids
    assert result.missing_chunk_ids == ()
    assert result.unexpected_chunk_ids == ()

    kwargs = client.scroll.call_args.kwargs
    assert kwargs["with_payload"] == ["tenant_id", "document_id", "chunk_id"]
    assert kwargs["with_vectors"] is False


def test_verify_document_reports_missing_chunks(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    client.collection_exists.return_value = True
    client.scroll.return_value = (
        [_verification_record("tenant-1", "doc-1", "chunk-1")],
        None,
    )

    result = store.verify_document(
        tenant_id="tenant-1",
        document_id="doc-1",
        expected_chunk_ids=("chunk-1", "chunk-2"),
    )

    assert result.is_verified is False
    assert result.verified_chunk_ids == ("chunk-1",)
    assert result.missing_chunk_ids == ("chunk-2",)
    assert result.unexpected_chunk_ids == ()


def test_verify_document_reports_unexpected_stale_chunks(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    client.collection_exists.return_value = True
    client.scroll.return_value = (
        [
            _verification_record("tenant-1", "doc-1", "chunk-1"),
            _verification_record("tenant-1", "doc-1", "stale-chunk"),
        ],
        None,
    )

    result = store.verify_document(
        tenant_id="tenant-1",
        document_id="doc-1",
        expected_chunk_ids=("chunk-1",),
    )

    assert result.is_verified is False
    assert result.verified_chunk_ids == ("chunk-1",)
    assert result.missing_chunk_ids == ()
    assert result.unexpected_chunk_ids == ("stale-chunk",)


def test_verify_document_reports_all_chunks_missing_when_collection_absent(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    client.collection_exists.return_value = False

    result = store.verify_document(
        tenant_id="tenant-1",
        document_id="doc-1",
        expected_chunk_ids=("chunk-1", "chunk-2"),
    )

    assert result.is_verified is False
    assert result.verified_chunk_ids == ()
    assert result.missing_chunk_ids == ("chunk-1", "chunk-2")
    client.scroll.assert_not_called()


def test_verify_document_supports_multiple_scroll_pages(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    client.collection_exists.return_value = True
    client.scroll.side_effect = [
        ([_verification_record("tenant-1", "doc-1", "chunk-1")], "next-page"),
        ([_verification_record("tenant-1", "doc-1", "chunk-2")], None),
    ]

    result = store.verify_document(
        tenant_id="tenant-1",
        document_id="doc-1",
        expected_chunk_ids=("chunk-1", "chunk-2"),
    )

    assert result.is_verified is True
    assert client.scroll.call_count == 2
    assert client.scroll.call_args_list[1].kwargs["offset"] == "next-page"


def test_verify_document_wraps_qdrant_errors(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    client.collection_exists.return_value = True
    client.scroll.side_effect = RuntimeError("Qdrant unavailable")

    with pytest.raises(VectorStoreOperationError, match="Failed to verify"):
        store.verify_document(
            tenant_id="tenant-1",
            document_id="doc-1",
            expected_chunk_ids=("chunk-1",),
        )


def test_verify_document_rejects_identity_mismatch(
    store: QdrantVectorStore,
    client: Mock,
) -> None:
    record = _verification_record("tenant-1", "doc-1", "chunk-1")
    record.payload["tenant_id"] = "another-tenant"
    client.collection_exists.return_value = True
    client.scroll.return_value = ([record], None)

    with pytest.raises(VectorStoreOperationError, match="payload identity"):
        store.verify_document(
            tenant_id="tenant-1",
            document_id="doc-1",
            expected_chunk_ids=("chunk-1",),
        )


@pytest.mark.parametrize(
    "expected_chunk_ids",
    [
        (),
        ("chunk-1", "chunk-1"),
        ("",),
        (" chunk-1",),
        "chunk-1",
    ],
)
def test_verify_document_rejects_invalid_expected_chunk_ids(
    store: QdrantVectorStore,
    expected_chunk_ids,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        store.verify_document(
            tenant_id="tenant-1",
            document_id="doc-1",
            expected_chunk_ids=expected_chunk_ids,
        )
