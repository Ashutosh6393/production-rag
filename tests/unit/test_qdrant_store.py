
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client.models import Filter, FilterSelector, PointIdsList

from app.config import Settings
from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddedDocument
from app.vector_store.exceptions import VectorStoreOperationError
from app.vector_store.qdrant.store import QdrantVectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def client() -> Mock:
    return Mock()


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

    collection_manager.ensure_collection.assert_called_once_with(
        vector_size=embedded_document.dimensions
    )

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
