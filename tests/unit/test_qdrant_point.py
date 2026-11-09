
import pytest
from qdrant_client.models import PointStruct

from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.embeddings.schemas import EmbeddedChunk
from app.vector_store.qdrant.point import (
    build_point,
    generate_point_id,
)


@pytest.fixture
def embedded_chunk() -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=DocumentChunk(
            chunk_id="chunk-1",
            chunk_index=0,
            text="Employee security policies must be followed.",
            source_name="policy.pdf",
            page_numbers=(1,),
            metadata={},
        ),
        vector=(0.1, 0.2, 0.3),
    )


def test_generate_point_id_is_deterministic():
    first = generate_point_id(
        tenant_id="tenant-1",
        document_id="doc-1",
        chunk_id="chunk-1",
    )

    second = generate_point_id(
        tenant_id="tenant-1",
        document_id="doc-1",
        chunk_id="chunk-1",
    )

    assert first == second


def test_generate_point_id_differs_between_tenants():
    first = generate_point_id(
        tenant_id="tenant-1",
        document_id="doc-1",
        chunk_id="chunk-1",
    )

    second = generate_point_id(
        tenant_id="tenant-2",
        document_id="doc-1",
        chunk_id="chunk-1",
    )

    assert first != second


def test_generate_point_id_differs_between_documents():
    first = generate_point_id(
        tenant_id="tenant-1",
        document_id="doc-1",
        chunk_id="chunk-1",
    )

    second = generate_point_id(
        tenant_id="tenant-1",
        document_id="doc-2",
        chunk_id="chunk-1",
    )

    assert first != second


def test_build_point_creates_correct_point(embedded_chunk: EmbeddedChunk):
    point = build_point(
        embedded_chunk,
        tenant_id="tenant-1",
        document_id="doc-1",
        embedding_model="text-embedding-3-small",
    )

    assert isinstance(point, PointStruct)
    assert point.id == generate_point_id(
        tenant_id="tenant-1",
        document_id="doc-1",
        chunk_id=embedded_chunk.chunk.chunk_id,
    )

    assert point.vector == list(embedded_chunk.vector)

    assert point.payload["tenant_id"] == "tenant-1"
    assert point.payload["document_id"] == "doc-1"
    assert point.payload["chunk_id"] == embedded_chunk.chunk.chunk_id
    assert point.payload["embedding_model"] == "text-embedding-3-small"
    assert point.payload["text"] == embedded_chunk.chunk.text


def test_generate_point_id_rejects_empty_identifiers():
    with pytest.raises(ValueError):
        generate_point_id(
            tenant_id="",
            document_id="doc-1",
            chunk_id="chunk-1",
        )
