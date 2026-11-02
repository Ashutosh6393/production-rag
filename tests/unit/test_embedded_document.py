import pytest
from pydantic import ValidationError

from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddedDocument


@pytest.fixture
def document_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-001",
        chunk_index=0,
        text="Employee security policies must be followed.",
        source_name="policy.pdf",
        page_numbers=(1,),
        metadata={},
    )


def test_create_embedded_chunk(
    document_chunk: DocumentChunk,
) -> None:
    embedded_chunk = EmbeddedChunk(
        chunk=document_chunk,
        vector=(0.1, 0.2, 0.3),
    )

    assert embedded_chunk.chunk == document_chunk
    assert embedded_chunk.vector == (0.1, 0.2, 0.3)


def test_reject_empty_vector(
    document_chunk: DocumentChunk,
) -> None:
    with pytest.raises(ValidationError):
        EmbeddedChunk(
            chunk=document_chunk,
            vector=(),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_reject_non_finite_vector_values(
    document_chunk: DocumentChunk,
    invalid_value: float,
) -> None:
    with pytest.raises(ValidationError):
        EmbeddedChunk(
            chunk=document_chunk,
            vector=(0.1, invalid_value),
        )


def test_create_embedded_document(
    document_chunk: DocumentChunk,
) -> None:
    embedded_document = EmbeddedDocument(
        model_name="text-embedding-3-small",
        chunks=(
            EmbeddedChunk(
                chunk=document_chunk,
                vector=(0.1, 0.2, 0.3),
            ),
        ),
    )

    assert embedded_document.model_name == "text-embedding-3-small"
    assert embedded_document.dimensions == 3
    assert len(embedded_document.chunks) == 1


def test_reject_inconsistent_dimensions(
    document_chunk: DocumentChunk,
) -> None:
    second_chunk = document_chunk.model_copy(
        update={
            "chunk_id": "chunk-002",
            "chunk_index": 1,
        }
    )

    with pytest.raises(ValidationError, match="identical vector dimensions"):
        EmbeddedDocument(
            model_name="test-model",
            chunks=(
                EmbeddedChunk(
                    chunk=document_chunk,
                    vector=(0.1, 0.2, 0.3),
                ),
                EmbeddedChunk(
                    chunk=second_chunk,
                    vector=(0.4, 0.5),
                ),
            ),
        )


def test_reject_duplicate_chunk_ids(
    document_chunk: DocumentChunk,
) -> None:
    with pytest.raises(ValidationError, match="Duplicate chunk IDs"):
        EmbeddedDocument(
            model_name="test-model",
            chunks=(
                EmbeddedChunk(
                    chunk=document_chunk,
                    vector=(0.1, 0.2),
                ),
                EmbeddedChunk(
                    chunk=document_chunk,
                    vector=(0.3, 0.4),
                ),
            ),
        )


def test_embedded_document_is_immutable(
    document_chunk: DocumentChunk,
) -> None:
    embedded_document = EmbeddedDocument(
        model_name="test-model",
        chunks=(
            EmbeddedChunk(
                chunk=document_chunk,
                vector=(0.1, 0.2),
            ),
        ),
    )

    with pytest.raises(ValidationError):
        embedded_document.model_name = "another-model"
