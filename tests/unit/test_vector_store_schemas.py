import pytest
from pydantic import ValidationError

from app.vector_store.schemas import VectorSearchResult


def test_creates_vector_search_result():
    result = VectorSearchResult(
        document_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=0,
        text="Employee security policies.",
        source_name="policy.pdf",
        page_numbers=(1, 2),
        metadata={"department": "engineering"},
        score=0.95,
    )

    assert result.document_id == "doc-1"
    assert result.score == 0.95
    assert result.page_numbers == (1, 2)


def test_rejects_empty_document_id():
    with pytest.raises(ValidationError):
        VectorSearchResult(
            document_id="",
            chunk_id="chunk-1",
            chunk_index=0,
            text="Some text",
            source_name="policy.pdf",
            score=0.95,
        )


def test_result_is_immutable():
    result = VectorSearchResult(
        document_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=0,
        text="Some text",
        source_name="policy.pdf",
        score=0.95,
    )

    with pytest.raises(ValidationError):
        result.score = 0.5
