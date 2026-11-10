import pytest
from pydantic import ValidationError

from app.vector_store.schemas import IndexVerificationResult, VectorSearchResult


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


def test_index_verification_result_reports_success() -> None:
    result = IndexVerificationResult(
        expected_chunk_ids=("chunk-1", "chunk-2"),
        verified_chunk_ids=("chunk-1", "chunk-2"),
        missing_chunk_ids=(),
    )

    assert result.is_verified is True


def test_index_verification_result_reports_missing_chunks() -> None:
    result = IndexVerificationResult(
        expected_chunk_ids=("chunk-1", "chunk-2"),
        verified_chunk_ids=("chunk-1",),
        missing_chunk_ids=("chunk-2",),
    )

    assert result.is_verified is False


@pytest.mark.parametrize(
    "values",
    [
        {
            "expected_chunk_ids": ("chunk-1", "chunk-1"),
            "verified_chunk_ids": ("chunk-1",),
            "missing_chunk_ids": (),
        },
        {
            "expected_chunk_ids": ("chunk-1",),
            "verified_chunk_ids": ("chunk-1",),
            "missing_chunk_ids": ("chunk-1",),
        },
        {
            "expected_chunk_ids": ("chunk-1", "chunk-2"),
            "verified_chunk_ids": ("chunk-1",),
            "missing_chunk_ids": (),
        },
    ],
)
def test_index_verification_result_rejects_inconsistent_chunk_sets(values) -> None:
    with pytest.raises(ValidationError):
        IndexVerificationResult(**values)
