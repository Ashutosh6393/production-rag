import pytest

from app.ingestion.chunking.splitter import split_document
from app.ingestion.schemas import ExtractedPage, ExtractionResult


@pytest.fixture
def extraction() -> ExtractionResult:
    return ExtractionResult(
        source_name="policy.pdf",
        source_format="pdf",
        page_count=2,
        pages=(
            ExtractedPage(
                page_number=1,
                text="First paragraph.\n\nSecond paragraph.",
            ),
            ExtractedPage(
                page_number=2,
                text="Third paragraph.",
            ),
        ),
    )


def test_splits_document_into_chunks() -> None:
    result = ExtractionResult(
        source_name="long.pdf",
        source_format="pdf",
        page_count=1,
        pages=(
            ExtractedPage(
                page_number=1,
                text="A" * 250,
            ),
        ),
    )

    chunks = split_document(result, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_preserves_page_numbers(
    extraction: ExtractionResult,
) -> None:
    chunks = split_document(extraction, chunk_size=100, overlap=10)

    assert chunks[0].page_numbers == (1,)
    assert chunks[-1].page_numbers == (2,)


def test_chunk_ids_are_deterministic(
    extraction: ExtractionResult,
) -> None:
    first = split_document(extraction)
    second = split_document(extraction)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_chunk_indices_are_sequential(
    extraction: ExtractionResult,
) -> None:
    chunks = split_document(extraction)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_empty_pages_produce_no_chunks() -> None:
    result = ExtractionResult(
        source_name="empty.pdf",
        source_format="pdf",
        page_count=1,
        pages=(ExtractedPage(page_number=1, text="  \n  "),),
    )

    assert split_document(result) == []


def test_invalid_chunk_size_is_rejected(
    extraction: ExtractionResult,
) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        split_document(extraction, chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size(
    extraction: ExtractionResult,
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_document(extraction, chunk_size=100, overlap=100)
