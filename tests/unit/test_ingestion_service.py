from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pymupdf
import pytest

from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.batching import TokenBatcher
from app.ingestion.embeddings.exceptions import EmbeddingProviderError
from app.ingestion.embeddings.schemas import EmbeddingBatchResult
from app.ingestion.embeddings.service import EmbeddingService
from app.ingestion.exceptions import EmptyDocumentError
from app.ingestion.factory import create_ingestion_service
from app.ingestion.registry import ParserRegistry
from app.ingestion.schemas import ExtractedPage, ExtractionResult
from app.ingestion.service import IngestionService


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for tests."""

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        return EmbeddingBatchResult(
            model_name="test-embedding-model",
            vectors=tuple((0.1, 0.2, 0.3) for _ in texts),
        )


def make_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        provider=FakeEmbeddingProvider(),
        batcher=TokenBatcher(),
    )


def make_extraction(text: str) -> ExtractionResult:
    """Create a minimal extraction result for ingestion tests."""

    return ExtractionResult(
        source_name="policy.pdf",
        source_format="pdf",
        page_count=1,
        pages=(
            ExtractedPage(
                page_number=1,
                text=text,
            ),
        ),
    )


def make_chunk(
    chunk_id: str,
    chunk_index: int,
    text: str,
) -> DocumentChunk:
    """Create a document chunk for ingestion tests."""

    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        text=text,
        source_name="policy.pdf",
        page_numbers=(1,),
        metadata={"source_format": "pdf"},
    )


def make_test_ingestion_service(
    extraction: ExtractionResult,
) -> tuple[IngestionService, Mock]:
    """Create an ingestion service with mocked external dependencies."""

    detector = Mock(spec=DocumentFormatDetector)
    detector.detect.return_value = "pdf"

    parser = Mock()
    parser.parse.return_value = extraction

    registry = Mock(spec=ParserRegistry)
    registry.get.return_value = parser

    embedding_service = Mock(spec=EmbeddingService)

    service = IngestionService(
        detector=detector,
        registry=registry,
        embedding_service=embedding_service,
        chunk_size=1000,
        overlap=0,
    )

    return service, embedding_service


def test_empty_document_does_not_call_embedding_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = make_extraction("   ")

    service, embedding_service = make_test_ingestion_service(extraction)

    monkeypatch.setattr(
        "app.ingestion.service.split_document",
        lambda *args, **kwargs: [],
    )

    with pytest.raises(EmptyDocumentError):
        service.process(Path("policy.pdf"))

    embedding_service.embed.assert_not_called()


def test_embedding_service_receives_chunk_texts_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = make_extraction("Some extracted document text.")

    chunks = [
        make_chunk("chunk-001", 0, "First chunk text."),
        make_chunk("chunk-002", 1, "Second chunk text."),
        make_chunk("chunk-003", 2, "Third chunk text."),
    ]

    service, embedding_service = make_test_ingestion_service(extraction)

    embedding_service.embed.return_value = EmbeddingBatchResult(
        model_name="test-model",
        vectors=(
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
            (0.7, 0.8, 0.9),
        ),
    )

    monkeypatch.setattr(
        "app.ingestion.service.split_document",
        lambda *args, **kwargs: chunks,
    )

    service.process(Path("policy.pdf"))

    embedding_service.embed.assert_called_once_with(
        [
            "First chunk text.",
            "Second chunk text.",
            "Third chunk text.",
        ]
    )


def test_vectors_are_associated_with_correct_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = make_extraction("Some extracted document text.")

    chunks = [
        make_chunk("chunk-001", 0, "First chunk text."),
        make_chunk("chunk-002", 1, "Second chunk text."),
    ]

    vectors = (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )

    service, embedding_service = make_test_ingestion_service(extraction)

    embedding_service.embed.return_value = EmbeddingBatchResult(
        model_name="test-model",
        vectors=vectors,
    )

    monkeypatch.setattr(
        "app.ingestion.service.split_document",
        lambda *args, **kwargs: chunks,
    )

    result = service.process(Path("policy.pdf"))

    embedded_chunks = result.embedded_document.chunks

    assert len(embedded_chunks) == 2

    assert embedded_chunks[0].chunk.chunk_id == "chunk-001"
    assert embedded_chunks[0].vector == vectors[0]

    assert embedded_chunks[1].chunk.chunk_id == "chunk-002"
    assert embedded_chunks[1].vector == vectors[1]


def test_multiple_chunks_preserve_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = make_extraction("Some extracted document text.")

    chunks = [
        make_chunk("chunk-001", 0, "First chunk."),
        make_chunk("chunk-002", 1, "Second chunk."),
        make_chunk("chunk-003", 2, "Third chunk."),
    ]

    service, embedding_service = make_test_ingestion_service(extraction)

    embedding_service.embed.return_value = EmbeddingBatchResult(
        model_name="test-model",
        vectors=(
            (0.1, 0.2),
            (0.3, 0.4),
            (0.5, 0.6),
        ),
    )

    monkeypatch.setattr(
        "app.ingestion.service.split_document",
        lambda *args, **kwargs: chunks,
    )

    result = service.process(Path("policy.pdf"))

    embedded_chunks = result.embedded_document.chunks

    assert [item.chunk.chunk_id for item in embedded_chunks] == [
        "chunk-001",
        "chunk-002",
        "chunk-003",
    ]

    assert [item.chunk.chunk_index for item in embedded_chunks] == [
        0,
        1,
        2,
    ]


def test_embedding_vector_count_mismatch_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = make_extraction("Some extracted document text.")

    chunks = [
        make_chunk("chunk-001", 0, "First chunk."),
        make_chunk("chunk-002", 1, "Second chunk."),
    ]

    service, embedding_service = make_test_ingestion_service(extraction)

    # Two chunks, but only one vector returned.
    embedding_service.embed.return_value = EmbeddingBatchResult(
        model_name="test-model",
        vectors=((0.1, 0.2, 0.3),),
    )

    monkeypatch.setattr(
        "app.ingestion.service.split_document",
        lambda *args, **kwargs: chunks,
    )

    with pytest.raises(EmbeddingProviderError, match="vector count"):
        service.process(Path("policy.pdf"))


def test_process_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Employee security policies must be followed.",
        )
        document.save(pdf_path)

    service = create_ingestion_service(
        embedding_provider=FakeEmbeddingProvider(),
    )

    result = service.process(pdf_path)

    assert result.extraction.page_count == 1

    embedded_chunks = result.embedded_document.chunks

    assert len(embedded_chunks) == 1
    assert "Employee security policies" in embedded_chunks[0].chunk.text
    assert embedded_chunks[0].chunk.page_numbers == (1,)
    assert embedded_chunks[0].vector == (0.1, 0.2, 0.3)
    assert result.embedded_document.model_name == "test-embedding-model"
    assert result.embedded_document.dimensions == 3


def test_detector_is_required() -> None:
    with pytest.raises(TypeError):
        IngestionService(
            registry=ParserRegistry(),
            embedding_service=make_embedding_service(),
        )


def test_registry_is_required() -> None:
    with pytest.raises(TypeError):
        IngestionService(
            detector=DocumentFormatDetector(),
            embedding_service=make_embedding_service(),
        )


@pytest.mark.parametrize("dependency", ["detector", "registry", "embedding_service"])
def test_dependencies_cannot_be_none(dependency: str) -> None:
    kwargs = {
        "detector": DocumentFormatDetector(),
        "registry": ParserRegistry(),
        "embedding_service": make_embedding_service(),
    }
    kwargs[dependency] = None

    with pytest.raises(TypeError):
        IngestionService(**kwargs)


@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_chunk_size_must_be_positive(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        IngestionService(
            detector=DocumentFormatDetector(),
            registry=ParserRegistry(),
            embedding_service=make_embedding_service(),
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("overlap", [-1, -10, -100])
def test_overlap_cannot_be_negative(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap cannot be negative"):
        IngestionService(
            detector=DocumentFormatDetector(),
            registry=ParserRegistry(),
            embedding_service=make_embedding_service(),
            overlap=overlap,
        )


@pytest.mark.parametrize("overlap", [1000, 1500, 2000])
def test_overlap_must_be_smaller_than_chunk_size(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        IngestionService(
            detector=DocumentFormatDetector(),
            registry=ParserRegistry(),
            embedding_service=make_embedding_service(),
            chunk_size=1000,
            overlap=overlap,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("chunk_size", True),
        ("chunk_size", 1000.5),
        ("overlap", False),
        ("overlap", 10.5),
    ],
)
def test_chunk_configuration_must_be_integer(
    parameter: str,
    value: object,
) -> None:
    kwargs = {
        "detector": DocumentFormatDetector(),
        "registry": ParserRegistry(),
        "embedding_service": make_embedding_service(),
        parameter: value,
    }

    with pytest.raises(TypeError):
        IngestionService(**kwargs)


def test_dependencies_are_injected_correctly() -> None:
    detector = DocumentFormatDetector()
    registry = ParserRegistry()
    embedding_service = make_embedding_service()

    service = IngestionService(
        detector=detector,
        registry=registry,
        embedding_service=embedding_service,
    )

    assert service.detector is detector
    assert service.registry is registry
    assert service.embedding_service is embedding_service


def test_chunk_configuration_is_stored_correctly() -> None:
    service = IngestionService(
        detector=DocumentFormatDetector(),
        registry=ParserRegistry(),
        embedding_service=make_embedding_service(),
        chunk_size=500,
        overlap=100,
    )

    assert service.chunk_size == 500
    assert service.overlap == 100
