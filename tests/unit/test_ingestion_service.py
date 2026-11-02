from collections.abc import Sequence
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.batching import TokenBatcher
from app.ingestion.embeddings.schemas import EmbeddingBatchResult
from app.ingestion.embeddings.service import EmbeddingService
from app.ingestion.factory import create_ingestion_service
from app.ingestion.registry import ParserRegistry
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

    embedded_chunks = result.embedding.chunks

    assert len(embedded_chunks) == 1
    assert "Employee security policies" in embedded_chunks[0].chunk.text
    assert embedded_chunks[0].chunk.page_numbers == (1,)
    assert embedded_chunks[0].vector == (0.1, 0.2, 0.3)
    assert result.embedding.model_name == "test-embedding-model"
    assert result.embedding.dimensions == 3


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
