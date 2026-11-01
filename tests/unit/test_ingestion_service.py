from pathlib import Path

import pymupdf
import pytest

from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.factory import create_ingestion_service
from app.ingestion.registry import ParserRegistry
from app.ingestion.service import IngestionService


def test_process_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Employee security policies must be followed.",
        )
        document.save(pdf_path)

    service = create_ingestion_service()

    extraction, chunks = service.process(pdf_path)

    assert extraction.page_count == 1
    assert len(chunks) == 1
    assert "Employee security policies" in chunks[0].text
    assert chunks[0].page_numbers == (1,)


def test_detector_is_required() -> None:
    with pytest.raises(TypeError):
        IngestionService(registry=ParserRegistry())


def test_registry_is_required() -> None:
    with pytest.raises(TypeError):
        IngestionService(detector=DocumentFormatDetector())


@pytest.mark.parametrize("dependency", ["detector", "registry"])
def test_dependencies_cannot_be_none(dependency: str) -> None:
    kwargs = {
        "detector": DocumentFormatDetector(),
        "registry": ParserRegistry(),
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
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("overlap", [-1, -10, -100])
def test_overlap_cannot_be_negative(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap cannot be negative"):
        IngestionService(
            detector=DocumentFormatDetector(),
            registry=ParserRegistry(),
            overlap=overlap,
        )


@pytest.mark.parametrize("overlap", [1000, 1500, 2000])
def test_overlap_must_be_smaller_than_chunk_size(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        IngestionService(
            detector=DocumentFormatDetector(),
            registry=ParserRegistry(),
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
        parameter: value,
    }

    with pytest.raises(TypeError):
        IngestionService(**kwargs)


def test_dependencies_are_injected_correctly() -> None:
    detector = DocumentFormatDetector()
    registry = ParserRegistry()

    service = IngestionService(
        detector=detector,
        registry=registry,
    )

    assert service.detector is detector
    assert service.registry is registry


def test_chunk_configuration_is_stored_correctly() -> None:
    service = IngestionService(
        detector=DocumentFormatDetector(),
        registry=ParserRegistry(),
        chunk_size=500,
        overlap=100,
    )

    assert service.chunk_size == 500
    assert service.overlap == 100