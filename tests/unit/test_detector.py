from pathlib import Path

import pymupdf
import pytest

from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.exceptions import (
    DocumentNotFoundError,
    InvalidDocumentError,
    UnsupportedDocumentError,
)
from app.ingestion.formats import DocumentFormat


@pytest.fixture
def detector() -> DocumentFormatDetector:
    return DocumentFormatDetector()


@pytest.fixture
def valid_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    return pdf_path


def test_detects_valid_pdf(
    detector: DocumentFormatDetector,
    valid_pdf: Path,
) -> None:
    assert detector.detect(valid_pdf) == DocumentFormat.PDF


def test_rejects_pdf_extension_with_non_pdf_content(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("This is not a PDF", encoding="utf-8")

    with pytest.raises(InvalidDocumentError, match="content is not a PDF"):
        detector.detect(fake_pdf)


def test_rejects_unsupported_extension(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "program.exe"
    file_path.write_bytes(b"%PDF-1.7")

    with pytest.raises(UnsupportedDocumentError):
        detector.detect(file_path)


def test_raises_error_for_missing_file(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    with pytest.raises(DocumentNotFoundError):
        detector.detect(tmp_path / "missing.pdf")


def test_rejects_directory(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    with pytest.raises(InvalidDocumentError):
        detector.detect(tmp_path)