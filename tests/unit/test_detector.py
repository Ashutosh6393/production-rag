import zipfile
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


def test_detects_valid_docx(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    docx_path = tmp_path / "sample.docx"

    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("word/document.xml", "<document></document>")

    assert detector.detect(docx_path) == DocumentFormat.DOCX


def test_rejects_invalid_docx(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    docx_path = tmp_path / "fake.docx"
    docx_path.write_text("This is not a DOCX file", encoding="utf-8")

    with pytest.raises(InvalidDocumentError, match="valid DOCX package"):
        detector.detect(docx_path)


def test_detects_valid_txt(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text(
        "Employee security policy",
        encoding="utf-8",
    )

    assert detector.detect(txt_path) == DocumentFormat.TXT


def test_detects_empty_txt(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    txt_path = tmp_path / "empty.txt"
    txt_path.write_bytes(b"")

    assert detector.detect(txt_path) == DocumentFormat.TXT


def test_rejects_invalid_utf8_txt(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    txt_path = tmp_path / "invalid.txt"
    txt_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(InvalidDocumentError, match="UTF-8"):
        detector.detect(txt_path)


def test_rejects_binary_content_in_txt(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    txt_path = tmp_path / "binary.txt"
    txt_path.write_bytes(b"Hello\x00World")

    with pytest.raises(InvalidDocumentError):
        detector.detect(txt_path)


def test_detects_valid_html(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "sample.html"

    html_path.write_text(
        "<!DOCTYPE html><html><body><h1>Company Policy</h1></body></html>",
        encoding="utf-8",
    )

    assert detector.detect(html_path) == DocumentFormat.HTML


def test_detects_htm_extension(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "sample.htm"

    html_path.write_text(
        "<html><body>Company Policy</body></html>",
        encoding="utf-8",
    )

    assert detector.detect(html_path) == DocumentFormat.HTML


def test_rejects_plain_text_with_html_extension(
    detector: DocumentFormatDetector,
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "fake.html"
    html_path.write_text(
        "This is just plain text.",
        encoding="utf-8",
    )

    with pytest.raises(InvalidDocumentError, match="HTML markup"):
        detector.detect(html_path)
