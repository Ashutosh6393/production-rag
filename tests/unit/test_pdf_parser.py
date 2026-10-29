from pathlib import Path

import pymupdf
import pytest

from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.schemas import ExtractionResult


@pytest.fixture
def parser() -> PDFParser:
    return PDFParser()


@pytest.fixture
def valid_pdf(tmp_path: Path) -> Path:
    """Create a valid PDF containing two pages of text."""
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "First page content.")

        second_page = document.new_page()
        second_page.insert_text((72, 72), "Second page content.")

        document.set_metadata(
            {
                "title": "Test Document",
                "author": "Test Author",
            }
        )
        document.save(pdf_path)

    return pdf_path


def test_extracts_text_and_page_numbers(
    parser: PDFParser,
    valid_pdf: Path,
) -> None:
    result = parser.parse(valid_pdf)

    assert isinstance(result, ExtractionResult)
    assert result.source_name == "sample.pdf"
    assert result.source_format == "pdf"
    assert result.page_count == 2

    assert result.pages[0].page_number == 1
    assert "First page content." in result.pages[0].text

    assert result.pages[1].page_number == 2
    assert "Second page content." in result.pages[1].text


def test_extracts_document_metadata(
    parser: PDFParser,
    valid_pdf: Path,
) -> None:
    result = parser.parse(valid_pdf)

    assert result.metadata["title"] == "Test Document"
    assert result.metadata["author"] == "Test Author"


def test_full_text_combines_pages(
    parser: PDFParser,
    valid_pdf: Path,
) -> None:
    result = parser.parse(valid_pdf)

    assert "First page content." in result.full_text
    assert "Second page content." in result.full_text


def test_missing_file_raises_error(
    parser: PDFParser,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.pdf")


def test_directory_is_rejected(
    parser: PDFParser,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not a file"):
        parser.parse(tmp_path)


def test_unsupported_extension_is_rejected(
    parser: PDFParser,
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "document.txt"
    text_file.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        parser.parse(text_file)


def test_empty_file_is_rejected(
    parser: PDFParser,
    tmp_path: Path,
) -> None:
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        parser.parse(empty_pdf)


def test_corrupt_pdf_is_rejected(
    parser: PDFParser,
    tmp_path: Path,
) -> None:
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.7\nnot a valid PDF")

    with pytest.raises(ValueError):
        parser.parse(corrupt_pdf)


def test_textless_pages_produce_warnings(
    parser: PDFParser,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "image_only.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    result = parser.parse(pdf_path)

    assert result.page_count == 1
    assert result.pages[0].page_number == 1
    assert result.pages[0].text == ""
    assert any("no extractable text" in warning for warning in result.warnings)
    assert any("OCR" in warning for warning in result.warnings)



def test_zero_page_pdf_is_rejected(
    parser: PDFParser,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "zero_pages.pdf"

    try:
        with pymupdf.open() as document:
            document.save(pdf_path)
    except ValueError as exc:
        pytest.skip(
            f"Installed PyMuPDF cannot create a zero-page PDF: {exc}"
        )

    result = parser.parse(pdf_path)

    assert result.page_count == 0
    assert result.pages == ()
    assert any("no pages" in warning for warning in result.warnings)
