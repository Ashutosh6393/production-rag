import pytest

from app.ingestion.exceptions import UnsupportedDocumentError
from app.ingestion.formats import DocumentFormat
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.registry import ParserRegistry


def test_returns_pdf_parser() -> None:
    registry = ParserRegistry()
    parser = PDFParser()

    registry.register(DocumentFormat.PDF, parser)

    assert isinstance(registry.get(DocumentFormat.PDF), PDFParser)


def test_returns_document_parser_interface() -> None:
    registry = ParserRegistry()
    registry.register(DocumentFormat.PDF, PDFParser())

    parser = registry.get(DocumentFormat.PDF)

    assert isinstance(parser, DocumentParser)


def test_raises_error_when_parser_is_not_registered() -> None:
    registry = ParserRegistry()

    with pytest.raises(UnsupportedDocumentError):
        registry.get(DocumentFormat.DOCX)


def test_registers_parser() -> None:
    registry = ParserRegistry()
    parser = PDFParser()

    registry.register(DocumentFormat.PDF, parser)

    assert registry.get(DocumentFormat.PDF) is parser


def test_rejects_duplicate_registration() -> None:
    registry = ParserRegistry()

    registry.register(DocumentFormat.PDF, PDFParser())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DocumentFormat.PDF, PDFParser())