from app.ingestion.exceptions import UnsupportedDocumentError
from app.ingestion.formats import DocumentFormat
from app.ingestion.parsers.base import DocumentParser


class ParserRegistry:
    """Maps supported document formats to parser implementations."""

    def __init__(self) -> None:
        self._parsers: dict[DocumentFormat, DocumentParser] = {}

    def get(self, document_format: DocumentFormat) -> DocumentParser:
        try:
            return self._parsers[document_format]
        except KeyError:
            raise UnsupportedDocumentError(f"No parser registered for format: {document_format.value}") from None
    
    def register(self, document_format: DocumentFormat, parser: DocumentParser) -> None:
        """Register a parser for a document format."""

        if document_format in self._parsers:
            raise ValueError(
                f"A parser is already registered for: {document_format.value}"
            )

        self._parsers[document_format] = parser

