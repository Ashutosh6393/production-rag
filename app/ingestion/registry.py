from app.ingestion.formats import DocumentFormat
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.pdf import PDFParser


class ParserRegistry:
    """Maps supported document formats to parser implementations."""

    def __init__(self) -> None:
        self._parsers: dict[DocumentFormat, DocumentParser] = {
            DocumentFormat.PDF: PDFParser(),
        }

    def get(self, document_format: DocumentFormat) -> DocumentParser:
        try:
            return self._parsers[document_format]
        except KeyError:
            raise ValueError(
                f"No parser registered for format: {document_format.value}"
            ) from None