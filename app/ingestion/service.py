from pathlib import Path

from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.chunking.splitter import split_document
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.schemas import ExtractionResult
from app.ingestion.parsers.base import DocumentParser


class IngestionService:
    """Coordinates document extraction and chunking."""

    def __init__(self, parser: DocumentParser | None = None, *, chunk_size: int = 1000, overlap: int = 150) -> None:
        self.parser = parser if parser is not None else PDFParser()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process(
        self,
        source: Path,
    ) -> tuple[ExtractionResult, list[DocumentChunk]]:
        """Extract and chunk a document."""

        extraction = self.parser.parse(source)

        chunks = split_document(
            extraction,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        return extraction, chunks
