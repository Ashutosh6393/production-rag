from pathlib import Path

from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.chunking.splitter import split_document
from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.registry import ParserRegistry
from app.ingestion.schemas import ExtractionResult


class IngestionService:
    """Coordinates document detection, extraction, and chunking."""

    def __init__(
        self,
        detector: DocumentFormatDetector | None = None,
        registry: ParserRegistry | None = None,
        *,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        self.detector = detector if detector is not None else DocumentFormatDetector()
        self.registry = registry if registry is not None else ParserRegistry()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process(
        self,
        source: Path,
    ) -> tuple[ExtractionResult, list[DocumentChunk]]:
        """Detect, extract, and chunk a supported document."""

        document_format = self.detector.detect(source)

        parser = self.registry.get(document_format)

        extraction = parser.parse(source)

        chunks = split_document(
            extraction,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        return extraction, chunks
