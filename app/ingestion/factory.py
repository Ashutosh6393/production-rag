from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.formats import DocumentFormat
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.registry import ParserRegistry
from app.ingestion.service import IngestionService


def create_ingestion_service() -> IngestionService:
    """Create an ingestion service with the application's registered parsers."""

    registry = ParserRegistry()

    registry.register(
        DocumentFormat.PDF,
        PDFParser(),
    )

    ## register the other parsers here

    return IngestionService(
        detector=DocumentFormatDetector(),
        registry=registry,
    )