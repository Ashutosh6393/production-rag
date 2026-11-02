from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.batching import TokenBatcher
from app.ingestion.embeddings.provider import OpenAIEmbeddingProvider
from app.ingestion.embeddings.service import EmbeddingService
from app.ingestion.formats import DocumentFormat
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.registry import ParserRegistry
from app.ingestion.service import IngestionService


def create_ingestion_service(
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> IngestionService:
    """Build the ingestion service with its required dependencies."""

    registry = ParserRegistry()
    registry.register(DocumentFormat.PDF, PDFParser())

    provider = embedding_provider or OpenAIEmbeddingProvider()

    embedding_service = EmbeddingService(
        provider=provider,
        batcher=TokenBatcher(),
    )

    return IngestionService(
        detector=DocumentFormatDetector(),
        registry=registry,
        embedding_service=embedding_service,
    )
