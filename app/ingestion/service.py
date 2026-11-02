from pathlib import Path

from app.ingestion.chunking.splitter import split_document
from app.ingestion.detector import DocumentFormatDetector
from app.ingestion.embeddings.schemas import EmbeddedChunk, EmbeddedDocument
from app.ingestion.embeddings.service import EmbeddingService
from app.ingestion.exceptions import EmptyDocumentError
from app.ingestion.registry import ParserRegistry
from app.ingestion.schemas import IngestionResult


class IngestionService:
    """Coordinates document detection, extraction, chunking, and embedding."""

    def __init__(
        self,
        detector: DocumentFormatDetector,
        registry: ParserRegistry,
        embedding_service: EmbeddingService,
        *,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        if detector is None:
            raise TypeError("detector cannot be None")

        if registry is None:
            raise TypeError("registry cannot be None")

        if embedding_service is None:
            raise TypeError("embedding_service cannot be None")

        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if not isinstance(overlap, int) or isinstance(overlap, bool):
            raise TypeError("overlap must be an integer")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.detector = detector
        self.registry = registry
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process(self, source: Path) -> IngestionResult:
        """Extract, chunk, and generate embeddings for a document."""

        document_format = self.detector.detect(source)
        parser = self.registry.get(document_format)

        extraction = parser.parse(source)

        chunks = split_document(
            extraction,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        if not chunks:
            raise EmptyDocumentError(f"No extractable text found in document: {source.name}")

        embedding_result = self.embedding_service.embed([chunk.text for chunk in chunks])

        if len(embedding_result.vectors) != len(chunks):
            raise RuntimeError("Embedding result count does not match the number of document chunks.")

        embedded_chunks = tuple(
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
            )
            for chunk, vector in zip(
                chunks,
                embedding_result.vectors,
                strict=True,
            )
        )

        embedded_document = EmbeddedDocument(
            model_name=embedding_result.model_name,
            chunks=embedded_chunks,
        )

        return IngestionResult(
            extraction=extraction,
            embedding=embedded_document,
        )
