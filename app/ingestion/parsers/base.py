from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.schemas import ExtractionResult


class DocumentParser(ABC):
    """Contract for document format parsers."""

    @abstractmethod
    def parse(self, source: Path) -> ExtractionResult:
        """Extract structured text from a document file."""

        raise NotImplementedError("Subclasses must implement this method.")
