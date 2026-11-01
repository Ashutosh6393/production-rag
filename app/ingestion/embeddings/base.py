from abc import ABC, abstractmethod
from collections.abc import Sequence


from app.ingestion.embeddings.schemas import EmbeddingBatchResult


class EmbeddingProvider(ABC):
    """Contract for embedding providers."""

    @abstractmethod
    def embed(
        self,
        texts: Sequence[str],

    )-> EmbeddingBatchResult:
        """Generates embeddings for a batch of texts."""

        raise NotImplementedError("Subclasses must implement this method.")