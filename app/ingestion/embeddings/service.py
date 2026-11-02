from collections.abc import Sequence

from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.batching import TokenBatcher
from app.ingestion.embeddings.exceptions import EmbeddingProviderError
from app.ingestion.embeddings.schemas import EmbeddingBatchResult


class EmbeddingService:
    """Coordinates batching and embedding generation."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        batcher: TokenBatcher,
    ) -> None:
        if provider is None:
            raise TypeError("provider cannot be None")

        if batcher is None:
            raise TypeError("batcher cannot be None")

        if not isinstance(provider, EmbeddingProvider):
            raise TypeError("provider must implement EmbeddingProvider")

        self.provider = provider
        self.batcher = batcher

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        """Generate embeddings for all supplied texts."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        if not texts:
            raise ValueError("At least one text is required")

        batches = self.batcher.split(texts)

        all_vectors: list[tuple[float, ...]] = []
        model_name: str | None = None
        dimensions: int | None = None

        for batch in batches:
            result = self.provider.embed(batch)

            if not isinstance(result, EmbeddingBatchResult):
                raise EmbeddingProviderError("Provider returned an invalid result type.")

            if len(result.vectors) != len(batch):
                raise EmbeddingProviderError("Provider returned an unexpected number of embeddings.")

            if model_name is None:
                model_name = result.model_name

            elif result.model_name != model_name:
                raise EmbeddingProviderError("Provider returned inconsistent model names across batches.")

            if dimensions is None:
                dimensions = result.dimensions

            elif result.dimensions != dimensions:
                raise EmbeddingProviderError("Provider returned inconsistent vector dimensions across batches.")

            all_vectors.extend(result.vectors)

        if model_name is None:
            raise EmbeddingProviderError("Embedding generation produced no results.")

        return EmbeddingBatchResult(
            model_name=model_name,
            vectors=tuple(all_vectors),
        )
