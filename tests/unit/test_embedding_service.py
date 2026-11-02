from collections.abc import Sequence

import pytest

from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.batching import (
    BatchingConfig,
    TokenBatcher,
)
from app.ingestion.embeddings.exceptions import EmbeddingProviderError
from app.ingestion.embeddings.schemas import EmbeddingBatchResult
from app.ingestion.embeddings.service import EmbeddingService


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic provider for testing embedding orchestration."""

    def __init__(self, model_name: str = "fake-model") -> None:
        self.model_name = model_name
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        self.calls.append(list(texts))

        return EmbeddingBatchResult(
            model_name=self.model_name,
            vectors=tuple((float(index), 0.2, 0.3) for index, _ in enumerate(texts)),
        )


@pytest.fixture
def batcher() -> TokenBatcher:
    config = BatchingConfig(
        max_inputs_per_batch=2,
        max_tokens_per_batch=1000,
        max_tokens_per_input=100,
    )
    return TokenBatcher(config=config)


def test_embedding_service_generates_embeddings(
    batcher: TokenBatcher,
) -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider=provider, batcher=batcher)

    result = service.embed(["First", "Second", "Third"])

    assert result.model_name == "fake-model"
    assert len(result.vectors) == 3
    assert result.dimensions == 3


def test_embedding_service_processes_multiple_batches(
    batcher: TokenBatcher,
) -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider=provider, batcher=batcher)

    service.embed(["A", "B", "C", "D", "E"])

    assert provider.calls == [
        ["A", "B"],
        ["C", "D"],
        ["E"],
    ]


def test_embedding_service_preserves_vector_order(
    batcher: TokenBatcher,
) -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider=provider, batcher=batcher)

    result = service.embed(["A", "B", "C"])

    assert result.vectors == (
        (0.0, 0.2, 0.3),
        (1.0, 0.2, 0.3),
        (0.0, 0.2, 0.3),
    )


def test_embedding_service_rejects_empty_input(
    batcher: TokenBatcher,
) -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider=provider, batcher=batcher)

    with pytest.raises(ValueError, match="At least one text"):
        service.embed([])

    assert provider.calls == []


def test_embedding_service_rejects_string_input(
    batcher: TokenBatcher,
) -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider=provider, batcher=batcher)

    with pytest.raises(TypeError):
        service.embed("Hello")


def test_embedding_service_rejects_missing_provider() -> None:
    with pytest.raises(TypeError, match="provider cannot be None"):
        EmbeddingService(provider=None, batcher=TokenBatcher())


def test_embedding_service_rejects_missing_batcher() -> None:
    with pytest.raises(TypeError, match="batcher cannot be None"):
        EmbeddingService(
            provider=FakeEmbeddingProvider(),
            batcher=None,
        )


def test_embedding_service_rejects_wrong_provider_type() -> None:
    with pytest.raises(TypeError, match="must implement EmbeddingProvider"):
        EmbeddingService(
            provider=object(),
            batcher=TokenBatcher(),
        )


def test_embedding_service_rejects_incorrect_vector_count(
    batcher: TokenBatcher,
) -> None:
    class BrokenProvider(EmbeddingProvider):
        def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
            return EmbeddingBatchResult(
                model_name="fake-model",
                vectors=((0.1, 0.2, 0.3),),
            )

    service = EmbeddingService(
        provider=BrokenProvider(),
        batcher=batcher,
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="unexpected number of embeddings",
    ):
        service.embed(["A", "B"])


def test_embedding_service_rejects_inconsistent_models(
    batcher: TokenBatcher,
) -> None:
    class ChangingModelProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self.call_count = 0

        def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
            self.call_count += 1

            model_name = "model-a" if self.call_count == 1 else "model-b"

            return EmbeddingBatchResult(
                model_name=model_name,
                vectors=tuple((0.1, 0.2, 0.3) for _ in texts),
            )

    service = EmbeddingService(
        provider=ChangingModelProvider(),
        batcher=batcher,
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="inconsistent model names",
    ):
        service.embed(["A", "B", "C"])


def test_embedding_service_rejects_inconsistent_dimensions(batcher: TokenBatcher) -> None:
    class ChangingDimensionsProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self.call_count = 0

        def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
            self.call_count += 1

            vector = (0.1, 0.2, 0.3) if self.call_count == 1 else (0.1, 0.2, 0.3, 0.4)

            return EmbeddingBatchResult(
                model_name="fake-model",
                vectors=tuple(vector for _ in texts),
            )

    service = EmbeddingService(
        provider=ChangingDimensionsProvider(),
        batcher=batcher,
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="inconsistent vector dimensions",
    ):
        service.embed(["A", "B", "C"])


def test_embedding_service_propagates_provider_errors(
    batcher: TokenBatcher,
) -> None:
    class FailingProvider(EmbeddingProvider):
        def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
            raise EmbeddingProviderError("Provider failed")

    service = EmbeddingService(
        provider=FailingProvider(),
        batcher=batcher,
    )

    with pytest.raises(EmbeddingProviderError, match="Provider failed"):
        service.embed(["Hello"])
