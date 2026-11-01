import pytest

from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.schemas import EmbeddingBatchResult


def test_embedding_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_concrete_provider_must_implement_embed() -> None:
    class IncompleteProvider(EmbeddingProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteProvider()


def test_concrete_provider_can_implement_embed() -> None:
    class FakeEmbeddingProvider(EmbeddingProvider):
        def embed(
            self,
            texts: list[str],
        ) -> EmbeddingBatchResult:
            return EmbeddingBatchResult(
                model_name="fake-model",
                vectors=tuple(
                    (0.1, 0.2, 0.3)
                    for _ in texts
                ),
            )

    provider = FakeEmbeddingProvider()

    result = provider.embed(["Hello", "World"])

    assert len(result.vectors) == 2
    assert result.dimensions == 3