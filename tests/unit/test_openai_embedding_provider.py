from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.ingestion.embeddings.exceptions import EmbeddingProviderError
from app.ingestion.embeddings.provider import OpenAIEmbeddingProvider


def create_response(
    vectors: list[tuple[int, list[float]]],
) -> SimpleNamespace:
    return SimpleNamespace(
        model="text-embedding-3-small",
        data=[
            SimpleNamespace(index=index, embedding=embedding)
            for index, embedding in vectors
        ],
    )


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def provider(mock_client: Mock) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(client=mock_client)


def test_embed_returns_embedding_batch(
    provider: OpenAIEmbeddingProvider,
    mock_client: Mock,
) -> None:
    mock_client.embeddings.create.return_value = create_response(
        [
            (0, [0.1, 0.2, 0.3]),
            (1, [0.4, 0.5, 0.6]),
        ]
    )

    result = provider.embed(["Hello", "World"])

    assert result.model_name == "text-embedding-3-small"
    assert result.dimensions == 3
    assert result.vectors == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )


def test_embed_uses_correct_model_and_parameters(
    provider: OpenAIEmbeddingProvider,
    mock_client: Mock,
) -> None:
    mock_client.embeddings.create.return_value = create_response(
        [(0, [0.1, 0.2, 0.3])]
    )

    provider.embed(["Hello"])

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["Hello"],
        encoding_format="float",
    )


def test_embed_preserves_input_order(
    provider: OpenAIEmbeddingProvider,
    mock_client: Mock,
) -> None:
    mock_client.embeddings.create.return_value = create_response(
        [
            (2, [0.7, 0.8]),
            (0, [0.1, 0.2]),
            (1, [0.4, 0.5]),
        ]
    )

    result = provider.embed(["First", "Second", "Third"])

    assert result.vectors == (
        (0.1, 0.2),
        (0.4, 0.5),
        (0.7, 0.8),
    )


@pytest.mark.parametrize(
    "texts",
    [
        [],
        (),
    ],
)
def test_embed_rejects_empty_inputs(
    provider: OpenAIEmbeddingProvider,
    texts: list[str] | tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="At least one text"):
        provider.embed(texts)


def test_embed_rejects_single_string(
    provider: OpenAIEmbeddingProvider,
) -> None:
    with pytest.raises(TypeError):
        provider.embed("Hello")


def test_embed_rejects_empty_text(
    provider: OpenAIEmbeddingProvider,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        provider.embed(["Hello", "   "])


def test_embed_rejects_non_string_input(
    provider: OpenAIEmbeddingProvider,
) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        provider.embed(["Hello", 123])  # type: ignore[list-item]


def test_embed_rejects_too_many_inputs(
    provider: OpenAIEmbeddingProvider,
) -> None:
    texts = ["Hello"] * 2049

    with pytest.raises(ValueError, match="Maximum 2048"):
        provider.embed(texts)


def test_embed_rejects_missing_embeddings(
    provider: OpenAIEmbeddingProvider,
    mock_client: Mock,
) -> None:
    mock_client.embeddings.create.return_value = create_response(
        [(0, [0.1, 0.2])]
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="number of embeddings",
    ):
        provider.embed(["Hello", "World"])


def test_embed_rejects_duplicate_indices(
    provider: OpenAIEmbeddingProvider,
    mock_client: Mock,
) -> None:
    mock_client.embeddings.create.return_value = create_response(
        [
            (0, [0.1, 0.2]),
            (0, [0.3, 0.4]),
        ]
    )

    with pytest.raises(EmbeddingProviderError, match="Duplicate"):
        provider.embed(["Hello", "World"])


def test_embed_rejects_invalid_vector(
    provider: OpenAIEmbeddingProvider,
    mock_client: Mock,
) -> None:
    mock_client.embeddings.create.return_value = create_response(
        [(0, [0.1, float("nan")])]
    )

    with pytest.raises(EmbeddingProviderError, match="invalid embedding response"):
        provider.embed(["Hello"])


def test_invalid_input_does_not_call_api(
    provider: OpenAIEmbeddingProvider,
    mock_client: Mock,
) -> None:
    with pytest.raises(ValueError):
        provider.embed([])

    mock_client.embeddings.create.assert_not_called()