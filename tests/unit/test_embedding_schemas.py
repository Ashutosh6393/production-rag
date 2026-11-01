import pytest
from pydantic import ValidationError

from app.ingestion.embeddings.schemas import EmbeddingBatchResult


def test_create_embedding_batch_result() -> None:
    result = EmbeddingBatchResult(
        model_name="test-model",
        vectors=(
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
        ),
    )

    assert result.model_name == "test-model"
    assert len(result.vectors) == 2
    assert result.dimensions == 3


def test_vectors_are_immutable() -> None:
    result = EmbeddingBatchResult(
        model_name="test-model",
        vectors=((0.1, 0.2, 0.3),),
    )

    with pytest.raises(ValidationError):
        result.vectors = ((0.4, 0.5, 0.6),)


def test_reject_empty_batch() -> None:
    with pytest.raises(ValidationError):
        EmbeddingBatchResult(
            model_name="test-model",
            vectors=(),
        )


def test_reject_empty_vector() -> None:
    with pytest.raises(ValidationError):
        EmbeddingBatchResult(
            model_name="test-model",
            vectors=((),),
        )


def test_reject_inconsistent_dimensions() -> None:
    with pytest.raises(ValidationError, match="Inconsistent vector dimensions"):
        EmbeddingBatchResult(
            model_name="test-model",
            vectors=(
                (0.1, 0.2, 0.3),
                (0.4, 0.5),
            ),
        )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_reject_non_finite_values(invalid_value: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        EmbeddingBatchResult(
            model_name="test-model",
            vectors=((0.1, invalid_value, 0.3),),
        )


def test_reject_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        EmbeddingBatchResult(
            model_name="",
            vectors=((0.1, 0.2),),
        )