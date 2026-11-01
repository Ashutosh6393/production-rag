import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingBatchResult(BaseModel):
    """Represents embeddings generated for a batch of texts."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(min_length=1)
    vectors: tuple[tuple[float, ...], ...] = Field(min_length=1)

    @field_validator("vectors")
    @classmethod
    def validate_vectors(cls, vectors: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:

        if not vectors:
            raise ValueError("Embedding batch cannot be empty")

        expected_dimensions = len(vectors[0])

        if expected_dimensions == 0:
            raise ValueError("Embedding vector cannot be empty")

        for index, vector in enumerate(vectors):
            if len(vector) != expected_dimensions:
                raise ValueError(
                    f"Inconsistent vector dimensions at index {index}. Expected {expected_dimensions}, got {len(vector)}."
                )

            if not all(math.isfinite(value) for value in vector):
                raise ValueError(f"Vector at index {index} contains non-finite values.")

        return vectors

    @property
    def dimensions(self) -> int:
        """Returns the number of dimensions of the embedding vectors."""
        return len(self.vectors[0])
