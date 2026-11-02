import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ingestion.chunking.schemas import DocumentChunk


class EmbeddedChunk(BaseModel):
    """A document chunk associated with its embedding vector."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    vector: tuple[float, ...] = Field(min_length=1)

    @field_validator("vector")
    @classmethod
    def validate_vector(
        cls,
        vector: tuple[float, ...],
    ) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding vector contains non-finite values")

        return vector


class EmbeddedDocument(BaseModel):
    """All embedded chunks belonging to one embedding operation."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(min_length=1)
    chunks: tuple[EmbeddedChunk, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chunks(self) -> "EmbeddedDocument":
        dimensions = {len(item.vector) for item in self.chunks}

        if len(dimensions) != 1:
            raise ValueError("All embedded chunks must have identical vector dimensions")

        chunk_ids = [item.chunk.chunk_id for item in self.chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk IDs are not allowed")

        return self

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return len(self.chunks[0].vector)


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
