from pydantic import BaseModel, ConfigDict, Field, model_validator


class VectorSearchResult(BaseModel):
    """A document chunk retrieved from a vector store."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)

    text: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    page_numbers: tuple[int, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)

    score: float


class IndexVerificationResult(BaseModel):
    """Result of checking whether every expected chunk is present in the vector index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_chunk_ids: tuple[str, ...] = Field(min_length=1)
    verified_chunk_ids: tuple[str, ...] = ()
    missing_chunk_ids: tuple[str, ...] = ()
    unexpected_chunk_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_chunk_sets(self) -> "IndexVerificationResult":
        expected = set(self.expected_chunk_ids)
        verified = set(self.verified_chunk_ids)
        missing = set(self.missing_chunk_ids)
        unexpected = set(self.unexpected_chunk_ids)

        if len(expected) != len(self.expected_chunk_ids):
            raise ValueError("expected_chunk_ids must not contain duplicates")

        if len(verified) != len(self.verified_chunk_ids):
            raise ValueError("verified_chunk_ids must not contain duplicates")

        if len(missing) != len(self.missing_chunk_ids):
            raise ValueError("missing_chunk_ids must not contain duplicates")

        if len(unexpected) != len(self.unexpected_chunk_ids):
            raise ValueError("unexpected_chunk_ids must not contain duplicates")

        if unexpected & expected:
            raise ValueError("unexpected_chunk_ids cannot contain expected chunk IDs")

        if verified & missing:
            raise ValueError("A chunk cannot be both verified and missing")

        if verified | missing != expected:
            raise ValueError(
                "verified_chunk_ids and missing_chunk_ids must partition expected_chunk_ids"
            )

        return self

    @property
    def is_verified(self) -> bool:
        """Whether every expected chunk was found in the index."""
        return (
            not self.missing_chunk_ids
            and not self.unexpected_chunk_ids
            and len(self.verified_chunk_ids) == len(self.expected_chunk_ids)
        )
