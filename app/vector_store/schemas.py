from pydantic import BaseModel, ConfigDict, Field


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
