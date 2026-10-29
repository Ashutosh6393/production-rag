from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """A traceable unit of text prepared for embedding."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1, description="Unique identifier for the chunk.")
    chunk_index: int = Field(ge=0, description="Index of the chunk in the document.")
    text: str = Field(min_length=1, description="Text of the chunk.")

    source_name: str = Field(description="Source of the document.")
    page_number: tuple[int, ...] = Field(min_length=1, description="Page numbers where the chunk appears.")

    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


# uv run python -c "from app.ingestion.chunking.schemas import DocumentChunk; print('Chunk schema imported successfully')"
