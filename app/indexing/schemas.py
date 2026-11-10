from pydantic import BaseModel, ConfigDict, Field


class IndexingResult(BaseModel):
    """Successful indexing and verification outcome for one document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    vector_dimensions: int = Field(gt=0)
    indexed_chunk_count: int = Field(gt=0)
    verified_chunk_count: int = Field(gt=0)
