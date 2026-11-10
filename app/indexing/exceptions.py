from app.vector_store.schemas import IndexVerificationResult


class IndexingError(Exception):
    """Base exception for indexing orchestration failures."""


class IndexingVerificationError(IndexingError):
    """Raised when one or more expected chunks are missing from the vector index."""

    def __init__(self, result: IndexVerificationResult) -> None:
        self.result = result
        missing_count = len(result.missing_chunk_ids)
        unexpected_count = len(result.unexpected_chunk_ids)
        super().__init__(
            "Vector index verification failed: "
            f"{missing_count} expected chunks are missing and "
            f"{unexpected_count} unexpected chunks are present."
        )
