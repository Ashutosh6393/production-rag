class VectorStoreError(Exception):
    """Base exception for vector storage failures."""

    retryable = False


class VectorStoreConnectionError(VectorStoreError):
    """The vector database cannot be reached."""

    retryable = True


class VectorStoreOperationError(VectorStoreError):
    """A vector database operation failed."""


class VectorStoreConfigurationError(VectorStoreError):
    """The vector database configuration is invalid."""
