from app.ingestion.exceptions import IngestionError


class EmbeddingProviderError(IngestionError):
    """Base exception for embedding provider failures."""

    retryable = False


class EmbeddingAuthenticationError(EmbeddingProviderError):
    """Provider authentication failed."""


class EmbeddingRequestError(EmbeddingProviderError):
    """Provider rejected an invalid or unsupported request."""


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Provider rate limit was exceeded."""

    retryable = True


class EmbeddingUnavailableError(EmbeddingProviderError):
    """Provider is temporarily unavailable."""

    retryable = True
