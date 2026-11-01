from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from openai.types import CreateEmbeddingResponse
from pydantic import ValidationError

from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.embeddings.exceptions import (
    EmbeddingAuthenticationError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    EmbeddingUnavailableError,
)
from app.ingestion.embeddings.schemas import EmbeddingBatchResult


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI implementation of the embedding provider contract."""

    MODEL_NAME = "text-embedding-3-small"
    MAX_INPUTS_PER_REQUEST = 2048

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = (
            client
            if client is not None
            else OpenAI(
                timeout=30.0,
                max_retries=0,
            )
        )

    def embed(
        self,
        texts: Sequence[str],
    ) -> EmbeddingBatchResult:
        """Generate embeddings for a sequence of texts."""

        self._validate_inputs(texts)

        try:
            response = self._client.embeddings.create(
                model=self.MODEL_NAME,
                input=list(texts),
                encoding_format="float",
            )

        except RateLimitError as exc:
            raise EmbeddingRateLimitError("OpenAI embedding rate limit exceeded.") from exc

        except AuthenticationError as exc:
            raise EmbeddingAuthenticationError("OpenAI authentication failed.") from exc

        except (APITimeoutError, APIConnectionError) as exc:
            raise EmbeddingUnavailableError("Unable to connect to OpenAI embedding service.") from exc

        except BadRequestError as exc:
            raise EmbeddingRequestError("OpenAI rejected the embedding request.") from exc

        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise EmbeddingUnavailableError("OpenAI embedding service encountered a server error.") from exc

            raise EmbeddingRequestError("OpenAI returned an unsuccessful response.") from exc

        except APIError as exc:
            raise EmbeddingProviderError("An unexpected OpenAI API error occurred.") from exc

        return self._build_result(response, expected_count=len(texts))

    def _validate_inputs(self, texts: Sequence[str]) -> None:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a string")

        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")

        if not texts:
            raise ValueError("At least one text is required")

        if len(texts) > self.MAX_INPUTS_PER_REQUEST:
            raise ValueError(f"Maximum {self.MAX_INPUTS_PER_REQUEST} inputs allowed per request")

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Input at index {index} must be a string")

            if not text.strip():
                raise ValueError(f"Input at index {index} cannot be empty")

    def _build_result(
        self,
        response: CreateEmbeddingResponse,
        expected_count: int,
    ) -> EmbeddingBatchResult:
        """Validate and convert the API response into our domain schema."""

        data = response.data

        if len(data) != expected_count:
            raise EmbeddingProviderError("OpenAI returned an unexpected number of embeddings.")

        try:
            model_name = response.model

            if not model_name:
                raise EmbeddingProviderError("OpenAI returned an empty model name.")

            vectors_by_index: dict[int, tuple[float, ...]] = {}

            for item in data:
                index = item.index

                if index in vectors_by_index:
                    raise EmbeddingProviderError(f"Duplicate embedding index received: {index}")

                vectors_by_index[index] = tuple(item.embedding)

            expected_count = len(data)

            if set(vectors_by_index) != set(range(expected_count)):
                raise EmbeddingProviderError("OpenAI returned missing or invalid embedding indices.")

            ordered_vectors = tuple(vectors_by_index[index] for index in range(expected_count))

            return EmbeddingBatchResult(
                model_name=model_name,
                vectors=ordered_vectors,
            )

        except EmbeddingProviderError:
            raise

        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise EmbeddingProviderError("OpenAI returned an invalid embedding response.") from exc
