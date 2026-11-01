from collections.abc import Sequence

import tiktoken
from pydantic import BaseModel, ConfigDict, Field


class BatchingConfig(BaseModel):
    """Configuration for token-aware embedding batches."""

    model_config = ConfigDict(frozen=True)

    max_inputs_per_batch: int = Field(default=128, gt=0, le=2048)
    max_tokens_per_batch: int = Field(default=100_000, gt=0, le=300_000)
    max_tokens_per_input: int = Field(default=8_000, gt=0, le=8191)

    def validate_configuration(self) -> "BatchingConfig":
        if self.max_tokens_per_input > self.max_tokens_per_batch:
            raise ValueError("max_tokens_per_input cannot exceed max_tokens_per_batch")

        return self


class TokenCounter:
    """Counts tokens using a model-compatible tokenizer."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except ValueError as exc:
            raise ValueError(f"Unsupported tokenizer encoding: {encoding_name}") from exc

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return len(self._encoding.encode(text))


class TokenBatcher:
    """Splits text inputs into token-aware batches."""

    def __init__(
        self,
        config: BatchingConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = (config or BatchingConfig()).validate_configuration()
        self.token_counter = token_counter or TokenCounter()

    def split(self, texts: Sequence[str]) -> list[list[str]]:
        """Split texts into batches while preserving their original order."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")

        if not texts:
            return []

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Input at index {index} must be a string")

            if not text.strip():
                raise ValueError(f"Input at index {index} cannot be empty")

            token_count = self.token_counter.count(text)

            if token_count > self.config.max_tokens_per_input:
                raise ValueError(
                    f"Input at index {index} exceeds the maximum token limit of {self.config.max_tokens_per_input}"
                )

            exceeds_input_limit = len(current_batch) >= self.config.max_inputs_per_batch

            exceeds_token_limit = current_tokens + token_count > self.config.max_tokens_per_batch

            if current_batch and (exceeds_input_limit or exceeds_token_limit):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += token_count

        if current_batch:
            batches.append(current_batch)

        return batches
