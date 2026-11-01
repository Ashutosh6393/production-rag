import pytest
from pydantic import ValidationError

from app.ingestion.embeddings.batching import BatchingConfig, TokenBatcher, TokenCounter


def test_default_batching_config() -> None:
    config = BatchingConfig()

    assert config.max_inputs_per_batch == 128
    assert config.max_tokens_per_batch == 100_000
    assert config.max_tokens_per_input == 8_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_inputs_per_batch", 0),
        ("max_inputs_per_batch", 2049),
        ("max_tokens_per_batch", 0),
        ("max_tokens_per_batch", 300001),
        ("max_tokens_per_input", 0),
        ("max_tokens_per_input", 8192),
    ],
)
def test_batching_config_rejects_invalid_limits(
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError):
        BatchingConfig(**{field: value})


def test_batching_config_rejects_input_limit_exceeding_batch_limit() -> None:
    with pytest.raises(ValueError):
        BatchingConfig(
            max_tokens_per_batch=100,
            max_tokens_per_input=200,
        ).validate_configuration()


def test_token_counter_counts_tokens() -> None:
    counter = TokenCounter()

    assert counter.count("Hello world") > 0


def test_token_counter_handles_empty_text() -> None:
    counter = TokenCounter()

    assert counter.count("") == 0


def test_token_counter_rejects_non_string_input() -> None:
    counter = TokenCounter()

    with pytest.raises(TypeError):
        counter.count(123)  # type: ignore[arg-type]


def test_token_counter_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported tokenizer"):
        TokenCounter("unknown_encoding")


def test_split_empty_input() -> None:
    batcher = TokenBatcher()

    assert batcher.split([]) == []


def test_split_preserves_all_inputs() -> None:
    batcher = TokenBatcher()

    texts = ["First chunk", "Second chunk", "Third chunk"]

    batches = batcher.split(texts)

    flattened = [text for batch in batches for text in batch]

    assert flattened == texts


def test_split_respects_max_inputs() -> None:
    config = BatchingConfig(
        max_inputs_per_batch=2,
        max_tokens_per_batch=1000,
        max_tokens_per_input=100,
    )

    batcher = TokenBatcher(config=config)

    batches = batcher.split(["First", "Second", "Third"])

    assert [len(batch) for batch in batches] == [2, 1]


def test_split_respects_max_tokens() -> None:
    config = BatchingConfig(
        max_inputs_per_batch=10,
        max_tokens_per_batch=3,
        max_tokens_per_input=3,
    )

    batcher = TokenBatcher(config=config)

    batches = batcher.split(["one two", "three four", "five six"])

    assert batches == [["one two"], ["three four"], ["five six"]]


def test_split_preserves_order_across_batches() -> None:
    config = BatchingConfig(
        max_inputs_per_batch=2,
        max_tokens_per_batch=1000,
        max_tokens_per_input=100,
    )

    batcher = TokenBatcher(config=config)

    texts = ["A", "B", "C", "D", "E"]

    batches = batcher.split(texts)

    assert batches == [
        ["A", "B"],
        ["C", "D"],
        ["E"],
    ]


def test_split_rejects_oversized_input() -> None:
    config = BatchingConfig(
        max_inputs_per_batch=10,
        max_tokens_per_batch=100,
        max_tokens_per_input=2,
    )

    batcher = TokenBatcher(config=config)

    with pytest.raises(ValueError, match="exceeds the maximum"):
        batcher.split(["This input is too long"])


def test_split_rejects_empty_text() -> None:
    batcher = TokenBatcher()

    with pytest.raises(ValueError, match="cannot be empty"):
        batcher.split(["Valid text", "   "])


def test_split_rejects_string_instead_of_sequence() -> None:
    batcher = TokenBatcher()

    with pytest.raises(TypeError):
        batcher.split("Hello world")


def test_split_rejects_non_string_input() -> None:
    batcher = TokenBatcher()

    with pytest.raises(TypeError):
        batcher.split(["Hello", 123])  # type: ignore[list-item]
