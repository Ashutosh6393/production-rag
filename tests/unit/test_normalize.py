
import pytest

from app.ingestion.chunking.normalize import normalize_text


def test_collapses_repeated_spaces() -> None:
    assert normalize_text("Employee   Security   Policy") == (
        "Employee Security Policy"
    )


def test_normalizes_line_endings() -> None:
    assert normalize_text("First line\r\nSecond line\rThird line") == (
        "First line\nSecond line\nThird line"
    )


def test_preserves_paragraph_boundaries() -> None:
    text = "First paragraph.\n\n\nSecond paragraph."

    assert normalize_text(text) == (
        "First paragraph.\n\nSecond paragraph."
    )


def test_removes_control_characters() -> None:
    assert normalize_text("Hello\x00World") == "HelloWorld"


def test_strips_outer_whitespace() -> None:
    assert normalize_text("  Hello world  \n") == "Hello world"


def test_empty_text_returns_empty_string() -> None:
    assert normalize_text("") == ""


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_text(None)  # type: ignore[arg-type]
