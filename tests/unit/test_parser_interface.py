from pathlib import Path

import pytest

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.schemas import ExtractionResult


def test_document_parser_cannot_be_instantiate():
    with pytest.raises(TypeError):
        DocumentParser()


def test_concrete_parser_must_implement_parse_method():
    class TestParser(DocumentParser):
        pass

    with pytest.raises(TypeError):
        TestParser()


def test_concrete_parser_can_implement_parse():
    class FakeParser(DocumentParser):
        def parse(self, source: Path):
            return ExtractionResult(
                source_name="test.pdf",
                source_format="pdf",
                page_count=1,
                pages=(),
                warnings=(),
            )
    
    parser = FakeParser()

    result = parser.parse(Path("test.pdf"))


    assert result.source_name == "test.pdf"
    assert isinstance(result, ExtractionResult)
