from pydantic import BaseModel, ConfigDict, Field


# page may not contain any text, but it will still be extracted it may contain images or other non-text content
class ExtractedPage(BaseModel):
    """Text extracted from single document page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="Page number in the document.")
    text: str = Field(description="Text extracted from the page.")


class ExtractionResult(BaseModel):
    """Structured output produced by a document parser."""

    model_config = ConfigDict(frozen=True)

    source_name: str = Field(
        description="Source of the document.",
    )
    source_format: str = Field(description="Format of the document.")
    page_count: int = Field(ge=0, description="Number of pages in the document.")
    pages: tuple[ExtractedPage, ...] = Field(
        description="Extracted pages from the document."
    )
    warnings: tuple[str, ...] = Field(
        description="Warnings raised during the extraction process.", default=()
    )

    @property
    def full_text(self) -> str:
        """Combine all pages while retaining page boundaries."""

        return "\n\n".join(page.text for page in self.pages)


# uv run python -c "from app.ingestion.schemas import ExtractedPage, ExtractionResult; p = ExtractedPage(page_number=1, text='Hello'); r = ExtractionResult(source_name='test.pdf', source_format='pdf', page_count=1, pages=(p,)); print(r.full_text)"

# expected output: Hello
