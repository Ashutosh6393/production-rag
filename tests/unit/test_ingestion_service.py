from pathlib import Path

import pymupdf

from app.ingestion.service import IngestionService


def test_process_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Employee security policies must be followed.",
        )
        document.save(pdf_path)

    service = IngestionService()

    extraction, chunks = service.process(pdf_path)

    assert extraction.page_count == 1
    assert len(chunks) == 1
    assert "Employee security policies" in chunks[0].text
    assert chunks[0].page_numbers == (1,)