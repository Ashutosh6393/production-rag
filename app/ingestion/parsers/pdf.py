from pathlib import Path
from typing import Any

import pymupdf

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.schemas import ExtractedPage, ExtractionResult


class PDFParser(DocumentParser):
    """Extract text and metadata from PDF files."""

    def parse(self, source: Path) -> ExtractionResult:

        if not source.exists():
            raise FileNotFoundError(f"PDF file does not exists: {source.name}")

        if not source.is_file():
            raise ValueError(f"PDF file is not a file: {source.name}")
        
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file extension: {source.suffix}")
        
        pages: list[ExtractedPage] = []
        warnings: list[str] = []

        try: 
            with pymupdf.open(source) as document: 
                if not document.is_pdf:
                    raise ValueError("The supplied file is not a valid PDF.")
                
                if document.needs_pass: 
                    raise ValueError("The supplied file is protected by a password.")


                metadata: dict[str, Any] = {
                    key: value
                    for key, value in (document.metadata or {}).items()
                    if value is not None
                }

                for page_index, page in enumerate(document):
                    extracted = page.get_text("text")
                    text = extracted.strip() if isinstance(extracted, str) else ""

                    if not text:
                        warnings.append(f"Page {page_index + 1} contains no extractable text.")


                    pages.append(ExtractedPage(
                        page_number=page_index + 1,
                        text = text
                    ))
                
                page_count = document.page_count

        except pymupdf.EmptyFileError as exc:
            raise ValueError("The PDF is empty.") from exc
        except pymupdf.FileDataError as exc: 
            raise ValueError("The PDF is currupt or unreadable") from exc
        


        if page_count == 0:
            warnings.append("The PDF contains no pages.")

        if pages and all(not page.text for page in pages):
            warnings.append("No extractable text was found. The document may require OCR.")

        

        return ExtractionResult(
            source_name=source.name,
            source_format="pdf",
            page_count=page_count,
            pages=pages,
            warnings=warnings,
            metadata=metadata
        )
