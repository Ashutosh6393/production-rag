
from pathlib import Path

from app.ingestion.parsers.pdf import PDFParser


def main() -> None:
    pdf_path = Path("sample.pdf")

    result = PDFParser().parse(pdf_path)

    print(f"Source: {result.source_name}")
    print(f"Pages: {result.page_count}")
    print(f"Metadata: {result.metadata}")
    print(f"Warnings: {result.warnings}")

    for page in result.pages:
        print(f"\n--- Page {page.page_number} ---")
        print(page.text[:500])


if __name__ == "__main__":
    main()
