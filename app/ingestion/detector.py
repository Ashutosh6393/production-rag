from app.ingestion.exceptions import UnsupportedDocumentError
from pathlib import Path

from app.ingestion.exceptions import DocumentNotFoundError, InvalidDocumentError
from app.ingestion.formats import DocumentFormat


class DocumentFormatDetector:
    """Detects and vallidates supported document formats."""

    _SIGNATURE_SIZE = 8

    def detect(self, source: Path) -> DocumentFormat:
        source = Path(source)

        if not source.exists():
            raise DocumentNotFoundError(f"Document does not exist: {source.name}")

        if not source.is_file():
            raise InvalidDocumentError(f"Document is not a file: {source.name}")

        extension = source.suffix.lower()

        supported_extensions = {
            ".pdf": DocumentFormat.PDF,
            ".docx": DocumentFormat.DOCX,
            ".txt": DocumentFormat.TXT,
            ".html": DocumentFormat.HTML,
            ".htm": DocumentFormat.HTML,
        }

        expected_format = supported_extensions.get(extension)

        if expected_format is None:
            raise UnsupportedDocumentError(f"Unsupported document format: {extension}")

        if expected_format == DocumentFormat.PDF:
            with source.open("rb") as file:
                signature = file.read(self._SIGNATURE_SIZE)

                if not signature.startswith(b"%PDF-"):
                    raise InvalidDocumentError("File exension indicates PDF, but content is not a PDF")

        else:
            raise NotImplementedError(f"Content detection is not implemented for {expected_format.value}.")

        return expected_format
