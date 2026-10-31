from app.ingestion.exceptions import UnsupportedDocumentError
from pathlib import Path
import codecs
import re
import zipfile

from app.ingestion.exceptions import DocumentNotFoundError, InvalidDocumentError
from app.ingestion.formats import DocumentFormat


class DocumentFormatDetector:
    """Detects and vallidates supported document formats."""

    _SIGNATURE_SIZE = 8

    _HTML_TAG_PATTERN = re.compile(
        r"<\s*(?:!doctype\s+html|html|head|body|title|main|article|section|h[1-6]|p|div|span|ul|ol|li|table|a|img|script|style)\b",
        re.IGNORECASE,
    )

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

        elif expected_format == DocumentFormat.DOCX:
            if not self._is_valid_docx(source):
                raise InvalidDocumentError("File extension indicates DOCX, but content is not a valid DOCX package.")

        elif expected_format == DocumentFormat.TXT:
            if not self._is_valid_txt(source):
                raise InvalidDocumentError("File extension indicates TXT, but content is not valid UTF-8 text.")

        elif expected_format == DocumentFormat.HTML:
            if not self._is_valid_html(source):
                raise InvalidDocumentError("File extension indicates HTML, but recognizable HTML markup was not found.")

        else:
            raise NotImplementedError(f"Content detection is not implemented for {expected_format.value}.")
        return expected_format

    def _is_valid_docx(self, source: Path) -> bool:
        """Check whether a ZIP archive has the required DOCX structure."""

        try:
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())

                required_entries = {
                    "[Content_Types].xml",
                    "word/document.xml",
                }

                return required_entries.issubset(names)

        except (zipfile.BadZipFile, OSError):
            return False

    def _is_valid_txt(self, source: Path) -> bool:
        """Check whether a file contains valid UTF-8 text."""

        decoder = codecs.getincrementaldecoder("utf-8")()

        try:
            with source.open("rb") as file:
                while chunk := file.read(8192):
                    if b"\x00" in chunk:
                        return False

                    decoder.decode(chunk)

                decoder.decode(b"", final=True)

            return True

        except (UnicodeDecodeError, OSError):
            return False

    def _is_valid_html(self, source: Path) -> bool:
        """Check whether a file contains recognizable HTML markup."""

        try:
            with source.open("rb") as file:
                content = file.read(1024 * 1024)

            text = content.decode("utf-8")

            if "\x00" in text:
                return False

            return bool(self._HTML_TAG_PATTERN.search(text))

        except (UnicodeDecodeError, OSError):
            return False
