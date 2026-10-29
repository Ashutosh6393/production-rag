# Why separate exceptions? Later, your worker can treat a corrupt PDF as a permanent failure instead of retrying it indefinitely. A transient storage or infrastructure error can have a different retry policy.


class IngestionError(Exception):
    """Base exception for ingestion failures"""


class UnsupportedDocumentError(IngestionError):
    """The document format is not supported."""


class InvalidDocumentError(IngestionError):
    """The document is currupt or cannot be read."""


class EncryptedDocumentError(IngestionError):
    """The document requires a password."""


class DocumentNotFoundError(IngestionError):
    """The source document does not exists."""


class ExtractionError(IngestionError):
    """Document text extraction failed unexpectedly."""
