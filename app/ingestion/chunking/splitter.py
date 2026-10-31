import hashlib

from app.ingestion.chunking.normalize import normalize_text
from app.ingestion.chunking.schemas import DocumentChunk
from app.ingestion.schemas import ExtractionResult


def _make_chunk_id(
    source_name: str,
    chunk_index: int,
    text: str,
) -> str:
    """Generate a deterministic ID for a chunk's current position and text."""

    payload = f"{source_name}\0{chunk_index}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into bounded chunks, preferring paragraph boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size -1")

    text = normalize_text(text)

    if not text:
        return []

    chunks: list[str] = []

    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind("\n\n", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)

        # avoid beginning the next chunk with whitespace.
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


def split_document(
    result: ExtractionResult,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Split extracted pages into chunks with source page metadata."""
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in result.pages:
        page_chunks = _split_text(
            page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for text in page_chunks:
            chunks.append(
                DocumentChunk(
                    chunk_id=_make_chunk_id(
                        result.source_name,
                        chunk_index,
                        text,
                    ),
                    chunk_index=chunk_index,
                    text=text,
                    source_name=result.source_name,
                    page_numbers=(page.page_number,),
                    metadata={
                        "source_format": result.source_format,
                    },
                )
            )
            chunk_index += 1

    return chunks
