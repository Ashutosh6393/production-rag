import json
from uuid import UUID, uuid5

from qdrant_client.models import PointStruct

from app.ingestion.embeddings.schemas import EmbeddedChunk

# Keep this namespace constant across application versions.
# Changing it will generate different IDs for existing chunks.
POINT_NAMESPACE = UUID("b8a7c6d5-e4f3-4210-9876-123456789abc")


def generate_point_id(
    *,
    tenant_id: str,
    document_id: str,
    chunk_id: str,
) -> str:
    """Generate a deterministic Qdrant point ID."""

    identifiers = {
        "tenant_id": _validate_identifier(tenant_id, "tenant_id"),
        "document_id": _validate_identifier(document_id, "document_id"),
        "chunk_id": _validate_identifier(chunk_id, "chunk_id"),
    }

    identity = json.dumps(
        identifiers,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return str(uuid5(POINT_NAMESPACE, identity))


def build_point(
    embedded_chunk: EmbeddedChunk,
    *,
    tenant_id: str,
    document_id: str,
    embedding_model: str,
) -> PointStruct:
    """Convert an embedded chunk into a Qdrant point."""

    if not isinstance(embedded_chunk, EmbeddedChunk):
        raise TypeError("embedded_chunk must be an EmbeddedChunk instance")

    tenant_id = _validate_identifier(tenant_id, "tenant_id")
    document_id = _validate_identifier(document_id, "document_id")
    embedding_model = _validate_identifier(embedding_model, "embedding_model")

    chunk = embedded_chunk.chunk

    point_id = generate_point_id(
        tenant_id=tenant_id,
        document_id=document_id,
        chunk_id=chunk.chunk_id,
    )

    payload = _serialize_payload(
        {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source_name": chunk.source_name,
            "page_numbers": list(chunk.page_numbers),
            "metadata": chunk.metadata,
            "embedding_model": embedding_model,
        }
    )


    return PointStruct(
        id=point_id,
        vector=list(embedded_chunk.vector),
        payload=payload,
    )


def _validate_identifier(value: str, field_name: str) -> str:
    """Validate identifiers used for point identity and payload."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    value = value.strip()

    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    return value


def _serialize_payload(payload: dict) -> dict:
    """Validate and normalize a payload to JSON-compatible values."""
    try:
        return json.loads(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Qdrant payload must contain only JSON-compatible values.") from exc
