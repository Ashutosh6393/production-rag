import pytest

from app.vector_store.base import VectorStore


def test_vector_store_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        VectorStore()


def test_vector_store_requires_upsert_document() -> None:
    class IncompleteVectorStore(VectorStore):
        def delete_document(
            self,
            *,
            tenant_id: str,
            document_id: str,
        ) -> None:
            pass

    with pytest.raises(TypeError):
        IncompleteVectorStore()


def test_vector_store_requires_delete_document() -> None:
    class IncompleteVectorStore(VectorStore):
        def upsert_document(
            self,
            embedded_document,
            *,
            tenant_id: str,
            document_id: str,
        ) -> None:
            pass

    with pytest.raises(TypeError):
        IncompleteVectorStore()


def test_concrete_vector_store_can_be_instantiated() -> None:
    class FakeVectorStore(VectorStore):
        def upsert_document(
            self,
            embedded_document,
            *,
            tenant_id: str,
            document_id: str,
        ) -> None:
            pass

        def delete_document(
            self,
            *,
            tenant_id: str,
            document_id: str,
        ) -> None:
            pass

    store = FakeVectorStore()

    assert isinstance(store, VectorStore)
