import pytest

from app.vector_store.base import VectorStore


def test_vector_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VectorStore()


def test_concrete_implementation_must_implement_all_methods():
    class IncompleteVectorStore(VectorStore):
        def upsert_document(self, embedded_document, *, tenant_id, document_id):
            pass

        def delete_document(self, *, tenant_id, document_id):
            pass

    with pytest.raises(TypeError):
        IncompleteVectorStore()
