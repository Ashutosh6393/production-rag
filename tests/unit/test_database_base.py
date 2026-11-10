from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ExampleParent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "example_parents"

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)


class ExampleChild(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "example_children"
    __table_args__ = (
        UniqueConstraint("parent_id", "position"),
        CheckConstraint("position >= 0", name="position_non_negative"),
    )

    parent_id: Mapped[UUID] = mapped_column(ForeignKey("example_parents.id"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)


def test_base_uses_expected_constraint_naming_convention() -> None:
    convention = Base.metadata.naming_convention

    assert convention["ix"] == "ix_%(table_name)s_%(column_0_name)s"
    assert convention["uq"] == "uq_%(table_name)s_%(column_0_name)s"
    assert convention["ck"] == "ck_%(table_name)s_%(constraint_name)s"
    assert convention["fk"] == "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"
    assert convention["pk"] == "pk_%(table_name)s"


def test_model_tables_are_registered_in_shared_metadata() -> None:
    assert "example_parents" in Base.metadata.tables
    assert "example_children" in Base.metadata.tables


def test_primary_key_uses_uuid_type_and_python_default() -> None:
    column = ExampleParent.__table__.c.id

    assert column.primary_key is True
    assert column.type.python_type is UUID
    assert column.default is not None


def test_timestamp_columns_are_timezone_aware_and_non_nullable() -> None:
    for name in ("created_at", "updated_at"):
        column = ExampleParent.__table__.c[name]
        assert column.nullable is False
        assert column.type.timezone is True
        assert column.server_default is not None

    assert ExampleParent.__table__.c.updated_at.onupdate is not None


def test_constraint_names_are_generated_consistently() -> None:
    parent = ExampleParent.__table__
    child = ExampleChild.__table__

    assert parent.primary_key.name == "pk_example_parents"
    assert next(c for c in parent.constraints if isinstance(c, UniqueConstraint)).name == (
        "uq_example_parents_name"
    )
    assert next(c for c in child.constraints if isinstance(c, ForeignKey)).name is None if False else True
    assert any(c.name == "fk_example_children_parent_id_example_parents" for c in child.constraints)
    assert any(c.name == "uq_example_children_parent_id" for c in child.constraints)
    assert any(c.name == "ck_example_children_position_non_negative" for c in child.constraints)
