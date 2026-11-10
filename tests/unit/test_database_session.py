import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.engine import URL

from app.config import Settings
from app.db.exceptions import DatabaseConfigurationError
from app.db.session import Database


def make_settings(database_url: str | None = "postgresql+asyncpg://user:password@localhost:5432/prod_rag") -> Settings:
    return Settings(_env_file=None, database_url=database_url)


def test_database_requires_database_url() -> None:
    with pytest.raises(DatabaseConfigurationError, match="DATABASE_URL is required"):
        Database(make_settings(None))


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://user:password@localhost:5432/prod_rag",
        "sqlite+aiosqlite:///./prod_rag.db",
        "not-a-url",
        "postgresql+asyncpg://user:password@localhost",
    ],
)
def test_database_rejects_unsupported_or_incomplete_url(database_url: str) -> None:
    with pytest.raises(DatabaseConfigurationError):
        Database(make_settings(database_url))


def test_database_url_is_parsed_as_sqlalchemy_url() -> None:
    url = Database._database_url(make_settings())

    assert isinstance(url, URL)
    assert url.drivername == "postgresql+asyncpg"
    assert url.host == "localhost"
    assert url.database == "prod_rag"


def test_database_configures_engine_and_session_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock()
    session_factory = MagicMock()
    create_engine = MagicMock(return_value=engine)
    create_factory = MagicMock(return_value=session_factory)

    monkeypatch.setattr("app.db.session.create_async_engine", create_engine)
    monkeypatch.setattr("app.db.session.async_sessionmaker", create_factory)

    settings = make_settings()
    database = Database(settings)

    create_engine.assert_called_once()
    args, kwargs = create_engine.call_args
    assert args[0].drivername == "postgresql+asyncpg"
    assert kwargs == {
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_timeout": settings.database_pool_timeout_seconds,
        "pool_recycle": settings.database_pool_recycle_seconds,
        "pool_pre_ping": settings.database_pool_pre_ping,
    }
    create_factory.assert_called_once_with(
        bind=engine,
        class_=pytest.importorskip("sqlalchemy.ext.asyncio").AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
    assert database.engine is engine
    assert database.session_factory is session_factory


def test_database_session_dependency_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    session_factory = MagicMock(return_value=session_context)

    monkeypatch.setattr("app.db.session.create_async_engine", MagicMock())
    monkeypatch.setattr("app.db.session.async_sessionmaker", MagicMock(return_value=session_factory))
    database = Database(make_settings())

    async def consume_dependency() -> list[object]:
        yielded: list[object] = []
        dependency: AsyncIterator = database.get_session()
        async for item in dependency:
            yielded.append(item)
        return yielded

    yielded = asyncio.run(consume_dependency())

    assert yielded == [session]
    session_context.__aexit__.assert_awaited_once()


def test_database_disposes_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    monkeypatch.setattr("app.db.session.create_async_engine", MagicMock(return_value=engine))
    monkeypatch.setattr("app.db.session.async_sessionmaker", MagicMock())

    database = Database(make_settings())
    asyncio.run(database.dispose())

    engine.dispose.assert_awaited_once()
