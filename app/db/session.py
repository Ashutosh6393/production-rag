from collections.abc import AsyncIterator

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.db.exceptions import DatabaseConfigurationError


class Database:
    """Owns the SQLAlchemy async engine and session factory.

    A Database instance should be created once per application process and disposed
    during application shutdown. Sessions are short-lived and created per unit of work.
    Transaction boundaries remain explicit: callers must use ``session.begin()`` or
    commit/rollback themselves rather than relying on an implicit commit here.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine = create_async_engine(
            self._database_url(self._settings),
            pool_size=self._settings.database_pool_size,
            max_overflow=self._settings.database_max_overflow,
            pool_timeout=self._settings.database_pool_timeout_seconds,
            pool_recycle=self._settings.database_pool_recycle_seconds,
            pool_pre_ping=self._settings.database_pool_pre_ping,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            autoflush=False,
            expire_on_commit=False,
        )

    @staticmethod
    def _database_url(settings: Settings) -> URL:
        """Return a validated async PostgreSQL URL without exposing credentials."""
        if settings.database_url is None:
            raise DatabaseConfigurationError(
                "DATABASE_URL is required to initialize the database."
            )

        raw_url = settings.database_url.get_secret_value()
        try:
            url = make_url(raw_url)
        except ArgumentError as exc:
            raise DatabaseConfigurationError("DATABASE_URL is not a valid SQLAlchemy URL.") from exc

        if url.drivername != "postgresql+asyncpg":
            raise DatabaseConfigurationError(
                "DATABASE_URL must use the 'postgresql+asyncpg' driver."
            )

        if not url.host or not url.database:
            raise DatabaseConfigurationError(
                "DATABASE_URL must include a PostgreSQL host and database name."
            )

        return url

    @property
    def engine(self) -> AsyncEngine:
        """Expose the engine for health checks and migration integration."""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Expose the configured factory for background units of work."""
        return self._session_factory

    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session and always close it after the request/unit of work.

        This is suitable for FastAPI's ``Depends``. It deliberately does not commit
        automatically, so a failed operation cannot accidentally be committed during
        dependency teardown.
        """
        async with self._session_factory() as session:
            yield session

    async def dispose(self) -> None:
        """Close all pooled database connections during application shutdown."""
        await self._engine.dispose()
