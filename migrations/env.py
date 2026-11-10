"""Alembic migration environment for the async PostgreSQL application."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import app.db.models  # noqa: F401  # Register ORM models before reading metadata.
from app.config import get_settings
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Read and validate the database URL without exposing it in logs."""
    settings = get_settings()
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required to run database migrations.")

    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError

    raw_url = settings.database_url.get_secret_value()
    try:
        parsed_url = make_url(raw_url)
    except ArgumentError as exc:
        raise RuntimeError("DATABASE_URL is not a valid SQLAlchemy URL.") from exc

    if parsed_url.drivername != "postgresql+asyncpg":
        raise RuntimeError("DATABASE_URL must use the 'postgresql+asyncpg' driver.")
    if not parsed_url.host or not parsed_url.database:
        raise RuntimeError("DATABASE_URL must include a PostgreSQL host and database name.")

    return raw_url


def run_migrations_offline() -> None:
    """Run migrations without creating an Engine."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create a short-lived async engine for the migration process."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
