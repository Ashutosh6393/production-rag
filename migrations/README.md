# Database migrations

Migrations are the version-controlled source of truth for the PostgreSQL schema.
Do not use `Base.metadata.create_all()` in application startup or production.

## Configuration

Set `DATABASE_URL` in the environment or `.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/prod_rag
```

The Alembic environment reads the same Pydantic Settings configuration as the
application. The placeholder URL in `alembic.ini` is not used for connections.

## Common commands

```bash
# Check the current database revision.
uv run alembic current

# Show migration history.
uv run alembic history

# Generate a migration after adding/changing ORM models.
uv run alembic revision --autogenerate -m "create document tables"

# Review the generated migration before applying it.
uv run alembic upgrade head

# Roll back one revision.
uv run alembic downgrade -1

# Generate SQL without connecting to PostgreSQL.
uv run alembic upgrade head --sql
```

## Migration workflow

1. Change ORM models.
2. Generate a migration with `revision --autogenerate`.
3. Read the generated Python file and verify every operation.
4. Apply it locally with `upgrade head`.
5. Run database integration tests.
6. Commit the model change and migration together.

Autogenerate is a draft generator, not a schema-diff oracle. It does not reliably
infer table or column renames, data migrations, all constraint changes, or every
database-specific operation. Review every generated migration.

Never edit a migration that has already been applied in a shared environment.
Create a new migration instead. Take a backup before destructive production
migrations and use an expand/contract strategy for changes that must remain
compatible with running application versions.
