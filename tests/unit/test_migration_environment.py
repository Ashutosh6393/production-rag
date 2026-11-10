from pathlib import Path


def test_alembic_configuration_points_to_migrations_directory() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "alembic.ini"
    env_path = project_root / "migrations" / "env.py"
    template_path = project_root / "migrations" / "script.py.mako"

    assert config_path.is_file()
    assert env_path.is_file()
    assert template_path.is_file()

    config_text = config_path.read_text(encoding="utf-8")
    assert "script_location = %(here)s/migrations" in config_text


def test_migration_environment_uses_shared_metadata_and_async_engine() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_text = (project_root / "migrations" / "env.py").read_text(encoding="utf-8")

    assert "target_metadata = Base.metadata" in env_text
    assert "async_engine_from_config" in env_text
    assert "compare_type=True" in env_text
    assert "poolclass=pool.NullPool" in env_text
    assert "import app.db.models" in env_text
