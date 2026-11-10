from functools import lru_cache

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    qdrant_url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = Field(default="prod_rag_chunks", min_length=1, max_length=128)
    qdrant_timeout_seconds: int = Field(default=10, gt=0, le=120)

    # Keep credentials masked when Settings is logged or inspected.
    database_url: SecretStr | None = None
    database_pool_size: int = Field(default=5, ge=1, le=100)
    database_max_overflow: int = Field(default=10, ge=0, le=100)
    database_pool_timeout_seconds: float = Field(default=30, gt=0, le=300)
    database_pool_recycle_seconds: int = Field(default=1800, gt=0, le=86400)
    database_pool_pre_ping: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the settings for the application."""
    return Settings()
