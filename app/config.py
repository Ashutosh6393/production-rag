from functools import lru_cache

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from evnironment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    qdrant_url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    qdrant_api_key: SecretStr | None = None

    qdrant_collection_name: str = Field(default="prod_rag_chunks", min_length=1, max_length=128)

    qdrant_timeout_seconds: float = Field(default=10.0, gt=0, le=120)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the settings for the application."""
    return Settings()
