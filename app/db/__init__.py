"""Database infrastructure."""

from app.db.exceptions import DatabaseConfigurationError
from app.db.session import Database

__all__ = ["Database", "DatabaseConfigurationError"]
