"""
SQLAlchemy declarative base and model registry for FinRadar.

Import order matters: Base must be defined before any model module is
imported, because each model module imports Base from here.

Usage:
    from finradar.models import Base, NewsItem, Topic, UserFeedback
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Import all models so that:
#   1. They are registered on Base.metadata (required for create_all / Alembic).
#   2. Relationship back-references resolve correctly.
from finradar.models.topic import Topic  # noqa: E402
from finradar.models.news import NewsItem  # noqa: E402
from finradar.models.feedback import UserFeedback  # noqa: E402

__all__ = [
    "Base",
    "NewsItem",
    "Topic",
    "UserFeedback",
]
