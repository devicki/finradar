"""
SQLAlchemy model for the `topics` table.

Topics are used to group and classify news items (e.g. "US Equities",
"Crypto", "Fed Policy").  Each NewsItem optionally belongs to one Topic.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finradar.models import Base

if TYPE_CHECKING:
    from finradar.models.news import NewsItem


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    news_items: Mapped[list[NewsItem]] = relationship(
        "NewsItem",
        back_populates="topic",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Topic id={self.id!r} slug={self.slug!r}>"
