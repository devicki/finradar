"""
SQLAlchemy model for the `user_feedback` table.

Stores explicit user feedback actions on news items (bookmark, like,
dislike, dismiss) to power the personalised ranking engine in Phase 3.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finradar.models import Base


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    news_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("news_items.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    action: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="bookmark | like | dislike | dismiss",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    news_item: Mapped[Optional[object]] = relationship(
        "NewsItem",
        back_populates="feedback",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<UserFeedback id={self.id!r} news_id={self.news_id!r} action={self.action!r}>"
