"""
SQLAlchemy model for the `user_feedback` table.

Stores explicit user feedback actions on news items (bookmark, like,
dislike, dismiss) to power the personalised ranking engine in Phase 3.

Multi-user ready: ``user_id`` defaults to ``"owner"`` for the single-user
Phase 3 deployment. Phase 4 (public service + auth) will start populating
it with real authenticated IDs without any schema change.

A unique ``(user_id, news_id, action)`` constraint enforces idempotence:
repeated clicks on the same feedback button don't stack rows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finradar.models import Base


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="owner",
        server_default="owner",
        comment="Actor of this feedback. Hardcoded 'owner' for Phase 3.",
    )
    news_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("news_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
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

    __table_args__ = (
        Index(
            "uq_feedback_user_news_action",
            "user_id",
            "news_id",
            "action",
            unique=True,
        ),
        Index(
            "idx_feedback_user_action_created",
            "user_id",
            "action",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<UserFeedback id={self.id!r} user={self.user_id!r} "
            f"news_id={self.news_id!r} action={self.action!r}>"
        )
