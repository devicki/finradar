"""
SQLAlchemy model for the `news_items` table.

Central table of the FinRadar platform.  Each row represents a unique
(topic_id, url) pair.  Duplicate URLs for the same topic increment
`hit_count` rather than creating new rows.

Column groups:
- Identity / source metadata
- Timestamps & deduplication
- AI analysis output (local sentiment: FinBERT/KR-FinBert-SC, cloud LLM summary/translation)
- Structured tags (tickers, sectors)
- Search artifacts (pgvector embedding, PostgreSQL FTS tsvector)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finradar.models import Base

if TYPE_CHECKING:
    from finradar.models.feedback import UserFeedback
    from finradar.models.topic import Topic


class NewsItem(Base):
    __tablename__ = "news_items"

    # ------------------------------------------------------------------
    # Primary key & foreign keys
    # ------------------------------------------------------------------
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Core content
    # ------------------------------------------------------------------
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str] = mapped_column(String, nullable=False)

    # ------------------------------------------------------------------
    # Timestamps & deduplication
    # ------------------------------------------------------------------
    # Source-declared publication time (RSS pubDate, tweet created_at,
    # trafilatura meta.date, …). Nullable — not every collector path can
    # recover it. User-facing feeds filter/sort on this column.
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the article was published at the source (may be NULL).",
    )
    # Time we first INSERTed this row. Distinct from published_at — this is
    # about OUR system's lifecycle, not the content's. Aligned with
    # created_at going forward.
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Time the same URL was most recently re-observed by a collector
    # (hit_count tick). Stays equal to first_seen_at when the article is
    # never re-collected.
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    hit_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # ------------------------------------------------------------------
    # AI analysis output
    # ------------------------------------------------------------------
    sentiment: Mapped[Optional[float]] = mapped_column(
        Float,
        comment=(
            "Continuous score in [-1.0, +1.0] from the local sentiment model "
            "(FinBERT for EN articles, KR-FinBert-SC for KO; selected by "
            "finradar.processors.sentiment.get_sentiment_analyzer)."
        ),
    )
    sentiment_label: Mapped[Optional[str]] = mapped_column(
        String(10),
        comment="positive | negative | neutral (from the local sentiment model)",
    )
    # Second sentiment signal from the cloud LLM, produced during enrichment.
    # Kept separate from the local-model columns above — see
    # migrate_006_llm_sentiment.sql.  Used by the alert dispatcher to require
    # local + LLM agreement on the strong_sentiment trigger; prevents
    # keyword-only misfires by FinBERT (EN) or KR-FinBert-SC (KO).  NULL for
    # pre-enrich / pre-migration rows, in which case evaluate_trigger falls
    # back to local-sentiment-only.
    llm_sentiment: Mapped[Optional[float]] = mapped_column(
        Float,
        comment="Continuous score in [-1.0, +1.0] emitted by the enrich LLM",
    )
    llm_sentiment_label: Mapped[Optional[str]] = mapped_column(
        String(10),
        comment="positive | negative | neutral (from the enrich LLM)",
    )
    source_type: Mapped[Optional[str]] = mapped_column(
        String(20),
        comment="rss | api | x_feed | url_report",
    )
    language: Mapped[Optional[str]] = mapped_column(
        String(5),
        comment="ISO 639-1 language code, e.g. en | ko | ja",
    )

    # ------------------------------------------------------------------
    # Translation & summarisation (cloud LLM)
    # ------------------------------------------------------------------
    translated_title: Mapped[Optional[str]] = mapped_column(Text)
    translated_summary: Mapped[Optional[str]] = mapped_column(Text)
    ai_summary: Mapped[Optional[str]] = mapped_column(Text)

    # ------------------------------------------------------------------
    # LLM enrichment bookkeeping (prevents infinite retry / token waste)
    # ------------------------------------------------------------------
    llm_enrich_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
        comment="Number of LLM enrichment calls made for this row; caps retries",
    )
    llm_last_attempt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        comment="Set when queued/executed; debounces reconcile re-queue",
    )

    # ------------------------------------------------------------------
    # Structured tags
    # ------------------------------------------------------------------
    tickers: Mapped[Optional[list[str]]] = mapped_column(
        ARRAY(String),
        comment="e.g. {AAPL, TSLA}",
    )
    sectors: Mapped[Optional[list[str]]] = mapped_column(
        ARRAY(String),
        comment="e.g. {반도체, AI}",
    )

    # ------------------------------------------------------------------
    # Search artifacts (non-Mapped — no Python-side type annotation needed)
    # ------------------------------------------------------------------
    embedding = mapped_column(
        Vector(384),
        comment="all-MiniLM-L6-v2 sentence embedding, 384 dimensions",
    )
    search_vector = mapped_column(
        TSVECTOR,
        comment="PostgreSQL full-text search tsvector",
    )

    # ------------------------------------------------------------------
    # Clustering (maintained by finradar.tasks.collection_tasks.cluster_news)
    # ------------------------------------------------------------------
    cluster_rep_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("news_items.id", ondelete="SET NULL"),
        nullable=True,
        comment=(
            "Representative article of this row's cluster (NULL for singletons). "
            "When cluster_rep_id == id, this row IS the cluster representative."
        ),
    )
    cluster_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment="Cached count of articles in the cluster (1 for singletons).",
    )
    similarity_to_rep: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Cosine similarity to the cluster representative (1.0 for the rep itself).",
    )

    # ------------------------------------------------------------------
    # Per-source metadata (feedparser entry, X tweet meta, API payload, ...)
    # ------------------------------------------------------------------
    raw_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "Source-specific metadata JSON. Shape varies by source_type. "
            "Examples: X tweets carry raw_data.x = {tweet_id, username, "
            "breaking, linked_url, public_metrics}; RSS carries feedparser entry."
        ),
    )

    # ------------------------------------------------------------------
    # Audit timestamps
    # ------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    topic: Mapped[Optional[Topic]] = relationship(
        "Topic",
        back_populates="news_items",
        lazy="select",
    )
    feedback: Mapped[list[UserFeedback]] = relationship(
        "UserFeedback",
        back_populates="news_item",
        lazy="select",
        cascade="all, delete-orphan",
    )

    # ------------------------------------------------------------------
    # Table-level constraints and indexes
    # ------------------------------------------------------------------
    __table_args__ = (
        UniqueConstraint("topic_id", "url", name="uq_topic_url"),
        # pgvector IVFFlat cosine index for ANN search
        Index(
            "idx_news_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index for PostgreSQL full-text search
        Index(
            "idx_news_search",
            "search_vector",
            postgresql_using="gin",
        ),
        # GIN index for array containment queries on tickers
        Index(
            "idx_news_tickers",
            "tickers",
            postgresql_using="gin",
        ),
        # B-tree descending index for recency-sorted queries
        Index(
            "idx_news_last_seen",
            "last_seen_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<NewsItem id={self.id!r} source_type={self.source_type!r} "
            f"sentiment_label={self.sentiment_label!r} title={self.title[:60]!r}>"
        )
