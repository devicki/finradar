"""
finradar.schemas.news
~~~~~~~~~~~~~~~~~~~~~

Pydantic v2 request / response schemas for the news API.

Mirrors the SQLAlchemy NewsItem / Topic / UserFeedback models but keeps
API contracts decoupled from persistence models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# News item schemas
# ---------------------------------------------------------------------------


class NewsItemResponse(BaseModel):
    """Full news item representation returned by the API."""

    id: int
    title: str
    summary: str | None
    url: str
    source_url: str
    source_type: str | None
    language: str | None
    sentiment: float | None
    sentiment_label: str | None
    translated_title: str | None
    translated_summary: str | None
    ai_summary: str | None
    tickers: list[str] | None
    sectors: list[str] | None
    first_seen_at: datetime
    last_seen_at: datetime
    hit_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class NewsListResponse(BaseModel):
    """Paginated list of news items."""

    items: list[NewsItemResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Search schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for the hybrid search endpoint."""

    query: str = Field(..., min_length=1, description="Free-text search query")
    source_type: str | None = Field(
        default=None,
        description="Filter by source type: rss | api | x_feed | url_report",
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code, e.g. en | ko | ja",
    )
    sentiment_label: Literal["positive", "negative", "neutral"] | None = Field(
        default=None,
        description="Filter by FinBERT sentiment label",
    )
    tickers: list[str] | None = Field(
        default=None,
        description="Filter by one or more ticker symbols (e.g. ['AAPL', 'TSLA'])",
    )
    sectors: list[str] | None = Field(
        default=None,
        description="Filter by one or more sectors (e.g. ['반도체', 'AI'])",
    )
    date_from: datetime | None = Field(
        default=None,
        description="Include only news first seen at or after this timestamp (ISO-8601)",
    )
    date_to: datetime | None = Field(
        default=None,
        description="Include only news first seen at or before this timestamp (ISO-8601)",
    )
    page: int = Field(default=1, ge=1, description="1-based page number")
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of results per page (max 100)",
    )


# ---------------------------------------------------------------------------
# Topic schemas
# ---------------------------------------------------------------------------


class TopicResponse(BaseModel):
    """Topic representation returned by the API."""

    id: int
    name: str
    slug: str
    description: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Feedback schemas
# ---------------------------------------------------------------------------


class FeedbackCreate(BaseModel):
    """Request body for submitting user feedback on a news item."""

    news_id: int = Field(..., description="ID of the news item being rated")
    action: Literal["bookmark", "like", "dislike", "dismiss"] = Field(
        ...,
        description="Feedback action: bookmark | like | dislike | dismiss",
    )


class FeedbackResponse(BaseModel):
    """Feedback record returned after successful creation."""

    id: int
    news_id: int | None
    action: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Feed summary schema
# ---------------------------------------------------------------------------


class SentimentDistribution(BaseModel):
    """Counts of each sentiment label within a time window."""

    positive: int = 0
    negative: int = 0
    neutral: int = 0


class FeedSummaryResponse(BaseModel):
    """Aggregated statistics for the daily / time-ranged feed summary."""

    total_count: int = Field(..., description="Total number of news items in the window")
    sentiment_distribution: SentimentDistribution
    top_tickers: list[dict] = Field(
        ...,
        description="Top ticker symbols with occurrence counts, e.g. [{'ticker': 'AAPL', 'count': 5}]",
    )
    top_sectors: list[dict] = Field(
        ...,
        description="Top sectors with occurrence counts, e.g. [{'sector': 'AI', 'count': 12}]",
    )
    window_hours: int = Field(..., description="Time window size in hours used for this summary")
    generated_at: datetime = Field(..., description="Timestamp when this summary was generated")
