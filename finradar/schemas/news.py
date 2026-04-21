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
    published_at: datetime | None = Field(
        default=None,
        description="Source-declared publication time (may be NULL for some ingest paths).",
    )
    first_seen_at: datetime
    last_seen_at: datetime
    hit_count: int
    created_at: datetime

    # --- Clustering (Phase 2 news grouping) ---------------------------------
    cluster_rep_id: int | None = Field(
        default=None,
        description="ID of the representative article of this row's cluster. NULL for singletons.",
    )
    cluster_size: int = Field(
        default=1,
        description="Number of articles in the same cluster (1 for singletons).",
    )
    similarity_to_rep: float | None = Field(
        default=None,
        description="Cosine similarity to the cluster representative (1.0 for the rep itself).",
    )

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


class ScoreBreakdown(BaseModel):
    """Per-signal scores that combine into the final search ranking score.

    All sub-scores are normalised to roughly [0, 1]:

    * ``fts``       — PostgreSQL ts_rank_cd, divided by the per-query max.
    * ``cosine``    — pgvector cosine similarity (1 − cosine_distance).
    * ``recency``   — exponential decay on ``last_seen_at`` (7-day half-life).
    * ``final``     — weighted sum: ``w_bm25·fts + w_cos·cosine + w_rec·recency``.
    """

    fts: float = Field(..., description="FTS rank normalised to [0, 1] within the result set")
    cosine: float = Field(..., description="pgvector cosine similarity (1 − distance)")
    recency: float = Field(..., description="Exponential-decay recency score")
    final: float = Field(..., description="Weighted combined score used for ranking")


class NewsItemSearchResponse(NewsItemResponse):
    """NewsItemResponse + optional hybrid-search scoring details."""

    score: float | None = Field(
        default=None,
        description="Final weighted hybrid score (equivalent to score_breakdown.final)",
    )
    score_breakdown: ScoreBreakdown | None = Field(
        default=None,
        description="Per-signal scores (only present when include_scores=true)",
    )


class QueryExpansionInfo(BaseModel):
    """Diagnostic info about synonym expansion applied to the FTS query.

    Present when the original query contained tokens matching the
    :py:data:`finradar.search.query_expansion.SYNONYMS` dictionary. The
    dashboard uses ``expanded_tokens`` to show users which synonyms were
    added and ``tsquery_expr`` for debugging FTS matching behaviour.
    """

    original: str = Field(..., description="Original user query")
    tsquery_expr: str = Field(..., description="Generated to_tsquery expression")
    expanded_tokens: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of input token → added synonyms (excluding the token itself)",
    )


class NewsSearchListResponse(BaseModel):
    """Paginated hybrid-search response."""

    items: list[NewsItemSearchResponse]
    total: int
    page: int
    page_size: int
    query_expansion: QueryExpansionInfo | None = Field(
        default=None,
        description="Synonym-expansion info (None if no expansion applied)",
    )


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
    include_scores: bool = Field(
        default=False,
        description="Include per-signal score breakdown in the response items",
    )
    dedup: bool = Field(
        default=False,
        description=(
            "When true, suppress duplicate articles within a cluster — only "
            "cluster representatives and singletons are returned. Default is "
            "false for search (users usually want to see all matches); the feed "
            "endpoint defaults to true."
        ),
    )
    # --- Hybrid ranking tunables (override settings defaults per-request) ---
    weight_bm25: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override rank_weight_bm25 for this request",
    )
    weight_cosine: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override rank_weight_cosine for this request",
    )
    weight_recency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override rank_weight_recency for this request",
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
    # Coverage metrics — help clients judge how much of the data is enriched
    # before drawing conclusions from the Top Tickers / Sectors charts.
    articles_with_tickers: int = Field(
        default=0,
        description="Articles in the window that have at least one ticker extracted",
    )
    articles_with_sectors: int = Field(
        default=0,
        description="Articles in the window that have at least one sector extracted",
    )
    articles_llm_enriched: int = Field(
        default=0,
        description=(
            "Articles in the window whose LLM enrichment (ai_summary) has completed. "
            "Tickers/sectors are populated mostly by the LLM step, so a low value here "
            "explains sparse Top Tickers / Sectors charts."
        ),
    )
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
