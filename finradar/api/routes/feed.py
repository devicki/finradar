"""
finradar.api.routes.feed
~~~~~~~~~~~~~~~~~~~~~~~~

Personalised news feed and aggregated daily summary endpoints.

Phase 1 implementation: the "personalized" feed is simply the latest
news sorted by ``last_seen_at`` DESC.  Phase 3 will add feedback-based
``personal_boost`` re-ranking.

Routes
------
GET /          — latest news feed (paginated, filterable)
GET /summary   — aggregated statistics for a rolling time window
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from finradar.api.deps import CommonFilters, DbSession, PaginationParams, _apply_common_filters
from finradar.models import NewsItem
from finradar.schemas import (
    FeedSummaryResponse,
    NewsItemResponse,
    NewsListResponse,
    SentimentDistribution,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=NewsListResponse,
    summary="Personalised news feed",
    description=(
        "Return the latest news items sorted by `last_seen_at` descending. "
        "Phase 1: ordering is purely chronological.  "
        "Phase 3 will incorporate personal_boost from user feedback."
    ),
)
async def get_feed(
    db: DbSession,
    pagination: Annotated[PaginationParams, Depends(PaginationParams)],
    filters: Annotated[CommonFilters, Depends(CommonFilters)],
) -> NewsListResponse:
    # Count
    count_stmt = select(func.count()).select_from(NewsItem)
    count_stmt = _apply_common_filters(count_stmt, filters)
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Data — most recent first
    data_stmt = (
        select(NewsItem)
        .order_by(NewsItem.last_seen_at.desc())
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    data_stmt = _apply_common_filters(data_stmt, filters)
    rows = (await db.execute(data_stmt)).scalars().all()

    return NewsListResponse(
        items=[NewsItemResponse.model_validate(row) for row in rows],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    response_model=FeedSummaryResponse,
    summary="Daily feed summary",
    description=(
        "Return aggregated statistics for news items collected within the "
        "last `hours` hours (default: 24).  Includes total count, "
        "sentiment distribution, and top tickers / sectors ranked by "
        "occurrence frequency."
    ),
)
async def get_feed_summary(
    db: DbSession,
    hours: int = Query(
        default=24,
        ge=1,
        le=720,
        description="Rolling time window in hours (1–720, default 24)",
    ),
    top_n: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Number of top tickers / sectors to return",
    ),
) -> FeedSummaryResponse:
    now_utc = datetime.now(tz=timezone.utc)
    window_start = now_utc - timedelta(hours=hours)

    # ---- Fetch all rows in the window (only the columns we need) -----------
    # We intentionally avoid pulling the large text / embedding columns here.
    stmt = (
        select(
            NewsItem.sentiment_label,
            NewsItem.tickers,
            NewsItem.sectors,
        )
        .where(NewsItem.first_seen_at >= window_start)
    )
    rows = (await db.execute(stmt)).all()

    # ---- Aggregate in Python (avoids complex unnest SQL for Phase 1) -------
    total_count = len(rows)
    sentiment_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()

    for row in rows:
        label = row.sentiment_label
        if label in ("positive", "negative", "neutral"):
            sentiment_counts[label] += 1

        if row.tickers:
            for ticker in row.tickers:
                if ticker:
                    ticker_counts[ticker.upper()] += 1

        if row.sectors:
            for sector in row.sectors:
                if sector:
                    sector_counts[sector] += 1

    # ---- Build response ----------------------------------------------------
    sentiment_dist = SentimentDistribution(
        positive=sentiment_counts.get("positive", 0),
        negative=sentiment_counts.get("negative", 0),
        neutral=sentiment_counts.get("neutral", 0),
    )

    top_tickers = [
        {"ticker": ticker, "count": count}
        for ticker, count in ticker_counts.most_common(top_n)
    ]
    top_sectors = [
        {"sector": sector, "count": count}
        for sector, count in sector_counts.most_common(top_n)
    ]

    return FeedSummaryResponse(
        total_count=total_count,
        sentiment_distribution=sentiment_dist,
        top_tickers=top_tickers,
        top_sectors=top_sectors,
        window_hours=hours,
        generated_at=now_utc,
    )
