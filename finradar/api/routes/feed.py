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
from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from finradar.api.deps import CommonFilters, DbSession, PaginationParams, _apply_common_filters
from finradar.api.routes.feedback import _current_user_id
from finradar.models import NewsItem, UserFeedback
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


# Effective recency expression — prefer the source's published_at, fall back
# to our first_seen_at so NULL-published_at rows don't sort to the bottom.
_EFFECTIVE_PUBLISHED = func.coalesce(NewsItem.published_at, NewsItem.first_seen_at)


_SORT_OPTIONS = {
    # Chronological by publication time — the default, matches the
    # "Latest News" mental model.
    "latest": (_EFFECTIVE_PUBLISHED.desc(),),
    # Big-story first — cluster size then publication recency.
    "cluster_size": (
        NewsItem.cluster_size.desc(),
        _EFFECTIVE_PUBLISHED.desc(),
    ),
    # Strong-signal sentiment first — surfaces articles the pipeline scored
    # strongly positive OR strongly negative.  Ties break on recency.
    "sentiment_strength": (
        desc(func.abs(func.coalesce(NewsItem.sentiment, 0.0))),
        _EFFECTIVE_PUBLISHED.desc(),
    ),
}


@router.get(
    "/",
    response_model=NewsListResponse,
    summary="Personalised news feed",
    description=(
        "Paginated news feed with optional sort / filter / dedup. Sort options: "
        "`latest` (default, most recent first), `cluster_size` (biggest stories "
        "first, useful for 'what's trending'), `sentiment_strength` (strongly "
        "positive/negative first, ties broken by recency). "
        "Dedup defaults to TRUE: only cluster representatives and singletons "
        "are returned so the same story isn't repeated."
    ),
)
async def get_feed(
    db: DbSession,
    pagination: Annotated[PaginationParams, Depends(PaginationParams)],
    filters: Annotated[CommonFilters, Depends(CommonFilters)],
    dedup: bool = Query(
        default=True,
        description=(
            "When true (default), suppress duplicate articles within a cluster — "
            "only the cluster representative (or singletons) are returned."
        ),
    ),
    sort: str = Query(
        default="latest",
        description=(
            "latest | cluster_size | sentiment_strength | personalized. "
            "`personalized` re-ranks the recent candidate pool by recency × "
            "(1 + personal_boost) using the current user's like/dislike "
            "history."
        ),
    ),
    hide_dismissed: bool = Query(
        default=True,
        description=(
            "When true (default), hide articles the current user dismissed "
            "via the 🙈 button. Set to false to audit / unhide from the "
            "Bookmarks 🙈 page."
        ),
    ),
) -> NewsListResponse:
    # Dedup predicate: keep singletons (cluster_rep_id IS NULL) and cluster reps
    # (cluster_rep_id = id).  Uses the idx_news_cluster_rep index for fast scans.
    def _apply_dedup(stmt):
        if not dedup:
            return stmt
        return stmt.where(
            or_(
                NewsItem.cluster_rep_id.is_(None),
                NewsItem.cluster_rep_id == NewsItem.id,
            )
        )

    # Dismiss filter: exclude rows the current user explicitly hid via 🙈.
    # Implemented as a correlated NOT EXISTS so the uq_feedback_user_news_action
    # index keeps the subquery cheap.
    def _apply_dismiss(stmt):
        if not hide_dismissed:
            return stmt
        user_id = _current_user_id()
        return stmt.where(
            ~select(UserFeedback.id)
            .where(
                UserFeedback.user_id == user_id,
                UserFeedback.news_id == NewsItem.id,
                UserFeedback.action == "dismiss",
            )
            .exists()
        )

    if sort == "personalized":
        return await _personalized_feed(
            db, filters, _apply_dedup, _apply_dismiss, pagination,
        )

    order_by = _SORT_OPTIONS.get(sort) or _SORT_OPTIONS["latest"]

    # Count
    count_stmt = select(func.count()).select_from(NewsItem)
    count_stmt = _apply_common_filters(count_stmt, filters)
    count_stmt = _apply_dedup(count_stmt)
    count_stmt = _apply_dismiss(count_stmt)
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Data
    data_stmt = (
        select(NewsItem)
        .order_by(*order_by)
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    data_stmt = _apply_common_filters(data_stmt, filters)
    data_stmt = _apply_dedup(data_stmt)
    data_stmt = _apply_dismiss(data_stmt)
    rows = (await db.execute(data_stmt)).scalars().all()

    return NewsListResponse(
        items=[NewsItemResponse.model_validate(row) for row in rows],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


# ---------------------------------------------------------------------------
# Personalised feed — recency × (1 + personal_boost), Python-side re-rank
# ---------------------------------------------------------------------------


async def _personalized_feed(
    db,
    filters,
    apply_dedup,
    apply_dismiss,
    pagination: PaginationParams,
) -> NewsListResponse:
    """Candidate pool from recent articles, Python-rerank by personal boost.

    We keep the candidate pool bounded (``_PERSONALIZED_POOL_SIZE``) so
    re-ranking stays cheap even with thousands of matching rows. Re-rank
    is done in Python because the affinity dict lives in Redis, not the
    DB — copying it into every query would be more expensive than the
    current approach.
    """
    from finradar.api.routes.feedback import _current_user_id  # noqa: PLC0415
    from finradar.personalization import get_affinity, personal_boost  # noqa: PLC0415
    from finradar.tasks.collection_tasks import SyncSessionLocal  # noqa: PLC0415

    pool_stmt = (
        select(NewsItem)
        .order_by(_EFFECTIVE_PUBLISHED.desc())
        .limit(_PERSONALIZED_POOL_SIZE)
    )
    pool_stmt = _apply_common_filters(pool_stmt, filters)
    pool_stmt = apply_dedup(pool_stmt)
    pool_stmt = apply_dismiss(pool_stmt)
    candidates = (await db.execute(pool_stmt)).scalars().all()

    # Affinity calculation uses a sync session (matches the rest of the
    # personalization module, which plugs into Celery tasks too).
    user_id = _current_user_id()
    with SyncSessionLocal() as sync_session:
        affinity = get_affinity(sync_session, user_id=user_id)

    # Recency component: exp(-days_since_published / 7). Published_at may be
    # NULL for some legacy rows; fall back to first_seen_at.
    from datetime import datetime, timezone  # noqa: PLC0415

    now_utc = datetime.now(tz=timezone.utc)

    def _recency(item: NewsItem) -> float:
        ts = item.published_at or item.first_seen_at
        if ts is None:
            return 0.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        days = max(0.0, (now_utc - ts).total_seconds() / 86400.0)
        return pow(2.718281828, -days / 7.0)

    scored = []
    for item in candidates:
        recency = _recency(item)
        boost = personal_boost(
            affinity,
            sectors=item.sectors or [],
            tickers=item.tickers or [],
        )
        score = (1.0 + boost) * recency
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    total = len(scored)
    start = pagination.offset
    page = scored[start : start + pagination.page_size]
    return NewsListResponse(
        items=[NewsItemResponse.model_validate(item) for _, item in page],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


_PERSONALIZED_POOL_SIZE = 300  # candidate cap before Python rerank


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
    # ai_summary is pulled as a boolean check (NULL vs not) via a computed column.
    stmt = (
        select(
            NewsItem.sentiment_label,
            NewsItem.tickers,
            NewsItem.sectors,
            NewsItem.ai_summary,
        )
        # Window on publication time (user-facing "last 24h" = 최근 발행).
        # COALESCE so NULL-published rows still have a chance via first_seen_at.
        .where(_EFFECTIVE_PUBLISHED >= window_start)
    )
    rows = (await db.execute(stmt)).all()

    # ---- Aggregate in Python (avoids complex unnest SQL for Phase 1) -------
    total_count = len(rows)
    sentiment_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()

    # Coverage counters — surface to the client so sparse Top charts are
    # explained ("only 3.9% of articles have tickers extracted") instead of
    # looking like a bug.
    articles_with_tickers = 0
    articles_with_sectors = 0
    articles_llm_enriched = 0

    for row in rows:
        label = row.sentiment_label
        if label in ("positive", "negative", "neutral"):
            sentiment_counts[label] += 1

        row_has_ticker = False
        if row.tickers:
            for ticker in row.tickers:
                if ticker:
                    ticker_counts[ticker.upper()] += 1
                    row_has_ticker = True
        if row_has_ticker:
            articles_with_tickers += 1

        row_has_sector = False
        if row.sectors:
            for sector in row.sectors:
                if sector:
                    sector_counts[sector] += 1
                    row_has_sector = True
        if row_has_sector:
            articles_with_sectors += 1

        if row.ai_summary:
            articles_llm_enriched += 1

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
        articles_with_tickers=articles_with_tickers,
        articles_with_sectors=articles_with_sectors,
        articles_llm_enriched=articles_llm_enriched,
        sentiment_distribution=sentiment_dist,
        top_tickers=top_tickers,
        top_sectors=top_sectors,
        window_hours=hours,
        generated_at=now_utc,
    )
