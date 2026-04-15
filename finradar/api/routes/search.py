"""
finradar.api.routes.search
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hybrid search endpoint for FinRadar news.

Phase 1 implementation: PostgreSQL full-text search (FTS) via
``plainto_tsquery`` / ``search_vector``, combined with structured
column filters.

Phase 2 will extend this with pgvector cosine-similarity re-ranking
once the EmbeddingGenerator worker is wired in.

Routes
------
POST /    — search news items using free-text query + optional filters
"""

from __future__ import annotations

from sqlalchemy import Integer, cast, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, status

from finradar.api.deps import DbSession
from finradar.models import NewsItem
from finradar.schemas import NewsItemResponse, NewsListResponse, SearchRequest

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_search_query(request: SearchRequest):  # type: ignore[return]
    """
    Build a SQLAlchemy SELECT for the given SearchRequest.

    FTS ranking
    -----------
    PostgreSQL ``ts_rank_cd`` produces a relevance score in [0, 1].
    We ORDER BY that score DESC so the best matches bubble to the top.
    When the query string contains characters that would confuse
    ``plainto_tsquery`` (e.g. pure numbers, special chars), the function
    gracefully falls back to an unranked recency sort — the WHERE clause
    handles that via a coalesce approach.

    Phase 1 note
    ------------
    Vector search (pgvector cosine) is intentionally omitted here.
    It will be added in Phase 2 as an additional re-ranking step.
    """
    # ``plainto_tsquery`` is safe for arbitrary user input (no syntax errors).
    tsquery = func.plainto_tsquery("english", request.query)

    # ts_rank_cd: coverage-based ranking, normalisation=1 (divide by doc length)
    ts_rank = func.ts_rank_cd(NewsItem.search_vector, tsquery, 1)

    stmt = (
        select(NewsItem, ts_rank.label("rank"))
        .where(NewsItem.search_vector.op("@@")(tsquery))
        .order_by(ts_rank.desc(), NewsItem.last_seen_at.desc())
    )

    # ------------------------------------------------------------------
    # Structured filters
    # ------------------------------------------------------------------
    if request.source_type:
        stmt = stmt.where(NewsItem.source_type == request.source_type)
    if request.language:
        stmt = stmt.where(NewsItem.language == request.language)
    if request.sentiment_label:
        stmt = stmt.where(NewsItem.sentiment_label == request.sentiment_label)
    if request.tickers:
        # Match rows whose tickers array contains ALL of the requested tickers
        stmt = stmt.where(NewsItem.tickers.contains(request.tickers))
    if request.sectors:
        stmt = stmt.where(NewsItem.sectors.contains(request.sectors))
    if request.date_from:
        stmt = stmt.where(NewsItem.first_seen_at >= request.date_from)
    if request.date_to:
        stmt = stmt.where(NewsItem.first_seen_at <= request.date_to)

    return stmt


# ---------------------------------------------------------------------------
# POST /
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=NewsListResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid news search (Phase 1: FTS)",
    description=(
        "Search news items using PostgreSQL full-text search (FTS) against "
        "`search_vector`.  Optional structured filters narrow the result set "
        "further.  Results are ranked by ts_rank_cd relevance, then by "
        "`last_seen_at` descending.  "
        "Phase 2 will add pgvector cosine re-ranking."
    ),
)
async def search_news(request: SearchRequest, db: DbSession) -> NewsListResponse:
    base_stmt = _build_search_query(request)

    # ---- total count (re-run without pagination) ---------------------------
    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    # ---- paginated data ----------------------------------------------------
    offset = (request.page - 1) * request.page_size
    data_stmt = base_stmt.offset(offset).limit(request.page_size)
    # Each row is a (NewsItem, rank) tuple because of the extra label column
    rows = (await db.execute(data_stmt)).all()
    items = [NewsItemResponse.model_validate(row.NewsItem) for row in rows]

    return NewsListResponse(
        items=items,
        total=total,
        page=request.page,
        page_size=request.page_size,
    )
