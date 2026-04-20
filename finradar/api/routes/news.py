"""
finradar.api.routes.news
~~~~~~~~~~~~~~~~~~~~~~~~

Endpoints for browsing, retrieving, and acting on individual news items.

Routes
------
GET  /                       — paginated, filterable news list
GET  /{news_id}              — single news item by primary key
GET  /topics/                — list all topics
GET  /topics/{slug}          — news items that belong to a topic (by slug)
POST /feedback               — submit user feedback (bookmark / like / dislike / dismiss)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select

from finradar.api.deps import (
    CommonFilters,
    DbSession,
    PaginationParams,
    _apply_common_filters,
)
from finradar.models import NewsItem, Topic, UserFeedback
from finradar.schemas import (
    FeedbackCreate,
    FeedbackResponse,
    NewsItemResponse,
    NewsListResponse,
    TopicResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=NewsListResponse,
    summary="List latest news",
    description=(
        "Return a paginated, optionally filtered list of news items "
        "sorted by `last_seen_at` descending (most recent first)."
    ),
)
async def list_news(
    db: DbSession,
    pagination: Annotated[PaginationParams, Depends(PaginationParams)],
    filters: Annotated[CommonFilters, Depends(CommonFilters)],
) -> NewsListResponse:
    # ---- count query -------------------------------------------------------
    count_stmt = select(func.count()).select_from(NewsItem)
    count_stmt = _apply_common_filters(count_stmt, filters)
    total: int = (await db.execute(count_stmt)).scalar_one()

    # ---- data query --------------------------------------------------------
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
# GET /topics/
# NOTE: This route must be declared BEFORE /{news_id} so that FastAPI does
#       not try to match the literal string "topics" as a news_id integer.
# ---------------------------------------------------------------------------


@router.get(
    "/topics/",
    response_model=list[TopicResponse],
    summary="List all topics",
    description="Return every topic, ordered alphabetically by name.",
)
async def list_topics(db: DbSession) -> list[TopicResponse]:
    stmt = select(Topic).order_by(Topic.name.asc())
    rows = (await db.execute(stmt)).scalars().all()
    return [TopicResponse.model_validate(row) for row in rows]


# ---------------------------------------------------------------------------
# GET /topics/{slug}
# ---------------------------------------------------------------------------


@router.get(
    "/topics/{slug}",
    response_model=NewsListResponse,
    summary="News items for a topic",
    description=(
        "Return paginated news items that belong to the topic identified "
        "by `slug`, sorted by `last_seen_at` descending."
    ),
)
async def news_by_topic(
    slug: str,
    db: DbSession,
    pagination: Annotated[PaginationParams, Depends(PaginationParams)],
    filters: Annotated[CommonFilters, Depends(CommonFilters)],
) -> NewsListResponse:
    # Resolve topic
    topic_stmt = select(Topic).where(Topic.slug == slug)
    topic = (await db.execute(topic_stmt)).scalar_one_or_none()
    if topic is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic with slug '{slug}' not found.",
        )

    # Count
    count_stmt = select(func.count()).select_from(NewsItem).where(
        NewsItem.topic_id == topic.id
    )
    count_stmt = _apply_common_filters(count_stmt, filters)
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Data
    data_stmt = (
        select(NewsItem)
        .where(NewsItem.topic_id == topic.id)
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
# POST /feedback
# NOTE: Declared before /{news_id} so it isn't captured by the int route.
# ---------------------------------------------------------------------------


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit user feedback",
    description=(
        "Record an explicit user action (bookmark, like, dislike, or dismiss) "
        "on a news item.  The referenced news item must exist."
    ),
)
async def submit_feedback(payload: FeedbackCreate, db: DbSession) -> FeedbackResponse:
    # Verify the referenced news item exists
    news_stmt = select(NewsItem).where(NewsItem.id == payload.news_id)
    news_item = (await db.execute(news_stmt)).scalar_one_or_none()
    if news_item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"NewsItem with id={payload.news_id} not found.",
        )

    feedback = UserFeedback(news_id=payload.news_id, action=payload.action)
    db.add(feedback)
    await db.flush()  # populate auto-generated id / created_at before returning
    await db.refresh(feedback)
    return FeedbackResponse.model_validate(feedback)


# ---------------------------------------------------------------------------
# GET /{news_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{news_id}",
    response_model=NewsItemResponse,
    summary="Get a single news item",
    description="Return a news item by its integer primary key.",
)
async def get_news_item(news_id: int, db: DbSession) -> NewsItemResponse:
    stmt = select(NewsItem).where(NewsItem.id == news_id)
    item = (await db.execute(stmt)).scalar_one_or_none()
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"NewsItem with id={news_id} not found.",
        )
    return NewsItemResponse.model_validate(item)


# ---------------------------------------------------------------------------
# GET /{news_id}/cluster
# ---------------------------------------------------------------------------


@router.get(
    "/{news_id}/cluster",
    response_model=NewsListResponse,
    summary="Cluster siblings of a news item",
    description=(
        "Return every article that shares a cluster with the given news item, "
        "sorted by cosine similarity to the cluster representative. "
        "If the item is a singleton (cluster_rep_id is NULL), the response "
        "contains just itself with total=1."
    ),
)
async def get_news_cluster(news_id: int, db: DbSession) -> NewsListResponse:
    # Resolve the target item
    item_stmt = select(NewsItem).where(NewsItem.id == news_id)
    item = (await db.execute(item_stmt)).scalar_one_or_none()
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"NewsItem with id={news_id} not found.",
        )

    # Singleton → just the item itself
    if item.cluster_rep_id is None:
        return NewsListResponse(
            items=[NewsItemResponse.model_validate(item)],
            total=1,
            page=1,
            page_size=1,
        )

    # Fetch all siblings (including the representative itself)
    siblings_stmt = (
        select(NewsItem)
        .where(NewsItem.cluster_rep_id == item.cluster_rep_id)
        .order_by(
            NewsItem.similarity_to_rep.desc().nullslast(),
            NewsItem.last_seen_at.desc(),
        )
    )
    rows = (await db.execute(siblings_stmt)).scalars().all()
    return NewsListResponse(
        items=[NewsItemResponse.model_validate(r) for r in rows],
        total=len(rows),
        page=1,
        page_size=len(rows) or 1,
    )
