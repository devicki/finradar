"""
finradar.api.routes.feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User feedback management: like / dislike / bookmark / dismiss.

Phase 3 hardcodes the actor as ``"owner"``. Phase 4 will replace
:py:data:`_current_user_id` with a real auth dependency once multi-user +
sessions ship — all downstream code (bookmarks page, dismiss filter,
personalisation engine) already reads from this single helper.

Action semantics
----------------
* ``like``      — positive personal signal; mutually exclusive with
                  ``dislike`` on the same (user, news).
* ``dislike``   — negative personal signal; mutually exclusive with ``like``.
* ``bookmark``  — "save for later"; independent of like/dislike.
* ``dismiss``   — "hide from my default feed"; independent of the others.

The :py:class:`UserFeedback` table has ``UNIQUE(user_id, news_id, action)``
so repeated clicks on the same button are idempotent at the DB layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, desc, select
from sqlalchemy.exc import IntegrityError

from finradar.api.deps import DbSession, PaginationParams
from finradar.models import NewsItem, UserFeedback
from finradar.schemas import NewsItemResponse, NewsListResponse


router = APIRouter()


# ---------------------------------------------------------------------------
# Current-user resolver (Phase 3 hardcoded; Phase 4 replaces with auth dep)
# ---------------------------------------------------------------------------


_DEFAULT_USER_ID = "owner"


def _current_user_id() -> str:
    """Return the actor for every feedback operation.

    Hardcoded for Phase 3. Swap with a proper Depends(get_current_user)
    when auth lands in Phase 4 — the rest of this module doesn't need to
    change.
    """
    return _DEFAULT_USER_ID


CurrentUser = Annotated[str, Depends(_current_user_id)]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


_ActionLiteral = Literal["like", "dislike", "bookmark", "dismiss"]


class FeedbackRequest(BaseModel):
    news_id: int = Field(..., description="Target news item id")
    action: _ActionLiteral = Field(..., description="like | dislike | bookmark | dismiss")


class FeedbackRow(BaseModel):
    id: int
    user_id: str
    news_id: int
    action: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class FeedbackStatus(BaseModel):
    news_id: int
    actions: list[str] = Field(
        default_factory=list,
        description="Actions this user currently has on the news item.",
    )


class FeedbackBatchRequest(BaseModel):
    news_ids: list[int] = Field(..., min_length=1, max_length=200)


class FeedbackBatchResponse(BaseModel):
    # news_id -> list of action strings
    states: dict[int, list[str]]


# ---------------------------------------------------------------------------
# POST /feedback — upsert + mutual-exclusion enforcement
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=FeedbackRow,
    status_code=status.HTTP_200_OK,
    summary="Record (or confirm) feedback",
    description=(
        "Upserts a feedback row. Clicking `like` on an article that already "
        "has `dislike` from the same user clears the dislike first, and "
        "vice versa — the two are mutually exclusive. bookmark and dismiss "
        "are independent of everything else."
    ),
)
async def create_feedback(
    payload: FeedbackRequest, db: DbSession, user_id: CurrentUser
) -> FeedbackRow:
    # Verify the referenced article exists (catch typos + broken FKs early).
    news = (
        await db.execute(select(NewsItem.id).where(NewsItem.id == payload.news_id))
    ).scalar_one_or_none()
    if news is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"NewsItem {payload.news_id} not found.",
        )

    # Mutual-exclusion: like ↔ dislike
    opposite = {"like": "dislike", "dislike": "like"}.get(payload.action)
    if opposite is not None:
        await db.execute(
            delete(UserFeedback).where(
                UserFeedback.user_id == user_id,
                UserFeedback.news_id == payload.news_id,
                UserFeedback.action == opposite,
            )
        )

    # Upsert the requested action. Unique (user_id, news_id, action) means a
    # second click just returns the existing row.
    existing = (
        await db.execute(
            select(UserFeedback).where(
                UserFeedback.user_id == user_id,
                UserFeedback.news_id == payload.news_id,
                UserFeedback.action == payload.action,
            )
        )
    ).scalar_one_or_none()

    if existing is not None:
        await db.commit()
        return FeedbackRow.model_validate(existing)

    row = UserFeedback(
        user_id=user_id,
        news_id=payload.news_id,
        action=payload.action,
    )
    db.add(row)
    try:
        await db.flush()
        await db.refresh(row)
    except IntegrityError:
        # Race with a concurrent insert — treat as idempotent.
        await db.rollback()
        existing = (
            await db.execute(
                select(UserFeedback).where(
                    UserFeedback.user_id == user_id,
                    UserFeedback.news_id == payload.news_id,
                    UserFeedback.action == payload.action,
                )
            )
        ).scalar_one()
        return FeedbackRow.model_validate(existing)

    await db.commit()
    return FeedbackRow.model_validate(row)


# ---------------------------------------------------------------------------
# DELETE /feedback/{news_id}/{action} — remove a specific feedback row
# ---------------------------------------------------------------------------


@router.delete(
    "/{news_id}/{action}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a feedback row",
    description=(
        "Clears the user's feedback for the given action on the given news "
        "item. No-op (still 204) when the row doesn't exist, so clients can "
        "implement simple toggle-off UI without pre-checking."
    ),
)
async def delete_feedback(
    news_id: int,
    action: _ActionLiteral,
    db: DbSession,
    user_id: CurrentUser,
) -> None:
    await db.execute(
        delete(UserFeedback).where(
            UserFeedback.user_id == user_id,
            UserFeedback.news_id == news_id,
            UserFeedback.action == action,
        )
    )
    await db.commit()


# ---------------------------------------------------------------------------
# GET /feedback/status/{news_id} — UI state for a single article
# ---------------------------------------------------------------------------


@router.get(
    "/status/{news_id}",
    response_model=FeedbackStatus,
    summary="List actions this user has on a news item",
    description=(
        "Used by the dashboard to render feedback buttons in their current "
        "state (outlined vs filled). Empty list means the user hasn't "
        "interacted with this article yet."
    ),
)
async def get_feedback_status(
    news_id: int, db: DbSession, user_id: CurrentUser
) -> FeedbackStatus:
    rows = (
        await db.execute(
            select(UserFeedback.action).where(
                UserFeedback.user_id == user_id,
                UserFeedback.news_id == news_id,
            )
        )
    ).scalars().all()
    return FeedbackStatus(news_id=news_id, actions=list(rows))


# ---------------------------------------------------------------------------
# POST /feedback/status/batch — bulk status lookup for a page of cards
# ---------------------------------------------------------------------------


@router.post(
    "/status/batch",
    response_model=FeedbackBatchResponse,
    summary="Bulk feedback state for a list of news ids",
    description=(
        "The dashboard calls this once per rendered page to paint all four "
        "feedback buttons in their current state. One SQL round-trip instead "
        "of one per card."
    ),
)
async def get_feedback_status_batch(
    payload: FeedbackBatchRequest, db: DbSession, user_id: CurrentUser
) -> FeedbackBatchResponse:
    rows = (
        await db.execute(
            select(UserFeedback.news_id, UserFeedback.action).where(
                UserFeedback.user_id == user_id,
                UserFeedback.news_id.in_(payload.news_ids),
            )
        )
    ).all()

    states: dict[int, list[str]] = {nid: [] for nid in payload.news_ids}
    for nid, action in rows:
        states.setdefault(nid, []).append(action)
    return FeedbackBatchResponse(states=states)


# ---------------------------------------------------------------------------
# GET /feedback/bookmarks — paginated list of bookmarked articles
# ---------------------------------------------------------------------------


@router.get(
    "/bookmarks",
    response_model=NewsListResponse,
    summary="My bookmarked articles",
    description=(
        "Returns the user's bookmarked news items, newest bookmark first. "
        "Paginates via the shared ?page&page_size query params."
    ),
)
async def list_bookmarks(
    db: DbSession,
    user_id: CurrentUser,
    pagination: Annotated[PaginationParams, Depends(PaginationParams)],
) -> NewsListResponse:
    return await _list_by_action(db, user_id, "bookmark", pagination)


# ---------------------------------------------------------------------------
# GET /feedback/dismissed — audit / unhide UX for the Bookmarks page
# ---------------------------------------------------------------------------


@router.get(
    "/dismissed",
    response_model=NewsListResponse,
    summary="My dismissed articles (hidden from default feeds)",
    description=(
        "Lets the dashboard show a 'Recently hidden' section so the user "
        "can un-dismiss an article they hid by accident."
    ),
)
async def list_dismissed(
    db: DbSession,
    user_id: CurrentUser,
    pagination: Annotated[PaginationParams, Depends(PaginationParams)],
) -> NewsListResponse:
    return await _list_by_action(db, user_id, "dismiss", pagination)


# ---------------------------------------------------------------------------
# Internal helper: paginate articles the user has taken a given action on
# ---------------------------------------------------------------------------


async def _list_by_action(
    db, user_id: str, action: str, pagination: PaginationParams
) -> NewsListResponse:
    # Count first so the UI can render a total even when the current page
    # fits well within the first fetch.
    count_stmt = select(UserFeedback.id).where(
        UserFeedback.user_id == user_id,
        UserFeedback.action == action,
    )
    total = len((await db.execute(count_stmt)).all())

    # Join news_items via the feedback FK, ordered by newest feedback first.
    data_stmt = (
        select(NewsItem)
        .join(UserFeedback, UserFeedback.news_id == NewsItem.id)
        .where(
            UserFeedback.user_id == user_id,
            UserFeedback.action == action,
        )
        .order_by(desc(UserFeedback.created_at))
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    rows = (await db.execute(data_stmt)).scalars().all()
    return NewsListResponse(
        items=[NewsItemResponse.model_validate(r) for r in rows],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )
