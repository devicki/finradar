"""
finradar.api.deps
~~~~~~~~~~~~~~~~~

Shared FastAPI dependency functions used across all route modules.

Provides:
- get_db                — async DB session (re-exported from db.session)
- PaginationParams      — query-string pagination (page, page_size)
- CommonFilters         — query-string common filters (source_type, language,
                          sentiment_label, ticker, sector, date_from, date_to)
- _apply_common_filters — helper that applies CommonFilters to a SELECT stmt
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

# Re-export so callers only need to import from finradar.api.deps
from finradar.db.session import get_db  # noqa: F401

if TYPE_CHECKING:
    from sqlalchemy import Select

    from finradar.processors.embeddings import EmbeddingGenerator

__all__ = [
    "get_db",
    "get_embedding_generator",
    "PaginationParams",
    "CommonFilters",
    "DbSession",
    "EmbeddingDep",
    "_apply_common_filters",
]

# ---------------------------------------------------------------------------
# Typed alias — convenience for route function signatures
# ---------------------------------------------------------------------------

DbSession = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# Shared EmbeddingGenerator (lazy singleton)
# ---------------------------------------------------------------------------
#
# The EmbeddingGenerator loads a ~90MB sentence-transformers model on first
# use. We keep a module-level singleton so repeated /search calls share the
# same loaded model. CUDA is auto-detected; the API container has no GPU
# access so this will fall back to CPU (still <100ms per query).

_embedding_generator: "EmbeddingGenerator | None" = None


def get_embedding_generator() -> "EmbeddingGenerator":
    """FastAPI dependency returning a process-wide EmbeddingGenerator."""
    global _embedding_generator
    if _embedding_generator is None:
        # Import here to avoid loading torch / transformers at module import
        # time (startup overhead).  First /search call triggers model load.
        from finradar.config import get_settings
        from finradar.processors.embeddings import EmbeddingGenerator

        settings = get_settings()
        _embedding_generator = EmbeddingGenerator(
            model_name=settings.embedding_model,
            device=settings.local_model_device,
        )
    return _embedding_generator


EmbeddingDep = Annotated["EmbeddingGenerator", Depends(get_embedding_generator)]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaginationParams:
    """
    Standard pagination query parameters.

    Injected via ``Depends(PaginationParams)`` in route handlers.
    """

    page: int = Query(default=1, ge=1, description="1-based page number")
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
        alias="page_size",
        description="Number of items per page (max 100)",
    )

    @property
    def offset(self) -> int:
        """Compute SQL OFFSET from page and page_size."""
        return (self.page - 1) * self.page_size


# ---------------------------------------------------------------------------
# Common query filters (shared by /news/ list and /feed/)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommonFilters:
    """
    Optional filtering parameters shared across list / feed endpoints.

    Injected via ``Depends(CommonFilters)`` in route handlers.
    """

    source_type: str | None = Query(
        default=None,
        description="Filter by source type: rss | api | x_feed | url_report",
    )
    language: str | None = Query(
        default=None,
        description="ISO 639-1 language code filter, e.g. en | ko | ja",
    )
    sentiment_label: str | None = Query(
        default=None,
        description="FinBERT label filter: positive | negative | neutral",
    )
    ticker: str | None = Query(
        default=None,
        description="Filter news items that mention a specific ticker (e.g. AAPL)",
    )
    sector: str | None = Query(
        default=None,
        description="Filter news items that belong to a specific sector (e.g. AI)",
    )
    date_from: str | None = Query(
        default=None,
        description="ISO-8601 lower bound for first_seen_at (inclusive)",
    )
    date_to: str | None = Query(
        default=None,
        description="ISO-8601 upper bound for first_seen_at (inclusive)",
    )


# ---------------------------------------------------------------------------
# Filter application helper
# ---------------------------------------------------------------------------


def _apply_common_filters(stmt: "Select", filters: CommonFilters) -> "Select":
    """
    Apply the predicates from a ``CommonFilters`` instance to *stmt*.

    Importing ``NewsItem`` is deferred inside the function to avoid a
    circular import (deps <- routes <- deps).  The import is cheap since
    the module is already loaded by the time any route is called.
    """
    from finradar.models import NewsItem  # local import to break circular dep

    if filters.source_type:
        stmt = stmt.where(NewsItem.source_type == filters.source_type)
    if filters.language:
        stmt = stmt.where(NewsItem.language == filters.language)
    if filters.sentiment_label:
        stmt = stmt.where(NewsItem.sentiment_label == filters.sentiment_label)
    if filters.ticker:
        stmt = stmt.where(NewsItem.tickers.contains([filters.ticker]))
    if filters.sector:
        stmt = stmt.where(NewsItem.sectors.contains([filters.sector]))
    if filters.date_from:
        try:
            dt_from = datetime.fromisoformat(filters.date_from)
            stmt = stmt.where(NewsItem.first_seen_at >= dt_from)
        except ValueError:
            pass  # silently skip malformed date strings
    if filters.date_to:
        try:
            dt_to = datetime.fromisoformat(filters.date_to)
            stmt = stmt.where(NewsItem.first_seen_at <= dt_to)
        except ValueError:
            pass
    return stmt
