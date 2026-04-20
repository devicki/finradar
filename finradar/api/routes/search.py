"""
finradar.api.routes.search
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hybrid news search endpoint.

Ranking combines three signals:

.. math::

    final = w_{bm25} \\cdot \\text{fts} + w_{cos} \\cdot \\text{cosine} + w_{rec} \\cdot \\text{recency}

* **fts**      — PostgreSQL ``ts_rank_cd`` (normalised to [0, 1] within the
                 candidate pool using per-query max).
* **cosine**   — pgvector cosine similarity = ``1 − (embedding <=> query_vec)``.
* **recency**  — exponential decay on ``last_seen_at`` with a 7-day half-life.

Candidate pool is every row that EITHER matches the tsquery OR has a
cosine similarity ≥ 0.5.  That set is re-ranked by the weighted score and
paginated.

Weights default to ``settings.rank_weight_*`` but can be overridden
per-request via ``weight_bm25 / weight_cosine / weight_recency``.

Routes
------
POST /    — hybrid search with optional structured filters + score breakdown
"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import Float, and_, cast, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, status

from finradar.api.deps import DbSession, EmbeddingDep
from finradar.config import get_settings
from finradar.models import NewsItem
from finradar.schemas import (
    NewsItemSearchResponse,
    NewsSearchListResponse,
    ScoreBreakdown,
    SearchRequest,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Candidate pool: include rows with cosine similarity ≥ this threshold even
# if FTS doesn't match. Low enough to catch semantically related items,
# high enough to keep the pool focused.
_VECTOR_CANDIDATE_THRESHOLD: float = 0.5

# Recency half-life in days. EXP(-dt / half_life_sec) gives 1.0 for now,
# 0.5 at +half_life days, 0.25 at +2*half_life, …
_RECENCY_HALF_LIFE_DAYS: float = 7.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_weights(request: SearchRequest) -> tuple[float, float, float]:
    """Return (w_bm25, w_cos, w_rec) using request overrides or defaults."""
    settings = get_settings()
    return (
        request.weight_bm25 if request.weight_bm25 is not None else settings.rank_weight_bm25,
        request.weight_cosine if request.weight_cosine is not None else settings.rank_weight_cosine,
        request.weight_recency if request.weight_recency is not None else settings.rank_weight_recency,
    )


async def _embed_query(text_value: str, embedder: Any) -> list[float]:
    """Run synchronous SentenceTransformer inference in a thread pool.

    Embedding generation is CPU/GPU-bound and would otherwise block the
    FastAPI event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedder.generate, text_value)


def _build_hybrid_sql(request: SearchRequest) -> text:
    """Build the parameterised hybrid-search SQL statement.

    Parameters expected at execution time:
      * ``q_text``   — the raw user query (plainto_tsquery input)
      * ``q_vec``    — the 384-dim query embedding (pgvector literal)
      * ``offset``   — pagination offset
      * ``limit``    — pagination limit
      * ``w_bm25``, ``w_cos``, ``w_rec`` — ranking weights

    Optional filter parameters are bound only when the corresponding field
    is provided in ``request``; unused placeholders are left off the SQL.
    """
    # --- Collect filter clauses as raw SQL fragments ------------------------
    # (parameterised via SQLAlchemy .bindparams() to avoid injection)
    filter_sql: list[str] = []
    if request.source_type:
        filter_sql.append("AND source_type = :source_type")
    if request.language:
        filter_sql.append("AND language = :language")
    if request.sentiment_label:
        filter_sql.append("AND sentiment_label = :sentiment_label")
    if request.tickers:
        filter_sql.append("AND tickers @> :tickers")
    if request.sectors:
        filter_sql.append("AND sectors @> :sectors")
    if request.date_from:
        filter_sql.append("AND first_seen_at >= :date_from")
    if request.date_to:
        filter_sql.append("AND first_seen_at <= :date_to")

    filters_clause = "\n          ".join(filter_sql)

    # --- Hybrid query --------------------------------------------------------
    # Use CAST(:q_vec AS vector) so pgvector parses the embedding literal.
    # The cosine-distance operator is <=>; similarity = 1 - distance.
    sql = f"""
    WITH q AS (
        SELECT plainto_tsquery('english', :q_text) AS ts,
               CAST(:q_vec AS vector)              AS vec
    ),
    candidates AS (
        SELECT
            n.id,
            ts_rank_cd(n.search_vector, q.ts, 1)                           AS fts_raw,
            CASE
                WHEN n.embedding IS NULL THEN 0.0
                ELSE GREATEST(0.0, 1.0 - (n.embedding <=> q.vec))
            END                                                            AS cos_score,
            EXP(
                -EXTRACT(EPOCH FROM (NOW() - n.last_seen_at))
                / (86400.0 * :half_life_days)
            )                                                              AS recency_score
        FROM news_items n, q
        WHERE (
              n.search_vector @@ q.ts
           OR (n.embedding IS NOT NULL
               AND 1.0 - (n.embedding <=> q.vec) >= :vec_threshold)
        )
        {filters_clause}
    ),
    normalized AS (
        SELECT
            id,
            CASE
                WHEN MAX(fts_raw) OVER () > 0 THEN fts_raw / MAX(fts_raw) OVER ()
                ELSE 0.0
            END                                                            AS fts_score,
            cos_score,
            recency_score
        FROM candidates
    ),
    scored AS (
        SELECT
            id,
            fts_score,
            cos_score,
            recency_score,
            (:w_bm25 * fts_score + :w_cos * cos_score + :w_rec * recency_score) AS final_score
        FROM normalized
    ),
    page AS (
        SELECT id, fts_score, cos_score, recency_score, final_score
        FROM scored
        ORDER BY final_score DESC, id DESC
        OFFSET :offset LIMIT :limit
    )
    SELECT
        n.*,
        page.fts_score,
        page.cos_score,
        page.recency_score,
        page.final_score,
        (SELECT COUNT(*) FROM scored) AS total_count
    FROM page
    JOIN news_items n ON n.id = page.id
    ORDER BY page.final_score DESC, page.id DESC
    """
    return text(sql)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=NewsSearchListResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid news search (FTS + pgvector + recency)",
    description=(
        "Combined ranking over PostgreSQL full-text search, pgvector cosine "
        "similarity, and exponential recency decay. Weights default to "
        "`rank_weight_bm25 / rank_weight_cosine / rank_weight_recency` from "
        "settings and can be overridden per-request. Pass `include_scores=true` "
        "to receive per-signal score breakdowns on each item."
    ),
)
async def search_news(
    request: SearchRequest, db: DbSession, embedder: EmbeddingDep
) -> NewsSearchListResponse:
    # ---- 1. Generate query embedding (thread-pooled) -----------------------
    query_vec = await _embed_query(request.query, embedder)

    # pgvector accepts a string literal like "[0.1,0.2,…]" — quick format.
    q_vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"

    # ---- 2. Resolve weights ------------------------------------------------
    w_bm25, w_cos, w_rec = _resolve_weights(request)

    # ---- 3. Execute hybrid SQL --------------------------------------------
    stmt = _build_hybrid_sql(request)

    offset = (request.page - 1) * request.page_size
    params: dict[str, Any] = {
        "q_text": request.query,
        "q_vec": q_vec_literal,
        "vec_threshold": _VECTOR_CANDIDATE_THRESHOLD,
        "half_life_days": _RECENCY_HALF_LIFE_DAYS,
        "w_bm25": w_bm25,
        "w_cos": w_cos,
        "w_rec": w_rec,
        "offset": offset,
        "limit": request.page_size,
    }
    if request.source_type:
        params["source_type"] = request.source_type
    if request.language:
        params["language"] = request.language
    if request.sentiment_label:
        params["sentiment_label"] = request.sentiment_label
    if request.tickers:
        params["tickers"] = request.tickers
    if request.sectors:
        params["sectors"] = request.sectors
    if request.date_from:
        params["date_from"] = request.date_from
    if request.date_to:
        params["date_to"] = request.date_to

    result = await db.execute(stmt, params)
    rows = result.mappings().all()

    total: int = int(rows[0]["total_count"]) if rows else 0

    # ---- 4. Marshall rows into Pydantic ------------------------------------
    items: list[NewsItemSearchResponse] = []
    for row in rows:
        # NewsItemSearchResponse inherits NewsItemResponse fields; the row
        # dict from RowMapping includes every NewsItem column plus our
        # computed score columns.
        payload: dict[str, Any] = {
            key: row[key] for key in NewsItemSearchResponse.model_fields.keys()
            if key in row and key not in ("score", "score_breakdown")
        }
        payload["score"] = float(row["final_score"])
        if request.include_scores:
            payload["score_breakdown"] = ScoreBreakdown(
                fts=float(row["fts_score"]),
                cosine=float(row["cos_score"]),
                recency=float(row["recency_score"]),
                final=float(row["final_score"]),
            )
        items.append(NewsItemSearchResponse.model_validate(payload))

    return NewsSearchListResponse(
        items=items,
        total=total,
        page=request.page,
        page_size=request.page_size,
    )
