"""
finradar.personalization.engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute sector/ticker affinity for a user from their feedback history and
turn it into a multiplicative personal_boost applied on top of existing
ranking scores.

Model
-----
For each (user, sector) and (user, ticker) pair we accumulate a signed
score over all of the user's feedback rows:

    +1.0   for every 'like' row on an article that mentions the tag
    +0.5   for 'bookmark'   (save-for-later signal: weaker than like)
    -1.0   for 'dislike'
    -0.3   for 'dismiss'    (don't-show-again signal: weak negative)

Then we **normalise** so heavy likers don't accidentally dominate:

    affinity[tag] = raw_score / (N_total_feedback_rows + SMOOTHING_PRIOR)

With ``SMOOTHING_PRIOR=5`` a user with zero feedback gets zero boost (cold
start friendly), and after ~10 diverse feedback rows the signal settles
into a stable ``[-1, +1]`` range.

At query time:

    personal_boost(article) =
        W_SECTOR * mean(affinity[s] for s in article.sectors)
      + W_TICKER * mean(affinity[t] for t in article.tickers)

    final_score = base_score * (1.0 + personal_boost)

Caching
-------
Affinity recomputation is cheap (single SELECT + Python aggregation over
at most a few thousand feedback rows) but we cache the result in Redis
with a short TTL so search / feed requests don't pay the cost on every
hit. The cache is invalidated from :py:func:`clear_affinity_cache`, which
feedback mutation code can call if it wants strong freshness — Phase 3
doesn't bother because the TTL is short.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from finradar.config import get_settings
from finradar.models import NewsItem, UserFeedback


logger = logging.getLogger("finradar.personalization")


# ---------------------------------------------------------------------------
# Tunables — conservative defaults friendly to single-user Phase 3 usage
# ---------------------------------------------------------------------------

ACTION_WEIGHT: dict[str, float] = {
    "like": 1.0,
    "bookmark": 0.5,
    "dislike": -1.0,
    "dismiss": -0.3,
}

SMOOTHING_PRIOR = 5.0   # feedback rows worth of zero-affinity "background"
W_SECTOR = 0.6          # share of personal_boost driven by sector affinity
W_TICKER = 0.4          # ...and by ticker affinity (sums to 1.0)

# personal_boost is clipped into this band so a heavily biased user can't
# drive a score to 0 or 2× overnight — keeps personalisation as a "nudge".
BOOST_MIN = -0.5
BOOST_MAX = 0.5

# Redis cache
_CACHE_TTL_SEC = 300            # 5 minutes
_CACHE_KEY_FMT = "finradar:affinity:{user_id}"


# ---------------------------------------------------------------------------
# Report type — lightweight enough to cache as JSON
# ---------------------------------------------------------------------------


@dataclass
class AffinityReport:
    user_id: str
    feedback_rows: int
    sectors: dict[str, float] = field(default_factory=dict)
    tickers: dict[str, float] = field(default_factory=dict)

    def top_sectors(self, k: int = 5, positive_only: bool = False) -> list[tuple[str, float]]:
        items = sorted(self.sectors.items(), key=lambda kv: kv[1], reverse=True)
        if positive_only:
            items = [(t, s) for t, s in items if s > 0]
        return items[:k]

    def top_tickers(self, k: int = 5, positive_only: bool = False) -> list[tuple[str, float]]:
        items = sorted(self.tickers.items(), key=lambda kv: kv[1], reverse=True)
        if positive_only:
            items = [(t, s) for t, s in items if s > 0]
        return items[:k]

    def to_json_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "feedback_rows": self.feedback_rows,
            "sectors": self.sectors,
            "tickers": self.tickers,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "AffinityReport":
        return cls(
            user_id=data["user_id"],
            feedback_rows=int(data.get("feedback_rows", 0)),
            sectors=dict(data.get("sectors") or {}),
            tickers=dict(data.get("tickers") or {}),
        )


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def _redis():
    """Lazy Redis handle; lets the module import when redis lib isn't needed."""
    import redis  # noqa: PLC0415

    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _cache_load(user_id: str) -> AffinityReport | None:
    try:
        r = _redis()
        raw = r.get(_CACHE_KEY_FMT.format(user_id=user_id))
        if not raw:
            return None
        return AffinityReport.from_json_dict(json.loads(raw))
    except Exception as exc:  # noqa: BLE001
        logger.debug("affinity cache load failed (%s): %s", user_id, exc)
        return None


def _cache_store(report: AffinityReport) -> None:
    try:
        r = _redis()
        r.setex(
            _CACHE_KEY_FMT.format(user_id=report.user_id),
            _CACHE_TTL_SEC,
            json.dumps(report.to_json_dict()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("affinity cache store failed (%s): %s", report.user_id, exc)


def clear_affinity_cache(user_id: str) -> None:
    """Drop a user's cached report — call after feedback mutations if needed."""
    try:
        _redis().delete(_CACHE_KEY_FMT.format(user_id=user_id))
    except Exception:  # noqa: BLE001 — cache is best-effort
        pass


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _compute_affinity(session: Session, user_id: str) -> AffinityReport:
    """Walk the user's feedback rows + joined article tags to build the report."""
    rows = session.execute(
        select(UserFeedback.action, NewsItem.sectors, NewsItem.tickers)
        .join(NewsItem, NewsItem.id == UserFeedback.news_id)
        .where(UserFeedback.user_id == user_id)
    ).all()

    raw_sectors: dict[str, float] = {}
    raw_tickers: dict[str, float] = {}
    total = 0

    for action, sectors, tickers in rows:
        weight = ACTION_WEIGHT.get(action)
        if weight is None:
            continue
        total += 1
        for s in sectors or []:
            if not s:
                continue
            raw_sectors[s] = raw_sectors.get(s, 0.0) + weight
        for t in tickers or []:
            if not t:
                continue
            raw_tickers[t.upper()] = raw_tickers.get(t.upper(), 0.0) + weight

    denom = total + SMOOTHING_PRIOR
    sectors = {k: v / denom for k, v in raw_sectors.items()}
    tickers = {k: v / denom for k, v in raw_tickers.items()}

    return AffinityReport(
        user_id=user_id,
        feedback_rows=total,
        sectors=sectors,
        tickers=tickers,
    )


def get_affinity(session: Session, user_id: str) -> AffinityReport:
    """Return the user's AffinityReport, using Redis cache when warm."""
    cached = _cache_load(user_id)
    if cached is not None:
        return cached
    report = _compute_affinity(session, user_id)
    _cache_store(report)
    return report


# ---------------------------------------------------------------------------
# Boost application
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def personal_boost(
    affinity: AffinityReport,
    *,
    sectors: Sequence[str] | None,
    tickers: Sequence[str] | None,
) -> float:
    """Compute the multiplicative boost for one article.

    Returns a value in ``[BOOST_MIN, BOOST_MAX]``. Zero when the affinity
    maps are empty (cold start) or the article has no sectors/tickers.
    """
    if affinity.feedback_rows == 0:
        return 0.0

    sector_hits = [
        affinity.sectors.get(s, 0.0) for s in (sectors or []) if s
    ]
    ticker_hits = [
        affinity.tickers.get((t or "").upper(), 0.0) for t in (tickers or []) if t
    ]

    sector_component = _mean(sector_hits)
    ticker_component = _mean(ticker_hits)

    boost = W_SECTOR * sector_component + W_TICKER * ticker_component
    return max(BOOST_MIN, min(BOOST_MAX, boost))
