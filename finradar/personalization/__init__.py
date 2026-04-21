"""Personalised ranking (Phase 3 Day 2).

Turns a user's like/dislike/bookmark/dismiss history into per-sector and
per-ticker affinity scores, then converts those into a multiplicative
boost applied to feed / search ranking.

Typical call pattern:

    from finradar.personalization import get_affinity, personal_boost

    aff = get_affinity(session, user_id="owner")          # Redis-cached
    boost = personal_boost(aff, sectors=["AI"], tickers=["NVDA"])
    final_score = base_score * (1.0 + boost)
"""

from finradar.personalization.engine import (
    AffinityReport,
    clear_affinity_cache,
    get_affinity,
    personal_boost,
)

__all__ = [
    "AffinityReport",
    "clear_affinity_cache",
    "get_affinity",
    "personal_boost",
]
