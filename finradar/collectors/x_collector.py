"""
finradar.collectors.x_collector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X (formerly Twitter) API v2 collector for tracked accounts.

Cost model
----------
X moved to pay-per-resource in 2025. Every tweet returned counts toward
monthly spend at ``settings.x_cost_per_read_usd`` (default $0.005). This
collector:

  * reads ``last_seen_id`` per account from Redis and passes ``since_id``
    so already-ingested tweets are never re-fetched.
  * excludes retweets and replies by default to avoid paying twice for
    the same content or for conversation cruft.
  * limits ``max_results`` per call so a misbehaving loop cannot run away.

Safety flag
-----------
``settings.x_enabled`` gates every API call. When disabled OR when
``X_BEARER_TOKEN`` is empty, :py:meth:`XCollector.collect` returns an
empty list without hitting the network.  The Celery task layer also
enforces a monthly budget cap (see ``collect_x_posts``) as an independent
second line of defence.

Storage mapping
---------------
Each tweet becomes a :py:class:`CollectedArticle` with:

  * ``source_type``  = ``"x_feed"``
  * ``source_url``   = ``"x_account:<username>"`` (treated like an RSS feed id)
  * ``url``          = ``"https://x.com/<username>/status/<tweet_id>"``
  * ``title``        = first 100 chars of tweet text (for search indexing)
  * ``summary``      = full tweet text (up to 4,000 chars on X Premium)
  * ``raw_data.x``   = a flat dict with ``tweet_id``, ``username``,
                       ``breaking`` (bool), and ``linked_url`` (first link
                       from entities; used for Bloomberg RSS dedup).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from finradar.collectors.base import BaseCollector, CollectedArticle
from finradar.config import get_settings


logger = logging.getLogger("finradar.collectors.x")


# Tweets starting with these uppercase prefixes are tagged as breaking in raw_data.
_BREAKING_PREFIXES = ("BREAKING:", "JUST IN:", "UPDATE:", "NEW:")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_accounts(raw: str) -> list[str]:
    """Comma-separated X usernames → deduped, lowercased, @-stripped list."""
    parts = [p.strip().lstrip("@") for p in (raw or "").split(",")]
    return [p.lower() for p in parts if p]


def _extract_first_url(entities: dict | None) -> str | None:
    """Return the first expanded URL from a tweet's ``entities.urls``."""
    if not entities:
        return None
    urls = entities.get("urls") or []
    for u in urls:
        expanded = u.get("expanded_url") or u.get("url")
        if expanded:
            return expanded
    return None


def _is_breaking(text: str) -> bool:
    """Detect ``BREAKING:``-style prefix (used for downstream alerting)."""
    stripped = text.lstrip().upper()
    return any(stripped.startswith(p) for p in _BREAKING_PREFIXES)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class XCollector(BaseCollector):
    """Collect recent tweets from a list of tracked X accounts.

    Parameters
    ----------
    since_id_by_account:
        Optional mapping of ``username → last_seen_tweet_id``. The caller
        (Celery task) maintains this across runs via Redis so that each
        poll only fetches tweets newer than the previous poll.
    """

    def __init__(
        self,
        since_id_by_account: dict[str, int] | None = None,
        max_concurrent: int = 2,
    ) -> None:
        super().__init__(name="x", max_concurrent=max_concurrent)
        self._since_id = since_id_by_account or {}
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """True when both the flag and the bearer token are set."""
        return bool(self._settings.x_enabled and self._settings.x_bearer_token)

    async def collect(self) -> list[CollectedArticle]:
        """Fetch new tweets from every tracked account.

        No-op (returns ``[]``) when the collector is disabled or bearer
        token is missing — this is the intended behaviour during local
        development and whenever the user wants to pause spend.
        """
        if not self.is_enabled():
            logger.debug("XCollector disabled (x_enabled=%s, token_set=%s)",
                         self._settings.x_enabled, bool(self._settings.x_bearer_token))
            return []

        accounts = _parse_accounts(self._settings.x_tracked_accounts)
        if not accounts:
            logger.info("XCollector: no accounts configured — skipping")
            return []

        # Import tweepy lazily so the app can boot without it in dev envs
        # that haven't rebuilt the image yet.
        import tweepy  # noqa: PLC0415

        try:
            client = tweepy.Client(
                bearer_token=self._settings.x_bearer_token,
                wait_on_rate_limit=False,  # we manage rate/budget ourselves
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("XCollector: failed to construct tweepy client: %s", exc)
            return []

        # Resolve usernames → user IDs (one call, up to 100 usernames at a time).
        try:
            users_resp = client.get_users(usernames=accounts)
        except Exception as exc:  # noqa: BLE001
            logger.error("XCollector: get_users failed: %s", exc, exc_info=True)
            return []

        user_map: dict[str, dict[str, Any]] = {}
        for u in users_resp.data or []:
            user_map[u.username.lower()] = {"id": u.id, "username": u.username}

        unknown = [a for a in accounts if a not in user_map]
        if unknown:
            logger.warning("XCollector: could not resolve users: %s", unknown)

        all_articles: list[CollectedArticle] = []

        for username, info in user_map.items():
            articles = self._fetch_one_account(client, username, info["id"], info["username"])
            all_articles.extend(articles)

        logger.info(
            "XCollector: collected %d tweets across %d accounts",
            len(all_articles), len(user_map),
        )
        return all_articles

    # ------------------------------------------------------------------
    # Per-account fetch
    # ------------------------------------------------------------------

    def _fetch_one_account(
        self,
        client: Any,
        username_lower: str,
        user_id: int,
        display_username: str,
    ) -> list[CollectedArticle]:
        """Fetch new tweets for one account, mapping each to a CollectedArticle."""
        since_id = self._since_id.get(username_lower)

        try:
            # Minimal tweet_fields: only what the ingest pipeline actually consumes.
            # Each requested field expansion counts as a billed resource on X API,
            # so dropping unused fields directly lowers $/tweet. Kept:
            #   - created_at  (timestamp)
            #   - lang        (language detection)
            #   - entities    (linked_url for Bloomberg RSS dedup)
            # Dropped (previously requested, unused downstream):
            #   - public_metrics  (likes/retweets counts)
            #   - referenced_tweets (reply/quote chain metadata)
            resp = client.get_users_tweets(
                id=user_id,
                since_id=since_id,
                max_results=self._settings.x_max_tweets_per_account,
                exclude=["retweets", "replies"],
                tweet_fields=["id", "text", "created_at", "lang", "entities"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "XCollector: get_users_tweets failed for @%s: %s",
                display_username, exc, exc_info=True,
            )
            return []

        tweets = resp.data or []
        if not tweets:
            logger.debug("XCollector: no new tweets for @%s", display_username)
            return []

        source_url = f"x_account:{username_lower}"
        articles: list[CollectedArticle] = []

        for tw in tweets:
            text = (tw.text or "").strip()
            if not text:
                continue
            title = (text[:100] + "…") if len(text) > 100 else text
            tweet_id = int(tw.id)
            url = f"https://x.com/{display_username}/status/{tweet_id}"

            published_at: datetime | None = tw.created_at
            if published_at is not None and published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)

            entities = tw.entities if hasattr(tw, "entities") else None
            linked_url = _extract_first_url(entities)
            breaking = _is_breaking(text)

            articles.append(
                CollectedArticle(
                    title=title,
                    url=url,
                    source_url=source_url,
                    source_type="x_feed",
                    language=(tw.lang if hasattr(tw, "lang") and tw.lang else "en"),
                    summary=text,
                    published_at=published_at,
                    tickers=[],
                    sectors=[],
                    raw_data={
                        "x": {
                            "tweet_id": tweet_id,
                            "username": display_username,
                            "breaking": breaking,
                            "linked_url": linked_url,
                        }
                    },
                )
            )

        logger.info(
            "XCollector: @%s → %d new tweets (since_id=%s)",
            display_username, len(articles), since_id,
        )
        return articles
