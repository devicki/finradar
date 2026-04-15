"""NewsAPI.org collector.

Fetches business/finance headlines and keyword-based articles from the
NewsAPI.org v2 REST API.  Two endpoints are used:

* ``/v2/top-headlines`` — top-of-the-moment business/finance stories.
* ``/v2/everything`` — broader search for key economic terms.

Rate limiting
-------------
The free tier allows 100 requests per day.  This collector is intentionally
conservative: it issues at most 1 + len(FINANCE_KEYWORDS) requests per
:py:meth:`collect` call (currently 6 total).  The calling code should
schedule collection no more frequently than every ~4 hours to stay within
limits.

Usage::

    from finradar.config import get_settings

    async with NewsAPICollector(api_key=get_settings().NEWSAPI_KEY) as collector:
        articles = await collector.safe_collect()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from finradar.collectors.base import BaseCollector, CollectedArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEWSAPI_BASE_URL = "https://newsapi.org/v2"

# Keywords searched via the /everything endpoint.  Each keyword generates one
# API request, so keep this list short to preserve the daily quota.
FINANCE_KEYWORDS: list[str] = [
    "economy",
    "market",
    "stocks",
    "fed",
    "inflation",
]

# NewsAPI language codes we accept.  Extend as needed.
_ACCEPTED_LANGUAGES = {"en"}


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class NewsAPICollector(BaseCollector):
    """Collect financial news from the NewsAPI.org v2 API.

    Parameters
    ----------
    api_key:
        NewsAPI.org API key.  If ``None`` or empty, :py:meth:`collect` will
        log an error and return an empty list rather than making a useless
        (and quota-wasting) unauthenticated request.
    page_size:
        Number of articles to request per API call (max 100 for the free tier).
    max_concurrent:
        Maximum number of simultaneous HTTP requests.  In practice the
        NewsAPI free tier is rate-limited to 1 req/s, so a value of 3 is
        sufficient to keep the pipeline moving without hitting 429s.
    """

    def __init__(
        self,
        api_key: str | None = None,
        page_size: int = 20,
        max_concurrent: int = 3,
    ) -> None:
        super().__init__(name="newsapi", max_concurrent=max_concurrent)
        self.api_key = api_key or ""
        self.page_size = min(page_size, 100)  # API hard cap

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def collect(self) -> list[CollectedArticle]:
        """Fetch headlines and keyword articles from NewsAPI.

        Issues one top-headlines request plus one /everything request per
        keyword in :py:data:`FINANCE_KEYWORDS`.
        """
        if not self.api_key:
            self.logger.error(
                "NewsAPI key is not configured — skipping collection. "
                "Set NEWSAPI_KEY in your environment / .env file."
            )
            return []

        all_articles: list[CollectedArticle] = []

        # Top business/finance headlines (single request)
        headline_articles = await self._fetch_headlines()
        all_articles.extend(headline_articles)

        # Keyword-based broad search (one request per keyword)
        import asyncio

        keyword_tasks = [self._fetch_everything(kw) for kw in FINANCE_KEYWORDS]
        keyword_results = await asyncio.gather(*keyword_tasks, return_exceptions=False)
        for batch in keyword_results:
            all_articles.extend(batch)

        # Deduplicate by URL preserving insertion order
        seen_urls: set[str] = set()
        unique_articles: list[CollectedArticle] = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        return unique_articles

    # ------------------------------------------------------------------
    # Private fetch methods
    # ------------------------------------------------------------------

    async def _fetch_headlines(self) -> list[CollectedArticle]:
        """Fetch top business/finance headlines via ``/v2/top-headlines``."""
        params: dict[str, Any] = {
            "category": "business",
            "language": "en",
            "pageSize": self.page_size,
            "page": 1,
        }
        endpoint = f"{NEWSAPI_BASE_URL}/top-headlines"
        return await self._get_articles(endpoint, params, source_tag="top-headlines")

    async def _fetch_everything(self, query: str) -> list[CollectedArticle]:
        """Fetch articles matching *query* via ``/v2/everything``.

        Sorted by popularity so the most-read articles surface first.
        The ``language=en`` filter limits results to English-language sources.
        """
        params: dict[str, Any] = {
            "q": query,
            "language": "en",
            "sortBy": "popularity",
            "pageSize": self.page_size,
            "page": 1,
        }
        endpoint = f"{NEWSAPI_BASE_URL}/everything"
        return await self._get_articles(endpoint, params, source_tag=f"everything:{query}")

    # ------------------------------------------------------------------
    # HTTP + parsing helpers
    # ------------------------------------------------------------------

    async def _get_articles(
        self,
        endpoint: str,
        params: dict[str, Any],
        source_tag: str,
    ) -> list[CollectedArticle]:
        """Make one GET request to *endpoint* and parse the article list.

        Handles HTTP errors, NewsAPI-level error responses, and malformed JSON
        without raising, so a single failing endpoint doesn't abort the whole
        collection run.
        """
        async with self.semaphore:
            try:
                client = await self.get_client()
                response = await client.get(
                    endpoint,
                    params=params,
                    headers={"X-Api-Key": self.api_key},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 401:
                    self.logger.error(
                        "NewsAPI returned 401 Unauthorized — check your API key."
                    )
                elif status == 426:
                    self.logger.warning(
                        "NewsAPI returned 426 Upgrade Required — "
                        "endpoint may require a paid plan."
                    )
                elif status == 429:
                    self.logger.warning("NewsAPI rate limit hit (429) on %s", endpoint)
                else:
                    self.logger.warning(
                        "HTTP %s from NewsAPI endpoint %s", status, endpoint
                    )
                return []
            except httpx.RequestError as exc:
                self.logger.warning("Request error reaching NewsAPI (%s): %s", endpoint, exc)
                return []

        try:
            data: dict[str, Any] = response.json()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to decode NewsAPI JSON response: %s", exc)
            return []

        # NewsAPI signals errors with {"status": "error", "code": ..., "message": ...}
        if data.get("status") != "ok":
            self.logger.warning(
                "NewsAPI error response [%s]: %s",
                data.get("code", "unknown"),
                data.get("message", "no message"),
            )
            return []

        raw_articles: list[dict[str, Any]] = data.get("articles") or []
        articles: list[CollectedArticle] = []
        for raw in raw_articles:
            article = self._parse_article(raw, source_tag=source_tag)
            if article is not None:
                articles.append(article)

        self.logger.debug(
            "Parsed %d articles from NewsAPI %s", len(articles), source_tag
        )
        return articles

    def _parse_article(
        self,
        article_data: dict[str, Any],
        source_tag: str = "newsapi",
    ) -> CollectedArticle | None:
        """Convert a NewsAPI article JSON object to a :py:class:`CollectedArticle`.

        Returns ``None`` for articles that are removed/redacted (NewsAPI
        sometimes returns ``[Removed]`` placeholders), or that are missing
        required fields.
        """
        title: str = (article_data.get("title") or "").strip()
        url: str = (article_data.get("url") or "").strip()

        # Skip removed/redacted articles and entries missing title or URL
        if not title or title == "[Removed]":
            return None
        if not url or url == "[Removed]":
            return None

        # description is NewsAPI's equivalent of a summary/excerpt
        description: str | None = (article_data.get("description") or "").strip() or None
        if description == "[Removed]":
            description = None

        # content contains the article body (truncated at 200 chars on free tier)
        # Use it as a richer summary only when description is absent
        content: str | None = (article_data.get("content") or "").strip() or None
        if content == "[Removed]":
            content = None

        summary: str | None = description or content

        published_at: datetime | None = _parse_iso_datetime(
            article_data.get("publishedAt")
        )

        # The 'source' object carries the publisher name and a slug id
        source_obj: dict[str, Any] = article_data.get("source") or {}
        source_name: str = source_obj.get("name") or source_obj.get("id") or "newsapi"
        # Use the NewsAPI endpoint URL as the canonical source_url
        source_url: str = f"{NEWSAPI_BASE_URL}/{source_tag.split(':')[0]}"

        return CollectedArticle(
            title=title,
            summary=summary,
            url=url,
            source_url=source_url,
            source_type="newsapi",
            language="en",  # we filter for en in all requests
            published_at=published_at,
            tickers=[],
            sectors=[],
            raw_data={
                **article_data,
                "_source_name": source_name,
                "_source_tag": source_tag,
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 UTC datetime string as returned by NewsAPI.

    NewsAPI always returns dates in the form ``2024-01-15T12:34:56Z``.
    """
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    logger.debug("Could not parse NewsAPI date string: %r", value)
    return None
