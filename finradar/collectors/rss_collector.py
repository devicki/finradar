"""RSS feed collector using feedparser + httpx.

Fetches a configurable list of financial RSS feeds concurrently, parses them
with feedparser (which handles the wide variety of date formats used by
different publishers), and maps each entry to a :py:class:`CollectedArticle`.

Usage::

    async with RSSCollector() as collector:
        articles = await collector.safe_collect()
"""

import asyncio
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from finradar.collectors.base import BaseCollector, CollectedArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default feed list
# ---------------------------------------------------------------------------

DEFAULT_RSS_FEEDS: list[dict[str, str]] = [
    # --- Global (English) ---
    {
        "url": "https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com",
        "name": "Reuters via Google News",
        "language": "en",
    },
    {
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "name": "CNBC Top News",
        "language": "en",
    },
    {
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "name": "Bloomberg Markets",
        "language": "en",
    },
    {
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "name": "Yahoo Finance",
        "language": "en",
    },
    {
        "url": "https://www.ft.com/?format=rss",
        "name": "Financial Times",
        "language": "en",
    },
    # --- Korean ---
    {
        "url": "https://www.hankyung.com/feed/economy",
        "name": "한국경제 경제",
        "language": "ko",
    },
    {
        "url": "https://www.hankyung.com/feed/finance",
        "name": "한국경제 증권",
        "language": "ko",
    },
    {
        "url": "https://www.hankyung.com/feed/international",
        "name": "한국경제 국제",
        "language": "ko",
    },
    {
        "url": "https://www.mk.co.kr/rss/30000001/",
        "name": "매일경제 헤드라인",
        "language": "ko",
    },
    {
        "url": "https://www.mk.co.kr/rss/50200011/",
        "name": "매일경제 증권",
        "language": "ko",
    },
    {
        "url": "http://rss.edaily.co.kr/edaily_news.xml",
        "name": "이데일리 전체",
        "language": "ko",
    },
    {
        "url": "http://rss.mt.co.kr/mt_news.xml",
        "name": "머니투데이 전체",
        "language": "ko",
    },
    {
        "url": "https://www.yonhapnewseconomytv.com/rss/allArticle.xml",
        "name": "연합뉴스경제TV 전체",
        "language": "ko",
    },
    {
        "url": "https://www.sedaily.com/rss/Economy",
        "name": "서울경제 경제",
        "language": "ko",
    },
    {
        "url": "https://rss.etnews.com/Section901.xml",
        "name": "전자신문 전체",
        "language": "ko",
    },
]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class RSSCollector(BaseCollector):
    """Collect news articles from multiple RSS/Atom feeds concurrently.

    Parameters
    ----------
    feeds:
        List of feed configuration dicts.  Each dict must contain ``url``,
        ``name``, and ``language`` keys.  Defaults to
        :py:data:`DEFAULT_RSS_FEEDS`.
    max_concurrent:
        Maximum number of feed fetch requests that may run simultaneously.
    """

    def __init__(
        self,
        feeds: list[dict[str, str]] | None = None,
        max_concurrent: int = 5,
    ) -> None:
        super().__init__(name="rss", max_concurrent=max_concurrent)
        self.feeds = feeds if feeds is not None else DEFAULT_RSS_FEEDS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def collect(self) -> list[CollectedArticle]:
        """Fetch all configured feeds concurrently and return merged articles."""
        tasks = [self._fetch_feed(feed) for feed in self.feeds]
        results: list[list[CollectedArticle]] = await asyncio.gather(
            *tasks, return_exceptions=False
        )
        all_articles: list[CollectedArticle] = []
        for batch in results:
            all_articles.extend(batch)
        return all_articles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_feed(self, feed_config: dict[str, str]) -> list[CollectedArticle]:
        """Fetch a single RSS feed and parse its entries.

        The raw feed XML is downloaded via httpx so we can apply our shared
        client (with a consistent User-Agent and timeout).  feedparser then
        parses the byte content directly.
        """
        url: str = feed_config["url"]
        async with self.semaphore:
            try:
                client = await self.get_client()
                response = await client.get(url)
                response.raise_for_status()
                raw_bytes: bytes = response.content
            except httpx.HTTPStatusError as exc:
                self.logger.warning(
                    "HTTP %s while fetching feed %s: %s",
                    exc.response.status_code,
                    feed_config.get("name", url),
                    exc,
                )
                return []
            except httpx.RequestError as exc:
                self.logger.warning(
                    "Request error fetching feed %s: %s",
                    feed_config.get("name", url),
                    exc,
                )
                return []

        # feedparser can parse bytes directly; pass the URL as 'response_headers'
        # source so relative links in the feed can be resolved.
        parsed = feedparser.parse(raw_bytes, response_headers={"content-location": url})

        if parsed.bozo and parsed.bozo_exception:
            # feedparser sets bozo=True for malformed feeds but still attempts
            # to parse them.  Log a warning but keep going.
            self.logger.warning(
                "Malformed feed at %s (%s); attempting partial parse",
                feed_config.get("name", url),
                type(parsed.bozo_exception).__name__,
            )

        articles: list[CollectedArticle] = []
        for entry in parsed.entries:
            article = self._parse_entry(entry, feed_config)
            if article is not None:
                articles.append(article)

        self.logger.debug(
            "Parsed %d articles from %s", len(articles), feed_config.get("name", url)
        )
        return articles

    def _parse_entry(
        self, entry: Any, feed_config: dict[str, str]
    ) -> CollectedArticle | None:
        """Convert a feedparser entry dict to a :py:class:`CollectedArticle`.

        Returns ``None`` if the entry lacks a title or a URL (both are
        required for deduplication downstream).
        """
        # ---- Required fields --------------------------------------------------
        title: str | None = _get_text(entry, "title")
        if not title:
            return None

        url: str | None = _get_link(entry)
        if not url:
            return None

        # ---- Optional fields --------------------------------------------------
        summary: str | None = _get_text(entry, "summary") or _get_text(entry, "description")

        published_at: datetime | None = _parse_published(entry)

        # feedparser sometimes exposes ticker-like tags in the categories list
        tickers: list[str] = _extract_tickers(entry)

        # ---- Build article ----------------------------------------------------
        return CollectedArticle(
            title=title.strip(),
            summary=summary.strip() if summary else None,
            url=url.strip(),
            source_url=feed_config["url"],
            source_type="rss",
            language=feed_config.get("language", "en"),
            published_at=published_at,
            tickers=tickers,
            sectors=[],
            raw_data=dict(entry),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_text(entry: Any, key: str) -> str | None:
    """Safely extract a text field from a feedparser entry."""
    value = getattr(entry, key, None)
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    # feedparser sometimes returns a list of Detail objects for 'summary'
    if isinstance(value, list) and value:
        first = value[0]
        if hasattr(first, "value"):
            return str(first.value) or None
        return str(first) or None
    return str(value) or None


def _get_link(entry: Any) -> str | None:
    """Return the canonical article URL from a feedparser entry."""
    # Prefer 'link', fall back to the first href in 'links'
    link: str | None = getattr(entry, "link", None)
    if link:
        return link
    links = getattr(entry, "links", [])
    for lnk in links:
        href = getattr(lnk, "href", None)
        if href:
            return href
    return None


def _parse_published(entry: Any) -> datetime | None:
    """Parse a publish timestamp from a feedparser entry.

    feedparser provides ``published_parsed`` (a time.struct_time in UTC) when
    the feed date is well-formed.  We fall back to a manual parse of the raw
    ``published`` string for less common formats.
    """
    # Fast path: feedparser already parsed it
    struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if struct is not None:
        try:
            return datetime(*struct[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass

    # Slow path: try to parse the raw string ourselves
    raw: str | None = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not raw:
        return None

    # RFC 2822 (most common in RSS)
    try:
        return parsedate_to_datetime(raw)
    except Exception:  # noqa: BLE001
        pass

    # ISO 8601 / Atom format
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    logger.debug("Could not parse date string: %r", raw)
    return None


def _extract_tickers(entry: Any) -> list[str]:
    """Heuristically extract stock tickers from feedparser entry tags/categories.

    Some financial feeds (e.g. Yahoo Finance) embed tickers in the category
    list as strings like ``$AAPL`` or plain ``AAPL``.
    """
    tickers: list[str] = []
    tags = getattr(entry, "tags", []) or []
    for tag in tags:
        term: str = getattr(tag, "term", "") or ""
        term = term.strip().lstrip("$")
        # Very rough heuristic: 1-5 uppercase letters is likely a ticker
        if term.isupper() and 1 <= len(term) <= 5 and term.isalpha():
            tickers.append(term)
    return tickers
