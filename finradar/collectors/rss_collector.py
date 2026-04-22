"""RSS feed collector using feedparser + httpx.

Fetches a configurable list of financial RSS feeds concurrently, parses them
with feedparser (which handles the wide variety of date formats used by
different publishers), and maps each entry to a :py:class:`CollectedArticle`.

Two post-parse steps normalise messy real-world feeds:

1. :py:func:`clean_rss_text` strips HTML tags, decodes HTML entities, and
   normalises whitespace on both titles and summaries. Applied universally
   (EN + KO) since multiple feeds ship HTML-polluted payloads.

2. A body-fetch fallback uses ``trafilatura`` to extract full article text
   when the RSS summary is too short to be useful (common with 한국경제,
   서울경제 which publish title-only feeds).

Usage::

    async with RSSCollector() as collector:
        articles = await collector.safe_collect()
"""

import asyncio
import html
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx
import trafilatura

from finradar.collectors.base import BaseCollector, CollectedArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text-cleaning constants
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# 한경(hankyung.com) 기사 본문에 반복 등장하는 UI 버튼 텍스트
_HANKYUNG_UI_NOISE = re.compile(
    r"\s*-\s*기사\s*스크랩\s*-\s*공유\s*-\s*댓글\s*-\s*클린뷰\s*-\s*프린트\s*"
)

# Summary below this length triggers trafilatura body-fetch fallback
_MIN_SUMMARY_LEN: int = 50

# Browser-like UA for article page fetches (some sites block default UAs)
_BODY_FETCH_UA: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_BODY_FETCH_TIMEOUT: float = 15.0

# Max concurrent article-page fetches for the body enrichment pass
_BODY_FETCH_CONCURRENCY: int = 10


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
    # --- Tier S: 고품질 + 신선도 높음 (Phase 3에서 추가) ---
    {
        "url": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
        "name": "WSJ Markets",
        "language": "en",
    },
    {
        "url": "http://feeds.marketwatch.com/marketwatch/topstories/",
        "name": "MarketWatch Top Stories",
        "language": "en",
    },
    {
        "url": "https://feeds.bloomberg.com/technology/news.rss",
        "name": "Bloomberg Technology",
        "language": "en",
    },
    {
        "url": "https://feeds.bloomberg.com/politics/news.rss",
        "name": "Bloomberg Politics",
        "language": "en",
    },
    {
        "url": "https://www.ft.com/companies?format=rss",
        "name": "FT Companies",
        "language": "en",
    },
    {
        "url": "https://www.ft.com/markets?format=rss",
        "name": "FT Markets",
        "language": "en",
    },
    {
        "url": "https://www.ft.com/global-economy?format=rss",
        "name": "FT Global Economy",
        "language": "en",
    },
    {
        # Reuters dropped public RSS; Google News site-restricted query is
        # the closest legitimate substitute for Reuters business news.
        "url": "https://news.google.com/rss/search?q=when:24h+site:reuters.com+business&hl=en",
        "name": "Reuters Business via Google News",
        "language": "en",
    },
    {
        "url": "https://markets.businessinsider.com/rss/news",
        "name": "Business Insider Markets",
        "language": "en",
    },
    {
        "url": "https://www.businessinsider.com/rss",
        "name": "Business Insider",
        "language": "en",
    },
    {
        "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "name": "CNBC Finance",
        "language": "en",
    },
    # --- Tier A: 보조 소스 ---
    {
        # Seeking Alpha ships empty <description> tags — rss_collector's
        # body-fetch fallback pulls the full article body via trafilatura.
        "url": "https://seekingalpha.com/feed.xml",
        "name": "Seeking Alpha",
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
# Text-cleaning helper (module-level so it can be reused / unit-tested)
# ---------------------------------------------------------------------------


def clean_rss_text(text: str | None, source_hint: str = "") -> str:
    """Strip HTML tags/entities and normalise whitespace from RSS text.

    Applied universally to titles and summaries from every feed (EN + KO).
    Many publishers ship HTML-laden payloads (``<table>``, ``<img>``,
    ``&amp;``, ``&#039;`` …) which waste LLM tokens and degrade summary quality.

    Args:
        text:        Raw text pulled from a feedparser entry.
        source_hint: URL of the feed or article; used to trigger source-specific
                     cleanup rules (e.g. 한경 UI button text).

    Returns:
        Cleaned text with all HTML stripped, entities decoded, and collapsed
        whitespace.  Returns an empty string for ``None`` / empty input.
    """
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = html.unescape(cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    if source_hint and "hankyung.com" in source_hint:
        cleaned = _HANKYUNG_UI_NOISE.sub(" ", cleaned)
        cleaned = _WS_RE.sub(" ", cleaned).strip()

    return cleaned


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
        # Body-fetch uses its own semaphore so it doesn't contend with the
        # feed-fetch semaphore and can run with higher parallelism.
        self._body_semaphore = asyncio.Semaphore(_BODY_FETCH_CONCURRENCY)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def collect(self) -> list[CollectedArticle]:
        """Fetch all configured feeds concurrently and return merged articles.

        After parsing all feeds, any article whose summary is shorter than
        :py:data:`_MIN_SUMMARY_LEN` has its body fetched via trafilatura as a
        fallback.  This rescues sources like 한국경제 / 서울경제 whose RSS
        feeds ship only titles.
        """
        tasks = [self._fetch_feed(feed) for feed in self.feeds]
        results: list[list[CollectedArticle]] = await asyncio.gather(
            *tasks, return_exceptions=False
        )
        all_articles: list[CollectedArticle] = [a for batch in results for a in batch]

        # Body-fetch fallback: enrich articles with thin RSS summaries.
        needs_body = [
            a for a in all_articles if len(a.summary or "") < _MIN_SUMMARY_LEN
        ]
        if needs_body:
            self.logger.info(
                "Body-fetch fallback: %d / %d articles need full-body extraction",
                len(needs_body),
                len(all_articles),
            )
            await self._enrich_bodies(needs_body)

        return all_articles

    # ------------------------------------------------------------------
    # Internal helpers — feed-level
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
        source_hint = feed_config["url"]

        # ---- Required fields --------------------------------------------------
        title = clean_rss_text(_get_text(entry, "title"), source_hint=source_hint)
        if not title:
            return None

        url: str | None = _get_link(entry)
        if not url:
            return None

        # ---- Optional fields --------------------------------------------------
        summary_raw = _get_text(entry, "summary") or _get_text(entry, "description")
        summary = clean_rss_text(summary_raw, source_hint=source_hint)

        published_at: datetime | None = _parse_published(entry)

        # feedparser sometimes exposes ticker-like tags in the categories list
        tickers: list[str] = _extract_tickers(entry)

        # ---- Build article ----------------------------------------------------
        return CollectedArticle(
            title=title,
            summary=summary or None,
            url=url.strip(),
            source_url=feed_config["url"],
            source_type="rss",
            language=feed_config.get("language", "en"),
            published_at=published_at,
            tickers=tickers,
            sectors=[],
            raw_data=dict(entry),
        )

    # ------------------------------------------------------------------
    # Internal helpers — body-fetch fallback
    # ------------------------------------------------------------------

    async def _enrich_bodies(self, articles: list[CollectedArticle]) -> None:
        """Concurrently fetch article bodies via trafilatura for short-summary items.

        Updates ``article.summary`` in place on success. Failures are swallowed
        so a single broken URL does not block the rest of the collection pass.
        """
        await asyncio.gather(
            *(self._fetch_body(a) for a in articles),
            return_exceptions=True,
        )

    async def _fetch_body(self, article: CollectedArticle) -> None:
        """Fetch article URL, extract body with trafilatura, update summary.

        Silently logs (at debug) and returns on any error so the main pipeline
        keeps the original (short) summary as a fallback.
        """
        async with self._body_semaphore:
            try:
                client = await self.get_client()
                response = await client.get(
                    article.url,
                    timeout=_BODY_FETCH_TIMEOUT,
                    headers={"User-Agent": _BODY_FETCH_UA},
                    follow_redirects=True,
                )
                response.raise_for_status()
                html_text: str = response.text
            except (httpx.HTTPError, httpx.RequestError) as exc:
                self.logger.debug(
                    "Body-fetch HTTP error for %s: %s", article.url, exc
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.logger.debug(
                    "Body-fetch unexpected error for %s: %s", article.url, exc
                )
                return

        # trafilatura is C-backed lxml; ~10ms per article. Fine to run on the
        # event loop thread without offloading.
        try:
            extracted = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug(
                "Trafilatura extraction failed for %s: %s", article.url, exc
            )
            return

        if not extracted:
            return

        cleaned = clean_rss_text(extracted, source_hint=article.url)
        if len(cleaned) >= _MIN_SUMMARY_LEN:
            article.summary = cleaned


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
