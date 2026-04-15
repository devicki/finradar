"""Abstract base class and shared data structures for all news collectors."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import httpx


@dataclass
class CollectedArticle:
    """Common structure for all collected news articles.

    Every collector produces a list of these, regardless of the underlying
    data source.  Downstream processing (sentiment analysis, embedding,
    translation) operates on this common representation.
    """

    title: str
    url: str
    source_url: str  # The feed/API endpoint this came from
    source_type: str  # "rss", "newsapi", "polygon"
    language: str  # "en", "ko", etc.
    summary: str | None = None
    published_at: datetime | None = None
    tickers: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)  # Original response payload

    def __post_init__(self) -> None:
        # Normalise tickers to uppercase so downstream code can rely on it
        self.tickers = [t.upper().strip() for t in self.tickers if t and t.strip()]


class BaseCollector(ABC):
    """Abstract base class for all news collectors.

    Subclasses must implement :py:meth:`collect`.  The :py:meth:`safe_collect`
    wrapper adds logging and exception handling so a single failing collector
    never blocks the rest of the pipeline.

    The shared :py:class:`httpx.AsyncClient` is created lazily on first use
    and reused across calls within the same collector lifetime.
    An :py:class:`asyncio.Semaphore` controls concurrency when a collector
    fans out to multiple URLs.
    """

    def __init__(self, name: str, max_concurrent: int = 5) -> None:
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(f"finradar.collectors.{name}")
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        """Return (or create) the shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
                headers={"User-Agent": "FinRadar/0.1.0"},
            )
        return self._client

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "BaseCollector":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @abstractmethod
    async def collect(self) -> list[CollectedArticle]:
        """Collect articles from this source.

        Returns a (possibly empty) list of :py:class:`CollectedArticle`
        instances.  Implementations should raise on unrecoverable errors and
        let :py:meth:`safe_collect` swallow them.
        """
        ...

    async def safe_collect(self) -> list[CollectedArticle]:
        """Run :py:meth:`collect` with error handling and logging.

        Always returns a list — never raises.  Use this in production pipelines
        so that a broken feed doesn't halt collection from healthy sources.
        """
        try:
            articles = await self.collect()
            self.logger.info("Collected %d articles from %s", len(articles), self.name)
            return articles
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error collecting from %s: %s", self.name, exc, exc_info=True)
            return []
