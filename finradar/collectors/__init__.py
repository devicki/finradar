"""FinRadar news collectors package.

Exports the common article dataclass and all concrete collector
implementations so they can be imported from a single location::

    from finradar.collectors import (
        CollectedArticle,
        BaseCollector,
        RSSCollector,
        NewsAPICollector,
        DEFAULT_RSS_FEEDS,
    )
"""

from finradar.collectors.base import BaseCollector, CollectedArticle
from finradar.collectors.newsapi_collector import NewsAPICollector
from finradar.collectors.rss_collector import DEFAULT_RSS_FEEDS, RSSCollector

__all__ = [
    "CollectedArticle",
    "BaseCollector",
    "RSSCollector",
    "NewsAPICollector",
    "DEFAULT_RSS_FEEDS",
]
