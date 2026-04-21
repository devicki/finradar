"""
finradar.collectors.youtube_collector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

YouTube community-post collector.

YouTube does not expose community posts through the Data API v3, so we fetch
``https://www.youtube.com/@{handle}/posts`` and parse the ``ytInitialData`` JSON
embedded in the HTML. Each page returns up to ~10 most recent posts, which is
more than enough for daily polling even with our smart schedule off by a couple
of hours.

Classification
--------------
Posts are tagged with a lightweight category so downstream consumers (dashboard
filters, Phase 3 alerting) can treat them differently without re-parsing:

  * ``daily_recap``   — titles containing "미국 증시 요약" / "US market recap" etc.
  * ``breaking``      — titles starting with 🚨 / ✔ / [속보] / BREAKING:
  * ``weekly_agenda`` — titles with "주간" / "weekly" / 주요 일정
  * ``general``       — everything else

The category lands in ``raw_data.youtube.category`` alongside the post_id,
handle, vote_count, and any attached image URLs.

Storage mapping
---------------
  * ``source_type``  = ``"youtube_post"``
  * ``source_url``   = ``"youtube_channel:<handle>"``
  * ``url``          = ``"https://www.youtube.com/post/<post_id>"``
  * ``title``        = first 100 chars of post content
  * ``summary``      = full post content
  * ``language``     = best-effort; we assume channel's primary language via
                       the ``default_language`` constructor arg (defaults to
                       ``"ko"`` since that's our pilot creator)
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from finradar.collectors.base import BaseCollector, CollectedArticle


logger = logging.getLogger("finradar.collectors.youtube")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Matches the embedded JSON blob YouTube ships in every channel HTML page.
# The greedy + non-greedy trick (.+?) keeps us from capturing the next assign.
_YT_INITIAL_DATA_RE = re.compile(
    r"var\s+ytInitialData\s*=\s*(\{.*?\});", re.DOTALL
)


# ---------------------------------------------------------------------------
# Category heuristics
# ---------------------------------------------------------------------------

_BREAKING_PREFIXES = ("🚨", "✔", "[속보]", "BREAKING:", "JUST IN:", "UPDATE:")
_DAILY_RECAP_MARKERS = (
    "미국 증시 요약",
    "증시 요약",
    "미국증시 요약",
    "US market recap",
    "market recap",
)
_WEEKLY_AGENDA_MARKERS = ("주간", "weekly", "주요 일정", "Weekly")


def _categorise(text: str) -> str:
    """Classify a post by its opening content."""
    first = (text or "").lstrip()[:80]
    if any(first.startswith(p) for p in _BREAKING_PREFIXES):
        return "breaking"
    if any(m in first for m in _WEEKLY_AGENDA_MARKERS):
        return "weekly_agenda"
    if any(m in first for m in _DAILY_RECAP_MARKERS):
        return "daily_recap"
    return "general"


# ---------------------------------------------------------------------------
# Relative-time text → absolute timestamp
# ---------------------------------------------------------------------------
#
# YouTube only publishes relative strings like "2시간 전" / "3 days ago". We
# rebuild an approximate timestamp by subtracting the unit × count from "now".
# This is intentionally best-effort — exact time isn't available without
# authenticated API access, and for feed ordering a rough bucket is fine.

_REL_TIME_PATTERNS_KO = {
    "분 전": "minutes",
    "시간 전": "hours",
    "일 전": "days",
    "주 전": "weeks",
    "개월 전": "months",
    "년 전": "years",
    "방금": "now",
}
_REL_TIME_PATTERNS_EN = {
    "minute": "minutes",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
    "month": "months",
    "year": "years",
}


def _relative_to_timestamp(rel_text: str, now: datetime | None = None) -> datetime | None:
    """Best-effort parse of '2시간 전', '3 days ago', '방금' into a UTC datetime."""
    if not rel_text:
        return None
    now = now or datetime.now(tz=timezone.utc)
    raw = rel_text.strip()

    if "방금" in raw or raw.startswith("just") or raw.startswith("now"):
        return now

    # KO patterns
    for marker, unit in _REL_TIME_PATTERNS_KO.items():
        if marker in raw:
            num_match = re.search(r"(\d+)", raw)
            if not num_match:
                return now
            n = int(num_match.group(1))
            return _subtract_unit(now, unit, n)

    # EN patterns (e.g. "3 days ago")
    for marker, unit in _REL_TIME_PATTERNS_EN.items():
        if marker in raw.lower():
            num_match = re.search(r"(\d+)", raw)
            if not num_match:
                return now
            n = int(num_match.group(1))
            return _subtract_unit(now, unit, n)

    return None


def _subtract_unit(now: datetime, unit: str, n: int) -> datetime:
    if unit == "minutes":
        return now - timedelta(minutes=n)
    if unit == "hours":
        return now - timedelta(hours=n)
    if unit == "days":
        return now - timedelta(days=n)
    if unit == "weeks":
        return now - timedelta(weeks=n)
    if unit == "months":
        return now - timedelta(days=30 * n)
    if unit == "years":
        return now - timedelta(days=365 * n)
    return now


# ---------------------------------------------------------------------------
# ytInitialData walker
# ---------------------------------------------------------------------------


def _collect_post_renderers(obj: Any, out: list[dict]) -> None:
    """Depth-first walk of ytInitialData collecting every backstagePostRenderer."""
    if isinstance(obj, dict):
        if "backstagePostRenderer" in obj and isinstance(obj["backstagePostRenderer"], dict):
            out.append(obj["backstagePostRenderer"])
        for v in obj.values():
            _collect_post_renderers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_post_renderers(v, out)


def _extract_text(content_text: dict | None) -> str:
    """Flatten YouTube's contentText.runs[] structure into a single string."""
    if not content_text:
        return ""
    runs = content_text.get("runs") or []
    return "".join((run.get("text") or "") for run in runs)


def _extract_image_urls(renderer: dict) -> list[str]:
    """Pull the largest thumbnail URL from every attached image, if any."""
    urls: list[str] = []
    attachment = renderer.get("backstageAttachment") or {}
    # Single image
    image_renderer = attachment.get("backstageImageRenderer")
    if image_renderer:
        thumbs = (image_renderer.get("image") or {}).get("thumbnails") or []
        if thumbs:
            urls.append(thumbs[-1].get("url", ""))
    # Multi-image
    post_multi = attachment.get("postMultiImageRenderer")
    if post_multi:
        for child in post_multi.get("images") or []:
            inner = child.get("backstageImageRenderer") or {}
            thumbs = (inner.get("image") or {}).get("thumbnails") or []
            if thumbs:
                urls.append(thumbs[-1].get("url", ""))
    return [u for u in urls if u]


def _relative_time_from_renderer(renderer: dict) -> str:
    """Pull the 'published 2 hours ago' style string."""
    ptt = renderer.get("publishedTimeText") or {}
    runs = ptt.get("runs") or []
    if runs:
        return runs[0].get("text", "") or ""
    return ptt.get("simpleText", "") or ""


def _vote_count(renderer: dict) -> int | None:
    """Best-effort parse of voteCount (accessibility 'label' has the count)."""
    v = renderer.get("voteCount") or {}
    label = (
        v.get("accessibility", {})
        .get("accessibilityData", {})
        .get("label", "")
    )
    m = re.search(r"(\d[\d,]*)", label)
    return int(m.group(1).replace(",", "")) if m else None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class YouTubePostsCollector(BaseCollector):
    """Scrape community posts from a list of YouTube channel handles.

    Parameters
    ----------
    handles:
        List of channel handles **without** the leading ``@``
        (e.g. ``["futuresnow"]``).
    default_language:
        Language code to tag each article with (YouTube doesn't expose per-post
        language in the public payload). Defaults to ``"ko"`` since the pilot
        creator publishes in Korean.
    timeout:
        Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        handles: list[str],
        *,
        default_language: str = "ko",
        timeout: float = 15.0,
    ) -> None:
        super().__init__(name="youtube", max_concurrent=2)
        self.handles = [h.strip().lstrip("@") for h in handles if h and h.strip()]
        self.default_language = default_language
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def collect(self) -> list[CollectedArticle]:
        if not self.handles:
            logger.debug("YouTubePostsCollector: no handles configured")
            return []

        all_articles: list[CollectedArticle] = []
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
        ) as client:
            for handle in self.handles:
                try:
                    articles = await self._fetch_handle(client, handle)
                    all_articles.extend(articles)
                except Exception as exc:  # noqa: BLE001 — any fetch/parse error is per-channel
                    logger.error(
                        "YouTubePostsCollector: @%s failed: %s", handle, exc, exc_info=True,
                    )
        logger.info(
            "YouTubePostsCollector: collected %d posts across %d handles",
            len(all_articles), len(self.handles),
        )
        return all_articles

    # ------------------------------------------------------------------
    # Per-handle flow
    # ------------------------------------------------------------------

    async def _fetch_handle(
        self, client: httpx.AsyncClient, handle: str,
    ) -> list[CollectedArticle]:
        url = f"https://www.youtube.com/@{handle}/posts"
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

        m = _YT_INITIAL_DATA_RE.search(html)
        if not m:
            logger.warning("YouTubePostsCollector: ytInitialData not found on @%s", handle)
            return []

        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError as exc:
            logger.warning(
                "YouTubePostsCollector: JSON decode failed for @%s: %s", handle, exc,
            )
            return []

        renderers: list[dict] = []
        _collect_post_renderers(data, renderers)

        now_utc = datetime.now(tz=timezone.utc)
        articles: list[CollectedArticle] = []

        for r in renderers:
            post_id = r.get("postId")
            if not post_id:
                continue
            text = _extract_text(r.get("contentText"))
            if not text.strip():
                continue

            category = _categorise(text)
            rel_time = _relative_time_from_renderer(r)
            published_at = _relative_to_timestamp(rel_time, now=now_utc)
            image_urls = _extract_image_urls(r)
            votes = _vote_count(r)

            title = text.strip().split("\n", 1)[0][:150]
            if len(title) < 20 and len(text) > 20:
                title = text.strip()[:150]

            articles.append(
                CollectedArticle(
                    title=title,
                    url=f"https://www.youtube.com/post/{post_id}",
                    source_url=f"youtube_channel:{handle}",
                    source_type="youtube_post",
                    language=self.default_language,
                    summary=text,
                    published_at=published_at,
                    tickers=[],
                    sectors=[],
                    raw_data={
                        "youtube": {
                            "post_id": post_id,
                            "handle": handle,
                            "category": category,
                            "published_time_text": rel_time,
                            "image_urls": image_urls,
                            "vote_count": votes,
                        }
                    },
                )
            )

        logger.info(
            "YouTubePostsCollector: @%s → %d posts parsed", handle, len(articles),
        )
        return articles
