"""
finradar.alerts.dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~~

Trigger detection, throttling, and Discord dispatch for breaking-news alerts.

Triggers (OR — any one fires an alert)
--------------------------------------
1. ``breaking``          — ``raw_data.x.breaking == True`` or
                           ``raw_data.youtube.category == 'breaking'``.
                           Explicit publisher signal — no extra quality gates.
2. ``strong_sentiment``  — ``|sentiment| ≥ settings.alerts_min_abs_sentiment`` AND
                           ``alerts_min_cluster_size ≤ cluster_size <
                           alerts_max_cluster_size`` AND (when
                           ``alerts_require_sectors=True``) the article has
                           at least one LLM-extracted sector. The cluster
                           upper bound suppresses Korean-news template chains
                           (Phase 2 known issue where clusters of 200+
                           unrelated rows form on headline style alone), and
                           the sectors requirement filters out celeb / social
                           news that still scores strongly on FinBERT.
3. ``ticker_watch``      — any ticker on the article matches
                           ``settings.alerts_tickers_watch`` AND
                           ``|sentiment| ≥ 0.3`` (skip fully neutral mentions).
                           Explicit watchlist hit — bypasses the cluster /
                           sectors gates.

Throttling
----------
* Per-article dedup via ``finradar:alerts:sent:{news_id}`` Redis key
  (30-day TTL) — an article is alerted at most once across all runs.
* Per-cluster dedup via ``finradar:alerts:cluster:{cluster_rep_id}``
  (24-hour TTL) — "Fed raises rates" from Reuters + CNBC + Bloomberg
  collapses to one alert.
* Per-hour cap via ``finradar:alerts:hourly:{YYYY-MM-DD-HH}`` — protects
  against trigger misconfig / sudden news rushes.

The task is idempotent by construction: failing to POST leaves the
Redis marker unset, so the next run will retry the article.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, or_, select

from finradar.config import get_settings
from finradar.models import NewsItem


logger = logging.getLogger("finradar.alerts.dispatcher")


# Redis keys + TTLs ----------------------------------------------------------
_ALERT_SENT_KEY_FMT = "finradar:alerts:sent:{news_id}"
_ALERT_CLUSTER_KEY_FMT = "finradar:alerts:cluster:{cluster_rep_id}"
_ALERT_HOURLY_KEY_FMT = "finradar:alerts:hourly:{hour_bucket}"

_SENT_TTL_SEC = 30 * 24 * 60 * 60  # 30 days
_CLUSTER_TTL_SEC = 24 * 60 * 60    # 24 hours
_HOURLY_TTL_SEC = 3700             # 1 hour + small slack


# ---------------------------------------------------------------------------
# Per-article evaluation
# ---------------------------------------------------------------------------


@dataclass
class AlertTrigger:
    """Outcome of :py:func:`evaluate_trigger` for one article."""

    news_id: int
    should_alert: bool
    reasons: list[str] = field(default_factory=list)


def _parse_watchlist(raw: str) -> set[str]:
    """Comma-separated tickers → normalised set."""
    return {t.strip().upper() for t in (raw or "").split(",") if t.strip()}


def evaluate_trigger(article: NewsItem) -> AlertTrigger:
    """Decide whether *article* deserves an alert and why.

    Returns an :py:class:`AlertTrigger` with the list of reasons so the
    Discord embed can surface them. An article may satisfy multiple
    triggers (e.g. BREAKING + ticker watch) — we keep all reasons for
    transparency.
    """
    settings = get_settings()
    reasons: list[str] = []

    # 1. BREAKING flag
    raw = article.raw_data or {}
    x_meta = raw.get("x") or {}
    yt_meta = raw.get("youtube") or {}
    if x_meta.get("breaking") is True or yt_meta.get("category") == "breaking":
        reasons.append("breaking")

    # 2. Strong sentiment + cluster size + sectors presence.
    # Language-specific cluster lower bound: English RSS produces smaller
    # real-story clusters than Korean (different styling by each outlet vs.
    # template-heavy KR press). Use the per-language floor so EN articles
    # aren't systematically filtered out.
    #
    # Dual-signal gate (migrate_006): when the LLM also produced a sentiment
    # score during enrichment, require BOTH the local sentiment model
    # (FinBERT for EN, KR-FinBert-SC for KO) and the LLM to exceed the
    # magnitude threshold AND agree on sign. Prevents keyword-only misfires
    # from the local model — e.g. "war fuels trading boom" tagged negative
    # by FinBERT, or Korean awards/crime headlines over-scoring on
    # KR-FinBert-SC. When llm_sentiment IS NULL (pre-migration rows / enrich
    # not yet run) we fall back to local-only so existing candidates don't
    # silently disappear.
    sentiment = article.sentiment
    llm_sentiment = article.llm_sentiment
    cluster_size = article.cluster_size or 1
    has_sectors = bool(article.sectors)
    min_cluster = (
        settings.alerts_min_cluster_size_en
        if article.language == "en"
        else settings.alerts_min_cluster_size
    )

    local_strong = (
        sentiment is not None
        and abs(sentiment) >= settings.alerts_min_abs_sentiment
    )
    if llm_sentiment is None:
        # Fallback path: LLM didn't emit a score (legacy row or enrich failure).
        sentiment_ok = local_strong
    else:
        llm_strong = abs(llm_sentiment) >= settings.alerts_min_abs_sentiment
        signs_agree = (
            sentiment is not None
            and (sentiment >= 0) == (llm_sentiment >= 0)
        )
        dual_agree = local_strong and llm_strong and signs_agree
        # LLM rescue: the local model sometimes under-scores market-impactful
        # articles (see notes on enrich filter coverage). When the LLM alone
        # is very confident — stricter than the dual-gate threshold to
        # compensate for the missing cross-model check — we still fire the
        # trigger. Cluster gates and the sectors requirement below continue
        # to apply; the rescue only relaxes the local-strength requirement.
        llm_rescue = abs(llm_sentiment) >= settings.alerts_llm_rescue_threshold
        sentiment_ok = dual_agree or llm_rescue

    if (
        sentiment_ok
        and min_cluster <= cluster_size < settings.alerts_max_cluster_size
        and (has_sectors or not settings.alerts_require_sectors)
    ):
        reasons.append("strong_sentiment")

    # 3. Ticker watchlist
    watch = _parse_watchlist(settings.alerts_tickers_watch)
    article_tickers = {(t or "").upper() for t in (article.tickers or []) if t}
    if (
        watch
        and article_tickers & watch
        and sentiment is not None
        and abs(sentiment) >= 0.3  # skip fully neutral mentions
    ):
        reasons.append("ticker_watch")

    return AlertTrigger(
        news_id=article.id,
        should_alert=bool(reasons),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def _redis():
    """Lazy import + connection so the module loads without redis in dev."""
    import redis  # noqa: PLC0415

    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _hour_bucket() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H")


def _already_sent(r, news_id: int) -> bool:
    return bool(r.exists(_ALERT_SENT_KEY_FMT.format(news_id=news_id)))


def _mark_sent(r, news_id: int) -> None:
    r.setex(_ALERT_SENT_KEY_FMT.format(news_id=news_id), _SENT_TTL_SEC, "1")


def _cluster_already_alerted(r, cluster_rep_id: int | None) -> bool:
    if not cluster_rep_id:
        return False
    return bool(r.exists(_ALERT_CLUSTER_KEY_FMT.format(cluster_rep_id=cluster_rep_id)))


def _mark_cluster(r, cluster_rep_id: int | None) -> None:
    if not cluster_rep_id:
        return
    r.setex(
        _ALERT_CLUSTER_KEY_FMT.format(cluster_rep_id=cluster_rep_id),
        _CLUSTER_TTL_SEC,
        "1",
    )


def _increment_hourly(r) -> int:
    key = _ALERT_HOURLY_KEY_FMT.format(hour_bucket=_hour_bucket())
    new_val = r.incr(key)
    r.expire(key, _HOURLY_TTL_SEC)
    return int(new_val)


def _peek_hourly(r) -> int:
    key = _ALERT_HOURLY_KEY_FMT.format(hour_bucket=_hour_bucket())
    val = r.get(key)
    try:
        return int(val) if val is not None else 0
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Candidate pull
# ---------------------------------------------------------------------------


def _pull_candidates(session, lookback_min: int) -> list[NewsItem]:
    """Articles the pipeline has fully processed in the last N minutes.

    We require sentiment to be populated because two of the three triggers
    depend on it. Rows with no sentiment yet are simply picked up on the
    next run (alerts_lookback_min buffer).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_min)
    stmt = (
        select(NewsItem)
        .where(
            NewsItem.first_seen_at >= cutoff,
            NewsItem.sentiment.is_not(None),
        )
        .order_by(NewsItem.first_seen_at.desc())
    )
    return session.execute(stmt).scalars().all()


# ---------------------------------------------------------------------------
# Article → payload dict (JSON-friendly copy for the Discord builder)
# ---------------------------------------------------------------------------


def _article_to_payload(article: NewsItem) -> dict[str, Any]:
    return {
        "id": article.id,
        "title": article.title,
        "summary": article.summary,
        "ai_summary": article.ai_summary,
        # Korean translation fields — populated by the LLM enrich step for
        # non-Korean articles. Earlier version of this payload silently
        # dropped them, so the Discord embed fell back to the English AI
        # summary only. Including them lets build_embed() render the
        # Korean-first description just like the dashboard cards.
        "translated_title": article.translated_title,
        "translated_summary": article.translated_summary,
        "url": article.url,
        "source_type": article.source_type,
        "language": article.language,
        "sentiment": float(article.sentiment) if article.sentiment is not None else None,
        "sentiment_label": article.sentiment_label,
        "tickers": list(article.tickers or []),
        "sectors": list(article.sectors or []),
        "published_at": article.published_at,
        "last_seen_at": article.last_seen_at,
    }


# ---------------------------------------------------------------------------
# Main dispatch entry point
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    scanned: int
    candidates: int
    sent: int
    skipped_already_sent: int
    skipped_cluster_dedup: int
    skipped_hourly_cap: int
    send_failures: int


def dispatch_pending_alerts(session) -> DispatchResult:
    """Scan recent articles, evaluate triggers, and fan out Discord alerts."""
    from finradar.alerts.discord import post_alert  # noqa: PLC0415

    settings = get_settings()

    if not settings.alerts_enabled:
        logger.debug("dispatch_pending_alerts: alerts_enabled=false, skipping")
        return DispatchResult(0, 0, 0, 0, 0, 0, 0)

    r = _redis()

    articles = _pull_candidates(session, settings.alerts_lookback_min)
    scanned = len(articles)

    candidates: list[tuple[NewsItem, AlertTrigger]] = []
    for art in articles:
        t = evaluate_trigger(art)
        if t.should_alert:
            candidates.append((art, t))

    sent = 0
    skipped_sent = 0
    skipped_cluster = 0
    skipped_cap = 0
    failures = 0

    for art, trigger in candidates:
        if _already_sent(r, art.id):
            skipped_sent += 1
            continue
        if _cluster_already_alerted(r, art.cluster_rep_id):
            # Mark the individual article too so we don't re-check its
            # cluster every cycle for 30 days.
            _mark_sent(r, art.id)
            skipped_cluster += 1
            continue
        if _peek_hourly(r) >= settings.alerts_hourly_cap:
            skipped_cap += 1
            continue

        # --- send ---------------------------------------------------------
        dest_ok = True
        if settings.discord_enabled and settings.discord_webhook_url:
            ok = post_alert(
                settings.discord_webhook_url,
                article=_article_to_payload(art),
                triggers=trigger.reasons,
            )
            dest_ok = ok
        else:
            # No active destinations — treat as dry-run success so the
            # per-article marker still goes in and logs explain why.
            logger.info(
                "dispatch_pending_alerts: no active destinations (news_id=%d)", art.id
            )
            dest_ok = True

        if not dest_ok:
            failures += 1
            continue

        _mark_sent(r, art.id)
        _mark_cluster(r, art.cluster_rep_id)
        _increment_hourly(r)
        sent += 1
        logger.info(
            "alerts: sent news_id=%d triggers=%s", art.id, trigger.reasons
        )

    logger.info(
        "dispatch_pending_alerts: scanned=%d candidates=%d sent=%d "
        "skip_sent=%d skip_cluster=%d skip_cap=%d fail=%d",
        scanned, len(candidates), sent,
        skipped_sent, skipped_cluster, skipped_cap, failures,
    )
    return DispatchResult(
        scanned=scanned,
        candidates=len(candidates),
        sent=sent,
        skipped_already_sent=skipped_sent,
        skipped_cluster_dedup=skipped_cluster,
        skipped_hourly_cap=skipped_cap,
        send_failures=failures,
    )
