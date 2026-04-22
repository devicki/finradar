"""
finradar.alerts.discord
~~~~~~~~~~~~~~~~~~~~~~~

Discord webhook client for breaking-news alerts.

The webhook URL embeds a channel-level secret, so treat settings
``discord_webhook_url`` like any other API credential: read it once,
never log its value.

Each alert becomes a single Discord embed with:
  * colored strip matching sentiment (green / red / gray)
  * trigger badge in the title (🚨 breaking / 💢 strong signal / 💹 ticker)
  * source + language + publication time in the embed footer
  * tickers / sectors as inline fields
  * AI summary as the description when available
  * "원문 보기" / "FinRadar에서 상세" action-style links as the final field
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx


logger = logging.getLogger("finradar.alerts.discord")


# Sentiment → Discord embed color (integer form Discord expects)
_COLOR_POSITIVE = 0x22C55E  # emerald-500
_COLOR_NEGATIVE = 0xEF4444  # red-500
_COLOR_NEUTRAL = 0x6B7280   # gray-500

_POST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def _color_for_sentiment(label: str | None) -> int:
    if label == "positive":
        return _COLOR_POSITIVE
    if label == "negative":
        return _COLOR_NEGATIVE
    return _COLOR_NEUTRAL


def _format_trigger_badge(triggers: list[str]) -> str:
    """Pick a single emoji for the embed title based on the most important trigger."""
    if "breaking" in triggers:
        return "🚨 BREAKING"
    if "strong_sentiment" in triggers:
        return "💢 강한 신호"
    if "ticker_watch" in triggers:
        return "💹 관심 티커"
    return "📰"


def _truncate(text: str | None, n: int) -> str:
    if not text:
        return ""
    return text if len(text) <= n else (text[: n - 1] + "…")


def build_embed(
    *,
    article: dict[str, Any],
    triggers: list[str],
    finradar_url: str | None = None,
) -> dict[str, Any]:
    """Build the Discord embed dict for a single article.

    Args:
        article: dict representation of a NewsItem (id, title, url, summary,
                 ai_summary, sentiment, sentiment_label, tickers, sectors,
                 source_type, language, published_at, last_seen_at, ...).
        triggers: list of reason strings returned by the dispatcher.
        finradar_url: optional internal URL (Streamlit Article page).
    """
    title_text = article.get("title") or "(제목 없음)"
    title = _format_trigger_badge(triggers) + " · " + _truncate(title_text, 200)

    language = (article.get("language") or "").lower()
    ai_summary = (article.get("ai_summary") or "").strip()
    translated_title = (article.get("translated_title") or "").strip()
    translated_summary = (article.get("translated_summary") or "").strip()
    original_summary = (article.get("summary") or "").strip()

    # Korean-first description: mirror the dashboard card template so the
    # Discord feed scans the same way. EN articles get the translated
    # Korean summary (with translated_title bolded on top when different
    # from the English headline); KO articles get their native ai_summary.
    description_parts: list[str] = []
    if language == "en" and translated_title and translated_title != title_text:
        description_parts.append(f"**🌐 {translated_title}**")

    if language == "en" and translated_summary:
        description_parts.append(translated_summary)
    elif ai_summary:
        description_parts.append(ai_summary)
    elif original_summary:
        description_parts.append(original_summary)

    description = _truncate("\n\n".join(description_parts), 1200)

    fields: list[dict[str, Any]] = []

    # For EN articles, keep the English AI summary available as a field so
    # readers can compare the original phrasing without leaving Discord.
    if language == "en" and ai_summary and ai_summary != translated_summary:
        fields.append(
            {
                "name": "🇺🇸 English AI Summary",
                "value": _truncate(ai_summary, 1024),
                "inline": False,
            }
        )

    tickers = article.get("tickers") or []
    if tickers:
        fields.append(
            {"name": "💹 Tickers", "value": ", ".join(tickers[:10]), "inline": True}
        )
    sectors = article.get("sectors") or []
    if sectors:
        fields.append(
            {"name": "🏷️ Sectors", "value": ", ".join(sectors[:10]), "inline": True}
        )

    sentiment_label = article.get("sentiment_label")
    sentiment_score = article.get("sentiment")
    if sentiment_label:
        sent_text = sentiment_label
        if sentiment_score is not None:
            sent_text = f"{sentiment_label} ({float(sentiment_score):+.2f})"
        fields.append({"name": "감성", "value": sent_text, "inline": True})

    original_url = article.get("url")
    link_lines: list[str] = []
    if original_url:
        link_lines.append(f"[🔗 원문 보기]({original_url})")
    if finradar_url:
        link_lines.append(f"[📄 FinRadar 상세]({finradar_url})")
    if link_lines:
        fields.append({"name": "링크", "value": "\n".join(link_lines), "inline": False})

    # Footer: source + language + trigger reasons
    footer_parts: list[str] = []
    if article.get("source_type"):
        footer_parts.append(str(article["source_type"]))
    if article.get("language"):
        footer_parts.append(str(article["language"]))
    if triggers:
        footer_parts.append("/".join(triggers))

    timestamp = article.get("published_at") or article.get("last_seen_at")
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp_iso = timestamp.isoformat()
    elif isinstance(timestamp, str):
        timestamp_iso = timestamp
    else:
        timestamp_iso = None

    embed: dict[str, Any] = {
        "title": title,
        "color": _color_for_sentiment(sentiment_label),
    }
    if description:
        embed["description"] = description
    if original_url:
        embed["url"] = original_url
    if fields:
        embed["fields"] = fields
    if footer_parts:
        embed["footer"] = {"text": "  ·  ".join(footer_parts)}
    if timestamp_iso:
        embed["timestamp"] = timestamp_iso

    return embed


def post_alert(
    webhook_url: str,
    *,
    article: dict[str, Any],
    triggers: list[str],
    finradar_url: str | None = None,
) -> bool:
    """Send a single alert to Discord. Returns True on success."""
    if not webhook_url:
        logger.debug("discord.post_alert: empty webhook_url — skipping")
        return False

    embed = build_embed(article=article, triggers=triggers, finradar_url=finradar_url)
    payload = {"embeds": [embed]}

    try:
        with httpx.Client(timeout=_POST_TIMEOUT) as client:
            r = client.post(webhook_url, json=payload)
            # Discord webhooks return 204 No Content on success
            if r.status_code in (200, 204):
                return True
            logger.warning(
                "discord.post_alert: HTTP %s body=%r", r.status_code, r.text[:200]
            )
            return False
    except httpx.HTTPError as exc:
        logger.error("discord.post_alert: request failed: %s", exc)
        return False
