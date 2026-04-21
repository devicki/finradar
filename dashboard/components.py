"""Shared Streamlit components for the FinRadar dashboard.

Keeps the news-card rendering identical across Search, Latest, and Article
pages so any polish we do (badges, cluster siblings expander, feedback
buttons in Phase 3) lands everywhere at once.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import streamlit as st

import api_client


# ---------------------------------------------------------------------------
# Utility helpers (also exported for pages that need them directly)
# ---------------------------------------------------------------------------


def format_ts(ts: str | None) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def sentiment_badge(label: str | None, score: float | None) -> str:
    if not label:
        return "⚪ -"
    colors = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    icon = colors.get(label, "⚪")
    score_txt = f" ({score:+.2f})" if score is not None else ""
    return f"{icon} {label}{score_txt}"


_SOURCE_BADGES: dict[str, str] = {
    "rss": "📰 RSS",
    "api": "📡 API",
    "x_feed": "🐦 X",
    "youtube_post": "▶️ YT",
    "url_report": "🔗 URL",
}


def source_badge(source_type: str | None) -> str:
    if not source_type:
        return ""
    return _SOURCE_BADGES.get(source_type, source_type)


# ---------------------------------------------------------------------------
# Main: render one news item as a bordered card
# ---------------------------------------------------------------------------


def render_news_card(
    item: dict[str, Any],
    *,
    index: int | None = None,
    show_score: bool = False,
    show_cluster_expander: bool = True,
    on_feedback: Callable[[int, str], None] | None = None,
    card_key: str | None = None,
) -> None:
    """Render one item in a bordered container.

    Args:
        item: News item dict as returned by the API (NewsItemResponse /
              NewsItemSearchResponse shape).
        index: Optional 1-based rank to prefix the title with "#N — …".
        show_score: When True and the item carries ``score`` /
                    ``score_breakdown`` (search results), render the
                    right-hand score panel.
        show_cluster_expander: When True and cluster_size ≥ 2, show the
                               inline "같은 스토리 N건" expander.
        on_feedback: Reserved for Phase 3. Signature: ``(news_id, action)``
                     where action ∈ {"like", "dislike", "bookmark", "dismiss"}.
        card_key: Unique key prefix for any Streamlit widgets inside the
                  card. Required if more than one card on a page has feedback
                  buttons (Streamlit widget keys must be globally unique).
    """
    with st.container(border=True):
        if show_score:
            left, right = st.columns([7, 3])
        else:
            left = st.container()
            right = None

        with left:
            _render_card_left(item, index=index, show_cluster_expander=show_cluster_expander)

        if show_score and right is not None:
            with right:
                _render_score_panel(item)

        # Phase 3: feedback buttons (stub — parent opts in by passing on_feedback)
        if on_feedback is not None and item.get("id") is not None:
            _render_feedback_row(item["id"], on_feedback, card_key or f"card_{item['id']}")


# ---------------------------------------------------------------------------
# Internal: card parts
# ---------------------------------------------------------------------------


def _render_card_left(
    item: dict[str, Any],
    *,
    index: int | None,
    show_cluster_expander: bool,
) -> None:
    title = item.get("title", "(제목 없음)")
    url = item.get("url", "#")
    prefix = f"#{index} — " if index is not None else ""
    st.markdown(f"**{prefix}[{title}]({url})**")

    translated = item.get("translated_title")
    if translated and translated != title:
        st.caption(f"🌐 {translated}")

    ai_summary = item.get("ai_summary") or item.get("summary") or ""
    if ai_summary:
        with st.expander("요약 펼치기", expanded=False):
            st.write(ai_summary[:1200])

    # --- meta row -----------------------------------------------------------
    # Prefer the source's published_at; fall back to first_seen_at when
    # missing (legacy rows, or ingest paths without a date).
    publish_ts = item.get("published_at") or item.get("first_seen_at")
    meta_parts = [
        sentiment_badge(item.get("sentiment_label"), item.get("sentiment")),
        f"🌏 {item.get('language') or '?'}",
        f"📅 {format_ts(publish_ts)}",
    ]
    sbadge = source_badge(item.get("source_type"))
    if sbadge:
        meta_parts.append(sbadge)
    if item.get("tickers"):
        meta_parts.append("💹 " + ", ".join(item["tickers"][:5]))
    if item.get("sectors"):
        meta_parts.append("🏷️ " + ", ".join(item["sectors"][:5]))
    st.caption("  ·  ".join(meta_parts))

    # --- inline cluster siblings expander -----------------------------------
    cluster_size = item.get("cluster_size", 1)
    if show_cluster_expander and cluster_size and cluster_size >= 2:
        with st.expander(f"🧩 같은 스토리 {cluster_size}건 펼쳐보기", expanded=False):
            cluster_data = api_client.get_cluster_siblings(item["id"])
            if "error" in cluster_data:
                st.error(f"클러스터 조회 실패: {cluster_data['error']}")
            else:
                siblings = [
                    s for s in cluster_data.get("items", [])
                    if s.get("id") != item.get("id")
                ]
                if not siblings:
                    st.caption("형제 기사 없음")
                for s in siblings:
                    sim = s.get("similarity_to_rep")
                    sim_str = f"`{sim:.3f}`" if sim is not None else "-"
                    st.markdown(
                        f"- {sim_str} "
                        f"[{s.get('title', '')[:100]}]({s.get('url', '#')})  "
                        f"([상세](/Article?news_id={s.get('id')}))"
                    )


def _render_score_panel(item: dict[str, Any]) -> None:
    sb = item.get("score_breakdown") or {}
    final = item.get("score", 0.0)
    st.metric("Final", f"{final:.3f}")
    st.progress(
        min(max(sb.get("fts", 0) or 0.0, 0.0), 1.0),
        text=f"fts={sb.get('fts', 0) or 0:.2f}",
    )
    st.progress(
        min(max(sb.get("cosine", 0) or 0.0, 0.0), 1.0),
        text=f"cos={sb.get('cosine', 0) or 0:.2f}",
    )
    st.progress(
        min(max(sb.get("recency", 0) or 0.0, 0.0), 1.0),
        text=f"rec={sb.get('recency', 0) or 0:.2f}",
    )
    article_id = item.get("id")
    if article_id:
        st.markdown(f"[상세 보기 →](/Article?news_id={article_id})")


def _render_feedback_row(
    news_id: int,
    on_feedback: Callable[[int, str], None],
    key_prefix: str,
) -> None:
    """Phase 3 feedback buttons (wiring stub).

    The buttons are rendered but their clicks delegate to the caller's
    `on_feedback` callback — Phase 3 will connect this to the backend.
    """
    c1, c2, c3, _ = st.columns([1, 1, 1, 7])
    if c1.button("👍", key=f"{key_prefix}_like", help="좋아요"):
        on_feedback(news_id, "like")
    if c2.button("👎", key=f"{key_prefix}_dislike", help="싫어요"):
        on_feedback(news_id, "dislike")
    if c3.button("🔖", key=f"{key_prefix}_bookmark", help="북마크"):
        on_feedback(news_id, "bookmark")
