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
    show_feedback: bool = False,
    feedback_state: list[str] | None = None,
    card_key: str | None = None,
) -> None:
    """Render one item in a bordered container.

    Args:
        item: News item dict as returned by the API.
        index: Optional 1-based rank to prefix the title with "#N — …".
        show_score: When True and the item carries ``score`` /
                    ``score_breakdown`` (search results), render the
                    right-hand score panel.
        show_cluster_expander: When True and cluster_size ≥ 2, show the
                               inline "같은 스토리 N건" expander.
        show_feedback: When True, render 👍/👎/🔖/🙈 buttons under the card.
                       The caller batch-fetches current states and passes
                       them via ``feedback_state`` so no per-card API calls.
        feedback_state: List of actions currently set on this item, e.g.
                        ``["like", "bookmark"]``. None = fall back to empty
                        (equivalent to no active actions).
        card_key: Unique key prefix for Streamlit widgets inside the card.
                  Must be unique per card on a page; defaults to
                  ``f"card_{item['id']}"`` when omitted.
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

        if show_feedback and item.get("id") is not None:
            _render_feedback_row(
                item["id"],
                feedback_state or [],
                card_key or f"card_{item['id']}",
            )


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

    translated_title = (item.get("translated_title") or "").strip()
    if translated_title and translated_title != title:
        st.caption(f"🌐 {translated_title}")

    # ------------------------------------------------------------------
    # Summary area — unified "Korean-first" template so EN and KO cards
    # feel the same to a Korean reader:
    #
    #                           INLINE (always shown)          EXPANDER
    #   KO article  |  📝 ai_summary (native Korean)      |  📰 원문
    #   EN article  |  📝 translated_summary (Korean)     |  🇺🇸 English AI
    #               |                                      |  📰 원문 (English)
    #
    # The inline block is always the Korean-readable digest so you can
    # scan a whole page without clicking; the expander exposes the
    # English original AI summary (EN only) plus the raw source body for
    # either language.
    # ------------------------------------------------------------------
    ai_summary = (item.get("ai_summary") or "").strip()
    translated_summary = (item.get("translated_summary") or "").strip()
    original_summary = (item.get("summary") or "").strip()
    language = (item.get("language") or "").lower()

    # Choose the "Korean-facing" summary for the inline preview.
    # EN articles: prefer translated_summary (Korean); fall back to ai_summary
    # KO articles: use ai_summary (already Korean); translated_summary is empty
    if language == "en" and translated_summary:
        inline_summary = translated_summary
    elif ai_summary:
        inline_summary = ai_summary
    else:
        inline_summary = ""

    _preview_len = 400
    if inline_summary:
        if len(inline_summary) <= _preview_len:
            st.write(f"📝 {inline_summary}")
        else:
            st.write(f"📝 {inline_summary[:_preview_len]}…")
    elif original_summary:
        # No AI enrichment yet (usually because FinBERT scored the item as
        # neutral and the enrich filter skipped it). Show the raw body so
        # the card still has readable content.
        preview = original_summary[:_preview_len]
        if len(original_summary) > _preview_len:
            preview += "…"
        st.write(f"📰 {preview}")

    # Expander decision: something extra to reveal beyond the inline preview?
    #   - inline summary truncated
    #   - EN article: English ai_summary is extra content we haven't shown yet
    #   - original RSS body is distinct from the inline preview
    has_more = bool(
        (inline_summary and len(inline_summary) > _preview_len)
        or (language == "en" and ai_summary and ai_summary != inline_summary)
        or (
            original_summary
            and original_summary != inline_summary
            and (len(original_summary) > _preview_len or inline_summary)
        )
    )
    if has_more:
        with st.expander("전체 요약 · 원문", expanded=False):
            # Korean-facing AI summary (either native KO or the translation).
            if inline_summary:
                label = "🌐 한국어 번역 요약" if language == "en" else "📝 AI 요약"
                st.markdown(f"**{label}**")
                st.write(inline_summary[:2500])

            # English AI summary — only when the original ai_summary differs
            # from the inline block (EN articles).
            if language == "en" and ai_summary and ai_summary != inline_summary:
                if inline_summary:
                    st.markdown("---")
                st.markdown("**🇺🇸 English AI Summary**")
                st.write(ai_summary[:2500])

            # Original RSS / body-fetch content (any language).
            if original_summary and original_summary not in (inline_summary, ai_summary):
                if inline_summary or ai_summary:
                    st.markdown("---")
                st.markdown("**📰 원문 (RSS / 본문)**")
                st.write(original_summary[:4000])

    # --- meta row -----------------------------------------------------------
    # Prefer the source's published_at; fall back to first_seen_at when
    # missing (legacy rows, or ingest paths without a date).
    #
    # Sentiment is now two signals — local model (FinBERT/KR-FinBert-SC) and
    # the cloud LLM. We label them with "L:" / "A:" (Local / AI) so a reader
    # can spot the gap at a glance. Pre-migration rows (no llm_sentiment yet)
    # render only the local badge, preserving the original compact layout.
    publish_ts = item.get("published_at") or item.get("first_seen_at")
    local_badge = "L:" + sentiment_badge(
        item.get("sentiment_label"), item.get("sentiment")
    )
    meta_parts: list[str] = [local_badge]
    if item.get("llm_sentiment") is not None or item.get("llm_sentiment_label"):
        meta_parts.append(
            "A:" + sentiment_badge(
                item.get("llm_sentiment_label"), item.get("llm_sentiment")
            )
        )
    meta_parts.extend(
        [
            f"🌏 {item.get('language') or '?'}",
            f"📅 {format_ts(publish_ts)}",
        ]
    )
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
    current_actions: list[str],
    key_prefix: str,
) -> None:
    """Render the 4 toggleable feedback buttons with live state.

    Active actions show an active icon (e.g. 👍 on, ⚪ off); clicks post to
    the API and immediately st.rerun so the next frame reflects the new
    state. Like / dislike are mutually exclusive on the server side — this
    UI simply submits whichever the user clicked and lets the API remove
    the opposite row.
    """
    active = set(current_actions)

    def _icon(action: str, on_icon: str, off_icon: str) -> str:
        return on_icon if action in active else off_icon

    labels = [
        ("like",     _icon("like",     "👍", "🤍"), "좋아요"),
        ("dislike",  _icon("dislike",  "👎", "⬜"), "싫어요"),
        ("bookmark", _icon("bookmark", "🔖", "📑"), "북마크"),
        ("dismiss",  _icon("dismiss",  "🙈", "👁️"), "숨김"),
    ]

    cols = st.columns([1, 1, 1, 1, 6])
    for i, (action, icon, helptext) in enumerate(labels):
        if cols[i].button(icon, key=f"{key_prefix}_{action}", help=helptext):
            _handle_feedback_click(news_id, action, action in active)


def _handle_feedback_click(news_id: int, action: str, currently_active: bool) -> None:
    """Toggle the feedback on the server, invalidate session cache, rerun."""
    if currently_active:
        api_client.delete_feedback(news_id, action)
    else:
        api_client.submit_feedback(news_id, action)

    # Invalidate the page's cached feedback snapshot so the next run re-fetches.
    if "feedback_states" in st.session_state:
        st.session_state.pop("feedback_states", None)
    st.rerun()


# ---------------------------------------------------------------------------
# Helper for pages: batch-load feedback state for a list of items
# ---------------------------------------------------------------------------


def load_feedback_states(items: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Return ``{news_id: [actions]}`` for the rendered items.

    Stores the result in ``st.session_state`` so toggle-driven reruns reuse
    the value until a feedback click explicitly invalidates it.
    """
    ids = [it["id"] for it in items if it.get("id") is not None]
    if not ids:
        return {}

    cache_key = "feedback_states"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = api_client.feedback_status_batch(ids)
    return st.session_state[cache_key]
