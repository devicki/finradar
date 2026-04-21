"""Article 상세 페이지.

쿼리 파라미터 ``?news_id=123`` 로 접근. enrichment 전체 표시 +
제목 기반 하이브리드 검색으로 유사 기사 top 5.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api_client  # noqa: E402


st.set_page_config(page_title="Article Detail", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# news_id 획득 — 쿼리 파라미터 or 수동 입력
# ---------------------------------------------------------------------------

qp = st.query_params
default_id = qp.get("news_id", "")

news_id_str = st.text_input(
    "Article ID",
    value=default_id,
    placeholder="예: 1207",
    help="Search 페이지의 '상세 보기' 링크를 누르거나 숫자 ID를 직접 입력하세요.",
)

if not news_id_str or not news_id_str.isdigit():
    st.info("article ID를 입력하면 상세 정보가 표시됩니다.")
    st.stop()

news_id = int(news_id_str)


# ---------------------------------------------------------------------------
# 기사 조회
# ---------------------------------------------------------------------------


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def _sent_badge(label: str | None, score: float | None) -> str:
    if not label:
        return "⚪ -"
    colors = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    icon = colors.get(label, "⚪")
    score_txt = f" ({score:+.3f})" if score is not None else ""
    return f"{icon} {label}{score_txt}"


with st.spinner(f"article #{news_id} 로드 중..."):
    article = api_client.get_article(news_id)

if "error" in article:
    st.error(f"API 호출 실패: {article['error']}")
    st.stop()

if not article or not article.get("id"):
    st.warning("기사를 찾을 수 없습니다.")
    st.stop()


# ---------------------------------------------------------------------------
# 본문 렌더
# ---------------------------------------------------------------------------

title = article.get("title", "(제목 없음)")
url = article.get("url", "#")
st.title(f"📄 {title}")
st.markdown(f"🔗 [원문 링크]({url})")

# 번역 제목
if article.get("translated_title") and article["translated_title"] != title:
    st.caption(f"🌐 {article['translated_title']}")

# 메타 정보
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("감성", _sent_badge(article.get("sentiment_label"), article.get("sentiment")))
mcol2.metric("언어", article.get("language") or "-")
mcol3.metric("조회수", f"{article.get('hit_count', 0):,}")
mcol4.metric(
    "Published",
    _fmt_ts(article.get("published_at") or article.get("first_seen_at")),
)

st.divider()

# 좌측: 요약들, 우측: 구조화 메타
left, right = st.columns([3, 2])

with left:
    st.subheader("AI Summary")
    ai_sum = article.get("ai_summary")
    if ai_sum:
        st.info(ai_sum)
    else:
        st.caption("AI 요약 없음")

    st.subheader("Translated Summary")
    tr_sum = article.get("translated_summary")
    if tr_sum:
        st.info(tr_sum)
    else:
        st.caption("번역 요약 없음")

    st.subheader("원본 Summary")
    orig = article.get("summary")
    if orig:
        with st.expander("원본 텍스트 펼치기", expanded=False):
            st.write(orig)
    else:
        st.caption("원본 summary 없음")

with right:
    st.subheader("Meta")
    st.write(f"**ID**: `{article.get('id')}`")
    st.write(f"**Source**: `{article.get('source_type')}`")
    st.write(f"**Feed**: {article.get('source_url')}")
    st.write(f"**First Seen**: {_fmt_ts(article.get('first_seen_at'))}")
    st.write(f"**Created**: {_fmt_ts(article.get('created_at'))}")

    tickers = article.get("tickers") or []
    if tickers:
        st.subheader("💹 Tickers")
        st.write(", ".join(f"`{t}`" for t in tickers))

    sectors = article.get("sectors") or []
    if sectors:
        st.subheader("🏷️ Sectors")
        st.write(", ".join(f"`{s}`" for s in sectors))

st.divider()


# ---------------------------------------------------------------------------
# 클러스터 형제 — 주기 태스크가 묶은 "같은 스토리" 기사들
# ---------------------------------------------------------------------------

cluster_size = article.get("cluster_size", 1)
cluster_rep_id = article.get("cluster_rep_id")

if cluster_rep_id is not None and cluster_size and cluster_size >= 2:
    st.subheader(f"🧩 같은 스토리 ({cluster_size}건)")
    st.caption(
        "30분 주기 클러스터링이 묶은 '같은 사건'의 다른 보도. "
        "대표 기사 기준 cosine 유사도 내림차순."
    )

    with st.spinner("클러스터 멤버 조회 중..."):
        cluster_data = api_client.get_cluster_siblings(news_id)

    if "error" in cluster_data:
        st.error(f"클러스터 조회 실패: {cluster_data['error']}")
    else:
        siblings = [x for x in cluster_data.get("items", []) if x.get("id") != news_id]
        for s in siblings:
            with st.container(border=True):
                lc, rc = st.columns([7, 2])
                with lc:
                    st.markdown(
                        f"**[{s.get('title', '')[:100]}]({s.get('url', '#')})**"
                    )
                    meta = [
                        _sent_badge(s.get("sentiment_label"), s.get("sentiment")),
                        f"🌏 {s.get('language') or '?'}",
                        f"📅 {_fmt_ts(s.get('published_at') or s.get('last_seen_at'))}",
                    ]
                    if s.get("tickers"):
                        meta.append("💹 " + ", ".join(s["tickers"][:3]))
                    if s.get("sectors"):
                        meta.append("🏷️ " + ", ".join(s["sectors"][:3]))
                    st.caption("  ·  ".join(meta))
                with rc:
                    sim = s.get("similarity_to_rep")
                    if sim is not None:
                        st.metric("유사도", f"{sim:.3f}")
                    st.markdown(f"[상세 →](?news_id={s.get('id')})")

    st.divider()


# ---------------------------------------------------------------------------
# 유사 기사 — 이 기사 제목을 쿼리로 하이브리드 검색
# ---------------------------------------------------------------------------

st.subheader("🔗 유사 기사 (하이브리드 검색 기반)")
st.caption(
    "이 기사의 제목을 쿼리로 재검색. 위의 '같은 스토리'보다 느슨한 기준이며 "
    "다른 각도의 관련 기사를 보여줍니다."
)

with st.spinner("유사 기사 검색 중..."):
    similar = api_client.search(
        query=title,
        include_scores=True,
        page_size=6,
        weight_bm25=0.3,
        weight_cosine=0.6,
        weight_recency=0.1,
    )

if "error" in similar:
    st.error(f"유사 기사 검색 실패: {similar['error']}")
else:
    items = [x for x in similar.get("items", []) if x.get("id") != news_id]
    if not items:
        st.caption("유사 기사 없음")
    for x in items[:5]:
        sb = x.get("score_breakdown") or {}
        with st.container(border=True):
            lc, rc = st.columns([7, 2])
            with lc:
                st.markdown(
                    f"**[{x.get('title', '')[:100]}]({x.get('url', '#')})**"
                )
                sectors_x = x.get("sectors") or []
                tickers_x = x.get("tickers") or []
                meta_parts = [
                    _sent_badge(x.get("sentiment_label"), x.get("sentiment")),
                    f"🌏 {x.get('language') or '?'}",
                    f"📅 {_fmt_ts(x.get('published_at') or x.get('last_seen_at'))}",
                ]
                if tickers_x:
                    meta_parts.append("💹 " + ", ".join(tickers_x[:3]))
                if sectors_x:
                    meta_parts.append("🏷️ " + ", ".join(sectors_x[:3]))
                st.caption("  ·  ".join(meta_parts))
            with rc:
                st.metric("Score", f"{x.get('score', 0):.3f}")
                st.caption(
                    f"fts={sb.get('fts', 0):.2f} · cos={sb.get('cosine', 0):.2f} · "
                    f"rec={sb.get('recency', 0):.2f}"
                )
                st.markdown(f"[상세 →](?news_id={x.get('id')})")
