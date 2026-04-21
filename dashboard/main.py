"""FinRadar 대시보드 — 메인/Search 페이지.

Streamlit 실행:
    streamlit run /app/dashboard/main.py --server.port 8501 --server.address 0.0.0.0
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

import api_client

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinRadar Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 사이드바 — 검색/필터/가중치
# ---------------------------------------------------------------------------

st.sidebar.title("🔍 Hybrid Search")

query = st.sidebar.text_input(
    "검색어",
    value=st.session_state.get("query", ""),
    placeholder="예: 반도체 실적, Nvidia earnings, 연준 금리…",
    key="query",
)

with st.sidebar.expander("필터", expanded=True):
    language = st.selectbox("언어", options=["(전체)", "en", "ko"], index=0)
    source_type = st.selectbox(
        "소스 타입",
        options=["(전체)", "rss", "api", "x_feed", "youtube_post", "url_report"],
        index=0,
    )
    sentiment_label = st.selectbox(
        "감성",
        options=["(전체)", "positive", "negative", "neutral"],
        index=0,
    )
    ticker_input = st.text_input("티커 (쉼표 구분)", placeholder="AAPL, NVDA")
    sector_input = st.text_input("섹터 (쉼표 구분)", placeholder="반도체, AI")

with st.sidebar.expander("하이브리드 가중치", expanded=True):
    st.caption("슬라이더 합이 1일 필요는 없음 — 상대 비율만 의미 있음")
    w_bm25 = st.slider("BM25 (FTS)", 0.0, 1.0, 0.6, 0.05)
    w_cos = st.slider("Cosine (의미)", 0.0, 1.0, 0.3, 0.05)
    w_rec = st.slider("Recency (최신성)", 0.0, 1.0, 0.1, 0.05)

with st.sidebar.expander("페이지네이션", expanded=False):
    page = st.number_input("페이지", min_value=1, value=1, step=1)
    page_size = st.select_slider("페이지당", options=[5, 10, 20, 50], value=20)

dedup = st.sidebar.toggle(
    "중복 제거 (클러스터 대표만)",
    value=False,
    help="같은 클러스터의 여러 기사 중 대표 1건만 표시",
)

search_clicked = st.sidebar.button("🔎 검색 실행", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# URL 직접 수집 — 기존 수집기가 닿지 않는 임의 URL을 즉석 ingest
# ---------------------------------------------------------------------------

with st.sidebar.expander("🔗 URL 직접 수집", expanded=False):
    st.caption(
        "블로그 / 리서치 PDF / 증권사 리포트 URL을 붙여넣으면 "
        "trafilatura · pdfplumber로 본문을 추출해 DB에 넣습니다. "
        "로그인 필요 URL은 에러로 알려줍니다."
    )
    ingest_input = st.text_area("URL", height=80, key="ingest_url_input", placeholder="https://...")
    ingest_force_pdf = st.checkbox("PDF로 강제 처리", value=False, key="ingest_force_pdf")
    if st.button("📥 Ingest", key="ingest_submit"):
        if not ingest_input.strip():
            st.warning("URL을 입력하세요.")
        else:
            with st.spinner("URL 수집 중..."):
                resp = api_client.ingest_url(
                    ingest_input.strip(),
                    force_pdf=ingest_force_pdf,
                )
            if resp.get("status") == "ok":
                st.success(
                    f"✅ id={resp['news_id']} / {resp.get('extracted_len', 0)}자 추출됨  \n"
                    f"**{resp.get('title', '(no title)')[:80]}**"
                )
                nid = resp.get("news_id")
                if nid:
                    st.markdown(f"[상세 페이지 →](/Article?news_id={nid})")
            elif resp.get("status") == "login_required":
                st.error(f"🔒 로그인 필요: {resp.get('message')}")
            elif resp.get("status") == "unsupported":
                st.warning(f"⚠️ 미지원 형식: {resp.get('message')}")
            else:
                st.error(f"❌ {resp.get('status')}: {resp.get('message') or resp.get('error')}")


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------


def _parse_list(text: str) -> list[str] | None:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts or None


def _sentiment_badge(label: str | None, score: float | None) -> str:
    if not label:
        return "⚪ -"
    colors = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    icon = colors.get(label, "⚪")
    score_txt = f" ({score:+.2f})" if score is not None else ""
    return f"{icon} {label}{score_txt}"


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


# ---------------------------------------------------------------------------
# 메인 영역
# ---------------------------------------------------------------------------

st.title("📡 FinRadar — Hybrid Search")
st.caption(
    "PostgreSQL FTS + pgvector cosine + recency decay 가중 랭킹. "
    "가중치 슬라이더와 필터를 조합해 어떤 신호가 어떻게 기여하는지 디버깅하세요."
)

if not query or not search_clicked:
    st.info("왼쪽 사이드바에서 검색어를 입력하고 '검색 실행'을 누르세요.")
    st.stop()

# ---- API 호출 ---------------------------------------------------------------

with st.spinner("하이브리드 검색 중..."):
    result = api_client.search(
        query=query,
        language=None if language == "(전체)" else language,
        source_type=None if source_type == "(전체)" else source_type,
        sentiment_label=None if sentiment_label == "(전체)" else sentiment_label,
        tickers=_parse_list(ticker_input),
        sectors=_parse_list(sector_input),
        include_scores=True,
        dedup=dedup,
        weight_bm25=w_bm25,
        weight_cosine=w_cos,
        weight_recency=w_rec,
        page=int(page),
        page_size=int(page_size),
    )

if "error" in result:
    st.error(f"API 호출 실패: {result['error']}")
    st.stop()

items = result.get("items", [])
total = result.get("total", 0)
expansion = result.get("query_expansion")

# ---- 헤더 요약 --------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 매칭", f"{total:,}")
col2.metric("현재 페이지", f"{page} / {(total + page_size - 1) // page_size or 1}")
if items:
    col3.metric("Top 점수", f"{items[0].get('score', 0):.3f}")
    col4.metric("평균 점수", f"{sum(i.get('score', 0) for i in items) / len(items):.3f}")
else:
    col3.metric("Top 점수", "-")
    col4.metric("평균 점수", "-")

# ---- 쿼리 확장 안내 --------------------------------------------------------
if expansion and expansion.get("expanded_tokens"):
    exp_badges = []
    for tok, syns in expansion["expanded_tokens"].items():
        syn_str = " · ".join(syns)
        exp_badges.append(f"**{tok}** → `{syn_str}`")
    st.info("🔎 쿼리 확장: " + "  |  ".join(exp_badges))

# ---- FTS 품질 경고 ---------------------------------------------------------
# Top-N (기본 3건) 기준으로 판단. 상위 결과 전원이 키워드 매칭 없이
# cosine-only로 선정됐으면 사용자가 이를 인지하도록 안내.
if items:
    top_n = min(3, len(items))
    top_items = items[:top_n]
    top_fts_zero = all(
        (i.get("score_breakdown") or {}).get("fts", 0) == 0 for i in top_items
    )
    any_fts_zero_in_page = any(
        (i.get("score_breakdown") or {}).get("fts", 0) == 0 for i in items
    )

    if top_fts_zero:
        st.warning(
            f"⚠️ **상위 {top_n}건이 모두 키워드 매칭 없이 의미 검색(cosine)만으로 선정됐습니다.** "
            "기사 본문에 해당 단어가 없어 정확도가 낮을 수 있습니다. "
            "더 구체적인 도메인 용어(예: '오일' → '유가'/'국제유가', "
            "'금리' → '기준금리')를 사용하거나 `finradar/search/query_expansion.py`에 "
            "동의어를 추가해보세요."
        )
    elif any_fts_zero_in_page:
        # 일부만 fts=0 이면 info 수준으로 약하게 안내
        zero_count = sum(
            1 for i in items if (i.get("score_breakdown") or {}).get("fts", 0) == 0
        )
        st.caption(
            f"ℹ️ 현재 페이지 {len(items)}건 중 {zero_count}건은 키워드 매칭 없이 "
            "의미 검색(cosine)으로만 선정됐습니다."
        )

st.divider()

if not items:
    st.warning("결과가 없습니다. 쿼리나 필터를 바꿔보세요.")
    st.stop()

# ---- 결과 리스트 ------------------------------------------------------------

for i, item in enumerate(items, start=1 + (int(page) - 1) * int(page_size)):
    sb = item.get("score_breakdown") or {}

    # 타이틀 영역
    with st.container(border=True):
        left, right = st.columns([7, 3])

        with left:
            # 제목 + 링크
            title = item.get("title", "(제목 없음)")
            url = item.get("url", "#")
            st.markdown(f"**#{i} — [{title}]({url})**")

            # 번역/요약
            translated = item.get("translated_title")
            if translated and translated != title:
                st.caption(f"🌐 {translated}")

            ai_summary = item.get("ai_summary") or item.get("summary") or ""
            if ai_summary:
                with st.expander("요약 펼치기", expanded=False):
                    st.write(ai_summary[:1000])

            # 소스 타입 배지 — 한눈에 기사 출처 형태 인식
            source_badges = {
                "rss": "📰 RSS",
                "api": "📡 API",
                "x_feed": "🐦 X",
                "youtube_post": "▶️ YT",
                "url_report": "🔗 URL",
            }
            source_badge = source_badges.get(item.get("source_type"), item.get("source_type") or "")

            # 메타 정보 (날짜는 published_at 우선, 없으면 first_seen_at)
            _publish_ts = item.get("published_at") or item.get("first_seen_at")
            meta_parts = [
                _sentiment_badge(item.get("sentiment_label"), item.get("sentiment")),
                f"🌏 {item.get('language') or '?'}",
                f"📅 {_fmt_ts(_publish_ts)}",
            ]
            if source_badge:
                meta_parts.append(source_badge)
            if item.get("tickers"):
                meta_parts.append("💹 " + ", ".join(item["tickers"][:5]))
            if item.get("sectors"):
                meta_parts.append("🏷️ " + ", ".join(item["sectors"][:5]))
            st.caption("  ·  ".join(meta_parts))

            # 클러스터 siblings 인라인 펼치기 (2건 이상일 때만)
            cluster_size = item.get("cluster_size", 1)
            if cluster_size and cluster_size >= 2:
                with st.expander(f"🧩 같은 스토리 {cluster_size}건 펼쳐보기", expanded=False):
                    cluster_data = api_client.get_cluster_siblings(item["id"])
                    if "error" in cluster_data:
                        st.error(f"클러스터 조회 실패: {cluster_data['error']}")
                    else:
                        siblings = [
                            s for s in cluster_data.get("items", [])
                            if s.get("id") != item["id"]
                        ]
                        if not siblings:
                            st.caption("형제 기사 없음")
                        for s in siblings:
                            sim = s.get("similarity_to_rep")
                            sim_str = f"`{sim:.3f}`" if sim is not None else "-"
                            st.markdown(
                                f"- {sim_str} "
                                f"[{s.get('title', '')[:100]}]({s.get('url', '#')})  "
                                f"([상세](?news_id={s.get('id')}))"
                            )

        with right:
            # 점수 breakdown
            final = item.get("score", 0.0)
            st.metric("Final", f"{final:.3f}")
            st.progress(min(max(final, 0.0), 1.0), text=f"fts={sb.get('fts', 0):.2f}")
            st.progress(min(max(sb.get("cosine", 0), 0.0), 1.0), text=f"cos={sb.get('cosine', 0):.2f}")
            st.progress(min(max(sb.get("recency", 0), 0.0), 1.0), text=f"rec={sb.get('recency', 0):.2f}")

            # 상세 페이지 링크 (쿼리 파라미터로 news_id 전달)
            article_id = item.get("id")
            if article_id:
                st.markdown(
                    f"[상세 보기 →](?news_id={article_id})",
                    help="Article 페이지로 이동",
                )
