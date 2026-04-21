"""Latest News — browse the full feed, not just aggregated summaries.

Defaults to reverse-chronological order with cluster-dedup ON, so the same
story isn't repeated. Users pick alternative sorts from the sidebar
(cluster_size = biggest stories / sentiment_strength = strong signal first).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

# Ensure imports resolve to the /app/dashboard module (same as other pages).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api_client  # noqa: E402
import components  # noqa: E402


st.set_page_config(page_title="Latest News", page_icon="📋", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar — sort + filters + pagination
# ---------------------------------------------------------------------------

st.sidebar.title("📋 Latest News")

sort_label_to_value = {
    "🕒 최신순 (recency)": "latest",
    "🧩 큰 스토리 우선": "cluster_size",
    "💢 감성 강한 순": "sentiment_strength",
    "🧭 개인화 (recency × personal_boost)": "personalized",
}
sort_label = st.sidebar.radio(
    "정렬",
    options=list(sort_label_to_value.keys()),
    index=0,
)
sort_value = sort_label_to_value[sort_label]

dedup = st.sidebar.toggle(
    "중복 제거 (클러스터 대표만)",
    value=True,
    help="켜져 있으면 같은 클러스터의 여러 기사 중 대표 1건만 보여줍니다.",
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
    ticker = st.text_input("티커 (단일)", placeholder="AAPL")
    sector = st.text_input("섹터 (단일)", placeholder="반도체")

with st.sidebar.expander("기간", expanded=False):
    today = datetime.utcnow().date()
    use_date_filter = st.checkbox("기간 필터 사용", value=False)
    default_from = today - timedelta(days=7)
    date_from = st.date_input("From", value=default_from)
    date_to = st.date_input("To", value=today)

with st.sidebar.expander("페이지네이션", expanded=False):
    page = st.number_input("페이지", min_value=1, value=1, step=1)
    page_size = st.select_slider("페이지당", options=[10, 20, 50, 100], value=20)


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

st.title("📋 Latest News")
st.caption(
    "모든 소스를 합친 최신 뉴스 목록. 사이드바에서 정렬/필터를 바꿔가며 피드를 훑어보세요."
)


with st.spinner("피드 로드 중..."):
    result = api_client.feed(
        language=None if language == "(전체)" else language,
        source_type=None if source_type == "(전체)" else source_type,
        sentiment_label=None if sentiment_label == "(전체)" else sentiment_label,
        ticker=ticker.strip() or None,
        sector=sector.strip() or None,
        date_from=date_from.isoformat() if use_date_filter else None,
        date_to=date_to.isoformat() if use_date_filter else None,
        dedup=dedup,
        sort=sort_value,
        page=int(page),
        page_size=int(page_size),
    )

if "error" in result:
    st.error(f"API 호출 실패: {result['error']}")
    st.stop()

items = result.get("items", [])
total = result.get("total", 0)

# Header metrics
c1, c2, c3 = st.columns(3)
c1.metric("매칭 건수", f"{total:,}")
c2.metric(
    "페이지",
    f"{page} / {max(1, (total + page_size - 1) // page_size)}",
)
c3.metric("정렬", sort_label.split(" (")[0])

st.divider()

if not items:
    st.info("조건에 맞는 기사가 없습니다. 필터를 조정해보세요.")
    st.stop()

fb_states = components.load_feedback_states(items)

start_index = (int(page) - 1) * int(page_size) + 1
for offset, item in enumerate(items):
    components.render_news_card(
        item,
        index=start_index + offset,
        show_score=False,
        show_cluster_expander=True,
        show_feedback=True,
        feedback_state=fb_states.get(item["id"], []),
    )
