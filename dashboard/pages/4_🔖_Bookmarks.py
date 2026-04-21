"""Bookmarks — 북마크한 기사 모음 + 숨김 기사 관리.

Phase 3 단일 user("owner") 기준. 피드백 버튼은 그대로 render해서 여기서도
북마크 해제 / dismiss 토글이 가능하게 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api_client  # noqa: E402
import components  # noqa: E402


st.set_page_config(page_title="Bookmarks", page_icon="🔖", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar — 탭 전환 + 페이지네이션
# ---------------------------------------------------------------------------

st.sidebar.title("🔖 My Bookmarks")

view = st.sidebar.radio(
    "뷰",
    options=["🔖 북마크", "🙈 숨긴 기사"],
    index=0,
    help="북마크: 나중에 다시 볼 기사 / 숨긴 기사: 피드에서 제외한 기사",
)

with st.sidebar.expander("페이지네이션", expanded=False):
    page = st.number_input("페이지", min_value=1, value=1, step=1)
    page_size = st.select_slider("페이지당", options=[10, 20, 50, 100], value=20)


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

if view.startswith("🔖"):
    st.title("🔖 Bookmarks")
    st.caption(
        "북마크한 기사들. 북마크 토글을 다시 누르면 이 목록에서 빠집니다."
    )
    with st.spinner("북마크 로드 중..."):
        result = api_client.list_bookmarks(page=int(page), page_size=int(page_size))
else:
    st.title("🙈 Dismissed")
    st.caption(
        "숨김 처리한 기사들. Latest/Search 기본 피드에 더 이상 등장하지 않습니다. "
        "실수로 숨긴 경우 🙈 버튼을 다시 눌러 복구하세요."
    )
    with st.spinner("숨긴 기사 로드 중..."):
        result = api_client.list_dismissed(page=int(page), page_size=int(page_size))

if "error" in result:
    st.error(f"API 호출 실패: {result['error']}")
    st.stop()

items = result.get("items", [])
total = result.get("total", 0)

c1, c2 = st.columns(2)
c1.metric("총 개수", f"{total:,}")
c2.metric("페이지", f"{page} / {max(1, (total + page_size - 1) // page_size)}")

st.divider()

if not items:
    if view.startswith("🔖"):
        st.info("아직 북마크한 기사가 없습니다. Latest / Search 페이지에서 🔖 버튼을 눌러보세요.")
    else:
        st.info("숨긴 기사가 없습니다.")
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
