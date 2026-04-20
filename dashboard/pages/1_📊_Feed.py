"""Feed 요약 페이지.

지정한 시간 범위 내 기사들의 감성 분포, 상위 티커/섹터 집계.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# dashboard/api_client.py 를 import 가능하게
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api_client  # noqa: E402


st.set_page_config(page_title="Feed Summary", page_icon="📊", layout="wide")

st.title("📊 Feed Summary")
st.caption("최근 N시간 동안 수집된 기사들의 감성 분포 + 상위 티커/섹터")

# ---------------------------------------------------------------------------
# 사이드바: 기간 선택
# ---------------------------------------------------------------------------

hours_option = st.sidebar.radio(
    "기간",
    options=[6, 24, 72, 168, 720],
    format_func=lambda h: {6: "6시간", 24: "24시간", 72: "3일", 168: "7일", 720: "30일"}[h],
    index=1,
)

with st.spinner(f"최근 {hours_option}시간 피드 요약 중..."):
    data = api_client.feed_summary(hours=int(hours_option))

if "error" in data:
    st.error(f"API 호출 실패: {data['error']}")
    st.stop()

# ---------------------------------------------------------------------------
# 상단 지표
# ---------------------------------------------------------------------------

total = data.get("total_count", 0)
dist = data.get("sentiment_distribution", {})
pos = dist.get("positive", 0)
neg = dist.get("negative", 0)
neu = dist.get("neutral", 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 기사", f"{total:,}")
c2.metric("🟢 Positive", f"{pos:,}", delta=f"{(pos/total*100 if total else 0):.1f}%")
c3.metric("🔴 Negative", f"{neg:,}", delta=f"{(neg/total*100 if total else 0):.1f}%")
c4.metric("⚪ Neutral", f"{neu:,}", delta=f"{(neu/total*100 if total else 0):.1f}%")

if total == 0:
    st.warning("해당 기간에 기사가 없습니다.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# 감성 분포 파이차트
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([3, 4])

with col_left:
    st.subheader("감성 분포")
    sent_df = pd.DataFrame(
        {
            "label": ["positive", "negative", "neutral"],
            "count": [pos, neg, neu],
        }
    )
    fig_pie = px.pie(
        sent_df,
        names="label",
        values="count",
        color="label",
        color_discrete_map={
            "positive": "#10b981",
            "negative": "#ef4444",
            "neutral": "#9ca3af",
        },
        hole=0.4,
    )
    fig_pie.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# 상위 티커 바차트
# ---------------------------------------------------------------------------

with col_right:
    st.subheader("Top Tickers")
    tickers = data.get("top_tickers", [])
    if tickers:
        df_t = pd.DataFrame(tickers)
        # 컬럼명 자동 추정 ({'ticker': 'AAPL', 'count': 5} 형식)
        df_t = df_t.rename(columns={df_t.columns[0]: "ticker", df_t.columns[1]: "count"})
        fig_t = px.bar(
            df_t.head(15).sort_values("count", ascending=True),
            x="count",
            y="ticker",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
        )
        fig_t.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("티커 데이터 없음")

# ---------------------------------------------------------------------------
# 상위 섹터 바차트 (전체 폭)
# ---------------------------------------------------------------------------

st.subheader("Top Sectors")
sectors = data.get("top_sectors", [])
if sectors:
    df_s = pd.DataFrame(sectors)
    df_s = df_s.rename(columns={df_s.columns[0]: "sector", df_s.columns[1]: "count"})
    fig_s = px.bar(
        df_s.head(20).sort_values("count", ascending=True),
        x="count",
        y="sector",
        orientation="h",
        color="count",
        color_continuous_scale="Purples",
    )
    fig_s.update_layout(height=500, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig_s, use_container_width=True)
else:
    st.info("섹터 데이터 없음")

st.divider()
st.caption(f"생성 시각: {data.get('generated_at', '-')}  ·  윈도우: {data.get('window_hours')}시간")
