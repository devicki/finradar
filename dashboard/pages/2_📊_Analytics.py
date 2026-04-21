"""Analytics — 지정한 시간 범위 기사들의 집계 뷰.

감성 분포, Top Tickers, Top Sectors, 그리고 LLM/티커/섹터 커버리지 메트릭.
개별 기사를 훑어보려면 Latest News 페이지를 사용하세요.
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


st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Analytics")
st.caption(
    "최근 N시간 동안 수집된 기사들의 감성 분포 · 상위 티커/섹터 · 처리율 커버리지. "
    "개별 기사를 읽으려면 **Latest News** 페이지로 이동하세요."
)

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

# ---------------------------------------------------------------------------
# 투명성 지표 — Top Ticker/Sector 차트의 신뢰도 판단용
# ---------------------------------------------------------------------------
#
# 티커/섹터는 LLM enrich 단계에서 추출되므로 최근(짧은 기간) 데이터에선 처리
# 미완 기사가 많아 집계가 빈약해 보일 수 있다. 사용자가 "차트가 이상하다"고
# 오해하지 않도록 베이스 비율을 투명하게 노출한다.

llm_done = data.get("articles_llm_enriched", 0)
with_tickers = data.get("articles_with_tickers", 0)
with_sectors = data.get("articles_with_sectors", 0)

llm_pct = (llm_done / total * 100) if total else 0.0
tick_pct = (with_tickers / total * 100) if total else 0.0

m1, m2, m3 = st.columns(3)
m1.metric(
    "LLM 처리율",
    f"{llm_pct:.1f}%",
    delta=f"{llm_done}/{total}",
    delta_color="off",
)
m2.metric(
    "티커 추출 기사",
    f"{with_tickers}",
    delta=f"{tick_pct:.1f}% of total",
    delta_color="off",
)
m3.metric(
    "섹터 태깅 기사",
    f"{with_sectors}",
    delta=f"{(with_sectors/total*100):.1f}% of total",
    delta_color="off",
)

# 경고 1) LLM 처리율이 낮으면 — 짧은 기간에서 자주 발생
if llm_pct < 70:
    st.warning(
        f"⚠️ 이 기간 기사 중 **{llm_pct:.0f}%만 LLM 처리 완료**. "
        "티커/섹터 집계가 실제보다 낮게 나올 수 있습니다. "
        "**더 긴 기간을 선택**하면 안정적인 집계를 볼 수 있어요. "
        "(티커·섹터는 LLM enrich 단계에서 추출됨)"
    )

# 경고 2) 티커 보유 비율 자체가 너무 낮으면
if with_tickers < 20:
    st.info(
        f"ℹ️ 티커가 추출된 기사가 **{with_tickers}건**으로 표본이 작습니다. "
        "Top Tickers 차트의 개별 count가 낮게 찍히는 것은 정상입니다."
    )

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
        # 표본이 빈약하면 subheader 바로 아래에 캡션으로 주의 환기
        max_count = max(t.get("count", 0) for t in tickers)
        if max_count < 3:
            st.caption(
                f"⚠️ 최다 언급도 {max_count}회에 불과 — 표본 부족. "
                "긴 기간을 선택하면 더 명확한 분포를 볼 수 있습니다."
            )
        df_t = pd.DataFrame(tickers)
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
        st.info("티커 데이터 없음 — LLM 처리가 완료되면 표시됩니다.")

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

# ---------------------------------------------------------------------------
# 내 선호도 스냅샷 — 개인화 엔진 투명성
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🧭 내 개인화 선호도")
st.caption(
    "피드백 기반 sector/ticker 선호도. "
    "양수 = 좋아요 많이 한 주제, 음수 = 싫어요/숨김 많이 한 주제. "
    "피드백이 5건 미만이면 의미 있는 신호가 아직 쌓이지 않습니다."
)

aff = api_client.get_affinity()
if "error" in aff:
    st.error(f"affinity 조회 실패: {aff['error']}")
else:
    st.caption(f"누적 피드백: **{aff.get('feedback_rows', 0)}건**")

    col_up, col_dn = st.columns(2)
    with col_up:
        st.markdown("**⬆ 선호 (positive)**")
        tops = (aff.get("top_sectors") or []) + (aff.get("top_tickers") or [])
        tops_pos = [x for x in tops if x.get("score", 0) > 0]
        if tops_pos:
            df_up = pd.DataFrame(tops_pos[:10])
            st.dataframe(df_up, use_container_width=True, hide_index=True)
        else:
            st.caption("아직 선호 신호 없음 — 👍 또는 🔖 버튼을 눌러보세요.")

    with col_dn:
        st.markdown("**⬇ 비선호 (negative)**")
        bottoms = (aff.get("bottom_sectors") or []) + (aff.get("bottom_tickers") or [])
        bottoms_neg = [x for x in bottoms if x.get("score", 0) < 0]
        if bottoms_neg:
            df_dn = pd.DataFrame(bottoms_neg[:10])
            st.dataframe(df_dn, use_container_width=True, hide_index=True)
        else:
            st.caption("아직 비선호 신호 없음 — 👎 또는 🙈 버튼을 눌러보세요.")
