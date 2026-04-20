"""FinRadar FastAPI 엔드포인트의 Streamlit 대시보드용 래퍼.

- 내부 Docker 네트워크에서는 ``http://app:8000`` 사용
- 로컬 실행(포트 매핑) 시 ``FINRADAR_API_URL`` env로 오버라이드 가능
- 모든 메서드는 HTTP 오류를 swallow하고 빈 값을 반환 — 대시보드에서 오류 배너로 표시
"""
from __future__ import annotations

import os
from typing import Any

import httpx


_API_BASE = os.getenv("FINRADAR_API_URL", "http://app:8000").rstrip("/")
_API_PREFIX = f"{_API_BASE}/api/v1"
_TIMEOUT = httpx.Timeout(30.0)


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET JSON, 오류 시 {'error': str} 반환."""
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(f"{_API_PREFIX}{path}", params=_clean(params))
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        return {"error": str(e)}


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.post(f"{_API_PREFIX}{path}", json=_clean(body))
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        return {"error": str(e)}


def _clean(d: dict[str, Any] | None) -> dict[str, Any]:
    """None 값을 제거해 쿼리/바디에서 빈 필드 누락."""
    if not d:
        return {}
    return {k: v for k, v in d.items() if v not in (None, "", [])}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(
    query: str,
    *,
    language: str | None = None,
    source_type: str | None = None,
    sentiment_label: str | None = None,
    tickers: list[str] | None = None,
    sectors: list[str] | None = None,
    include_scores: bool = True,
    dedup: bool = False,
    weight_bm25: float | None = None,
    weight_cosine: float | None = None,
    weight_recency: float | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """하이브리드 검색.

    응답에는 매칭된 기사 items 와, 동의어 확장이 적용된 경우
    ``query_expansion`` (original, tsquery_expr, expanded_tokens) 필드가 포함된다.
    """
    return _post(
        "/search/",
        {
            "query": query,
            "language": language,
            "source_type": source_type,
            "sentiment_label": sentiment_label,
            "tickers": tickers,
            "sectors": sectors,
            "include_scores": include_scores,
            "dedup": dedup,
            "weight_bm25": weight_bm25,
            "weight_cosine": weight_cosine,
            "weight_recency": weight_recency,
            "page": page,
            "page_size": page_size,
        },
    )


def get_cluster_siblings(news_id: int) -> dict[str, Any]:
    """Fetch all articles sharing a cluster with the given news item."""
    return _get(f"/news/{news_id}/cluster")


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------


def feed(
    *,
    language: str | None = None,
    source_type: str | None = None,
    sentiment_label: str | None = None,
    ticker: str | None = None,
    sector: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """시간순 뉴스 피드."""
    return _get(
        "/feed/",
        {
            "language": language,
            "source_type": source_type,
            "sentiment_label": sentiment_label,
            "ticker": ticker,
            "sector": sector,
            "page": page,
            "page_size": page_size,
        },
    )


def feed_summary(hours: int = 24) -> dict[str, Any]:
    """피드 요약 (감성 분포, top 티커/섹터)."""
    return _get("/feed/summary", {"hours": hours})


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def get_article(news_id: int) -> dict[str, Any]:
    """기사 단건 조회."""
    return _get(f"/news/{news_id}")


def list_topics() -> list[dict[str, Any]] | dict[str, Any]:
    """토픽 목록."""
    result = _get("/news/topics")
    # get_topics 는 list 반환. 오류 시 dict 반환.
    return result if isinstance(result, (list, dict)) else []


# ---------------------------------------------------------------------------
# Base URL helper (for rendering links)
# ---------------------------------------------------------------------------


def api_base() -> str:
    return _API_BASE
