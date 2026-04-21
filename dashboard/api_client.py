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
    personalize: bool = False,
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
            "personalize": personalize,
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
# URL ingest
# ---------------------------------------------------------------------------


def ingest_url(url: str, language: str | None = None, force_pdf: bool = False) -> dict[str, Any]:
    """POST an arbitrary URL to /ingest/url for on-demand scraping.

    Returns the server response verbatim (includes status, news_id, message…).
    """
    return _post(
        "/ingest/url",
        {
            "url": url,
            "language": language,
            "force_pdf": force_pdf,
        },
    )


# ---------------------------------------------------------------------------
# Feedback (like / dislike / bookmark / dismiss)
# ---------------------------------------------------------------------------


def submit_feedback(news_id: int, action: str) -> dict[str, Any]:
    """POST /feedback — upsert a feedback row for the current user."""
    return _post("/feedback/", {"news_id": news_id, "action": action})


def delete_feedback(news_id: int, action: str) -> dict[str, Any]:
    """DELETE /feedback/{news_id}/{action} — toggle off."""
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.delete(f"{_API_PREFIX}/feedback/{news_id}/{action}")
            if r.status_code == 204:
                return {"status": "ok"}
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        return {"error": str(e)}


def feedback_status_batch(news_ids: list[int]) -> dict[int, list[str]]:
    """POST /feedback/status/batch → {news_id: [actions]} (best-effort)."""
    if not news_ids:
        return {}
    resp = _post("/feedback/status/batch", {"news_ids": news_ids})
    if "error" in resp:
        return {}
    # JSON object keys come back as strings; coerce back to int for convenience.
    return {int(k): v for k, v in (resp.get("states") or {}).items()}


def list_bookmarks(page: int = 1, page_size: int = 20) -> dict[str, Any]:
    return _get("/feedback/bookmarks", {"page": page, "page_size": page_size})


def list_dismissed(page: int = 1, page_size: int = 20) -> dict[str, Any]:
    return _get("/feedback/dismissed", {"page": page, "page_size": page_size})


def get_affinity() -> dict[str, Any]:
    """Top / bottom sectors + tickers the engine learned from my feedback."""
    return _get("/feedback/affinity")


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
    date_from: str | None = None,
    date_to: str | None = None,
    dedup: bool = True,
    sort: str = "latest",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """뉴스 피드 (Latest News 페이지).

    Args:
        sort: latest | cluster_size | sentiment_strength
        dedup: True → 클러스터 대표만 (기본), False → 모두
    """
    return _get(
        "/feed/",
        {
            "language": language,
            "source_type": source_type,
            "sentiment_label": sentiment_label,
            "ticker": ticker,
            "sector": sector,
            "date_from": date_from,
            "date_to": date_to,
            "dedup": "true" if dedup else "false",
            "sort": sort,
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
