"""한경/서울경제 등 '제목만 피드' 소스의 기존 기사 본문 재보강.

RSS 수집 단계의 trafilatura body-fetch가 적용되기 전에 수집된 한국어 기사
(title == ai_summary 인 기사) 대상으로 URL 재접속 → 본문 추출 → summary
업데이트 → LLM enrich 재트리거.

안전장치:
  - 본문이 100자 미만이면 skip (실제로 짧은 속보 기사)
  - 본문과 기존 title이 거의 같으면 skip (의미 없는 업데이트)
  - HTTP 실패 / trafilatura 실패 시 조용히 skip

Usage:
    docker compose exec app python /app/scripts/repair_ko_body_fetch.py --dry-run
    docker compose exec app python /app/scripts/repair_ko_body_fetch.py
    docker compose exec app python /app/scripts/repair_ko_body_fetch.py --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import httpx
import trafilatura
from sqlalchemy import func, select, update

from finradar.collectors.rss_collector import clean_rss_text
from finradar.models.news import NewsItem
from finradar.tasks.collection_tasks import SyncSessionLocal

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_TIMEOUT = 15.0
_MIN_BODY_LEN = 100
_CONCURRENCY = 8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Body-fetch repair for legacy Korean articles")
    p.add_argument("--limit", type=int, default=None, help="최대 처리 건수")
    p.add_argument("--dry-run", action="store_true", help="DB 업데이트/큐잉 없이 미리보기")
    return p.parse_args()


def _fetch_targets(limit: int | None) -> list[dict[str, Any]]:
    """title == ai_summary 인 한국어 기사 목록 조회."""
    with SyncSessionLocal() as session:
        stmt = (
            select(
                NewsItem.id,
                NewsItem.url,
                NewsItem.title,
                NewsItem.source_url,
            )
            .where(
                NewsItem.language == "ko",
                NewsItem.ai_summary.is_not(None),
                NewsItem.title == NewsItem.ai_summary,
            )
            .order_by(NewsItem.source_url, NewsItem.id)
        )
        if limit:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).all()
    return [
        {"id": r.id, "url": r.url, "title": r.title, "source_url": r.source_url}
        for r in rows
    ]


async def _fetch_body(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    target: dict[str, Any],
) -> tuple[dict[str, Any], str | None, str]:
    """단일 기사 본문 추출.

    Returns:
        (target, body_or_None, status)  — status: "ok", "short", "duplicate",
        "fetch_err", "extract_err"
    """
    async with semaphore:
        try:
            resp = await client.get(
                target["url"],
                timeout=_TIMEOUT,
                headers={"User-Agent": _UA},
                follow_redirects=True,
            )
            resp.raise_for_status()
            html_text = resp.text
        except (httpx.HTTPError, httpx.RequestError) as exc:
            return target, None, f"fetch_err: {type(exc).__name__}"

    try:
        body = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as exc:  # noqa: BLE001
        return target, None, f"extract_err: {type(exc).__name__}"

    if not body:
        return target, None, "extract_err: empty"

    cleaned = clean_rss_text(body, source_hint=target["url"])
    if len(cleaned) < _MIN_BODY_LEN:
        return target, None, "short"

    # 본문이 title과 거의 같으면 의미 없음 (유사도 대충 체크)
    if cleaned.strip() == target["title"].strip():
        return target, None, "duplicate"

    return target, cleaned, "ok"


async def _fetch_all(targets: list[dict[str, Any]]) -> list[tuple[dict, str | None, str]]:
    """전체 타겟 병렬 fetch."""
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *(_fetch_body(client, semaphore, t) for t in targets),
            return_exceptions=False,
        )


def _apply_updates(
    results: list[tuple[dict, str | None, str]], dry_run: bool
) -> tuple[int, int]:
    """성공한 fetch 결과를 DB에 반영하고 enrich 재큐잉.

    Returns:
        (updated_count, queued_count)
    """
    # Stage 1: DB 업데이트만 (모두 commit 후 worker가 NULL을 볼 수 있게)
    updated_ids: list[int] = []
    if not dry_run:
        with SyncSessionLocal() as session:
            for target, body, status in results:
                if status != "ok" or not body:
                    continue
                session.execute(
                    update(NewsItem)
                    .where(NewsItem.id == target["id"])
                    .values(
                        summary=body,
                        ai_summary=None,
                        translated_title=None,
                        translated_summary=None,
                        tickers=None,
                        sectors=None,
                        llm_enrich_attempts=0,
                        llm_last_attempt_at=None,
                        updated_at=func.now(),
                    )
                )
                updated_ids.append(target["id"])
            session.commit()  # <- Stage 2 이전에 반드시 커밋
    else:
        updated_ids = [t["id"] for t, body, s in results if s == "ok" and body]

    # Stage 2: commit이 끝난 뒤 enrich_with_llm 큐잉
    queued = 0
    if not dry_run:
        from finradar.tasks.collection_tasks import enrich_with_llm

        for news_id in updated_ids:
            enrich_with_llm.apply_async(args=[news_id], queue="finradar.llm")
            queued += 1
    else:
        queued = len(updated_ids)

    return len(updated_ids), queued


def main() -> int:
    args = _parse_args()
    targets = _fetch_targets(args.limit)
    print(f"[repair] 대상: language='ko', title == ai_summary, 총 {len(targets)}건")

    if not targets:
        print("[repair] 처리할 기사 없음.")
        return 0

    # 소스별 분포
    from collections import Counter

    by_source = Counter(t["source_url"] for t in targets)
    print("[repair] 소스별 분포:")
    for src, cnt in by_source.most_common():
        print(f"  {cnt:4d}  {src}")

    print(f"[repair] 병렬 fetch 시작 (concurrency={_CONCURRENCY})...")
    results = asyncio.run(_fetch_all(targets))

    # 상태별 집계
    status_counts = Counter(status for _, _, status in results)
    print("[repair] fetch 결과:")
    for status, cnt in status_counts.most_common():
        print(f"  {cnt:4d}  {status}")

    updated, queued = _apply_updates(results, args.dry_run)
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"[repair] DB 업데이트: {updated}건{suffix}")
    print(f"[repair] enrich_with_llm 큐잉: {queued}건{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
