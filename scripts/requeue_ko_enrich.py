"""ai_summary가 NULL이지만 summary는 채워져 있는 한국어 기사의 enrich 큐잉 보충.

body-fetch repair 스크립트의 이전 실행에서 race condition으로 skip된
케이스를 수습하는 용도.

Usage:
    docker compose exec app python /app/scripts/requeue_ko_enrich.py --dry-run
    docker compose exec app python /app/scripts/requeue_ko_enrich.py
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import func, select

from finradar.models.news import NewsItem
from finradar.tasks.collection_tasks import SyncSessionLocal


_TARGET_SOURCES = (
    "https://www.hankyung.com/feed/economy",
    "https://www.hankyung.com/feed/finance",
    "https://www.hankyung.com/feed/international",
    "https://www.sedaily.com/rss/Economy",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with SyncSessionLocal() as session:
        rows = session.execute(
            select(NewsItem.id)
            .where(
                NewsItem.language == "ko",
                NewsItem.ai_summary.is_(None),
                NewsItem.summary.is_not(None),
                func.length(NewsItem.summary) >= 100,
                NewsItem.source_url.in_(_TARGET_SOURCES),
            )
        ).scalars().all()

    print(f"[requeue] 대상: {len(rows)}건")
    if args.dry_run:
        print(f"[requeue] (dry-run) 큐잉 생략")
        return 0

    from finradar.tasks.collection_tasks import enrich_with_llm

    for news_id in rows:
        enrich_with_llm.apply_async(args=[news_id], queue="finradar.llm")

    print(f"[requeue] enrich_with_llm 큐잉 완료: {len(rows)}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
