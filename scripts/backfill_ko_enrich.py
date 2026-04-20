"""한국어 기사 백필: HTML 정리 + LLM enrich.

최근 N일치 한국어 기사 중 ai_summary가 비어있는 것들을 대상으로:
  1. 기존 DB의 title/summary를 clean_rss_text()로 재정리 (HTML 태그/엔티티 제거)
  2. enrich_with_llm Celery 태스크를 큐잉

기본값은 최근 7일. --days 로 조정 가능, --limit 로 상한.

Usage (docker exec):
    docker compose exec app python /app/scripts/backfill_ko_enrich.py --days 7 --dry-run
    docker compose exec app python /app/scripts/backfill_ko_enrich.py --days 7
    docker compose exec app python /app/scripts/backfill_ko_enrich.py --days 7 --limit 10
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update

from finradar.collectors.rss_collector import clean_rss_text
from finradar.models.news import NewsItem
from finradar.tasks.collection_tasks import SyncSessionLocal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Korean article enrichment backfill")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="대상 기사 조회 일수 (기본 7일)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="최대 처리 건수 (기본: 전체)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB 업데이트/Celery 큐잉 없이 대상만 출력",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="title/summary HTML 정리 단계 건너뛰기",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="enrich_with_llm 큐잉 단계 건너뛰기 (정리만)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    print(
        f"[backfill] 대상: language='ko', ai_summary IS NULL, "
        f"first_seen_at >= {cutoff.isoformat()}"
    )

    with SyncSessionLocal() as session:
        stmt = (
            select(NewsItem)
            .where(
                NewsItem.language == "ko",
                NewsItem.ai_summary.is_(None),
                NewsItem.first_seen_at >= cutoff,
            )
            .order_by(NewsItem.first_seen_at.desc())
        )
        if args.limit:
            stmt = stmt.limit(args.limit)

        items = session.execute(stmt).scalars().all()
        print(f"[backfill] 대상 기사: {len(items)}건")

        if not items:
            print("[backfill] 처리할 기사 없음.")
            return 0

        # ------------------------------------------------------------------
        # Stage 1: HTML 정리 (title + summary)
        # ------------------------------------------------------------------
        cleaned_count = 0
        if not args.skip_clean:
            for item in items:
                new_title = clean_rss_text(item.title, source_hint=item.source_url)
                new_summary = clean_rss_text(item.summary, source_hint=item.source_url)

                title_changed = new_title and new_title != item.title
                summary_changed = new_summary != (item.summary or "")

                if title_changed or summary_changed:
                    if not args.dry_run:
                        session.execute(
                            update(NewsItem)
                            .where(NewsItem.id == item.id)
                            .values(
                                title=new_title if title_changed else item.title,
                                summary=new_summary if summary_changed else item.summary,
                                updated_at=func.now(),
                            )
                        )
                    cleaned_count += 1

            if not args.dry_run:
                session.commit()
            print(
                f"[backfill] HTML 정리: {cleaned_count}건 "
                f"{'(dry-run: DB 변경 없음)' if args.dry_run else '업데이트 완료'}"
            )

        # ------------------------------------------------------------------
        # Stage 2: enrich_with_llm 큐잉
        # ------------------------------------------------------------------
        queued_count = 0
        if not args.skip_enrich:
            from finradar.tasks.collection_tasks import enrich_with_llm

            for item in items:
                if args.dry_run:
                    print(f"  [dry-run] would enqueue enrich_with_llm(news_id={item.id})")
                else:
                    enrich_with_llm.apply_async(
                        args=[item.id], queue="finradar.llm"
                    )
                queued_count += 1

            print(
                f"[backfill] enrich_with_llm 큐잉: {queued_count}건 "
                f"{'(dry-run: 큐잉 없음)' if args.dry_run else '완료'}"
            )

        print(f"[backfill] 완료. 총 대상 {len(items)}건 / 정리 {cleaned_count}건 / 큐잉 {queued_count}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
