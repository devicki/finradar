"""즉시 클러스터링 실행 (백필용).

Celery Beat의 30분 주기를 기다리지 않고 바로 cluster_recent_articles() 실행.
새 모델 배포 후 기존 기사에 클러스터 ID를 채우는 용도.

Usage:
    docker compose exec celery_worker python /app/scripts/cluster_now.py
    docker compose exec celery_worker python /app/scripts/cluster_now.py --window 7 --threshold 0.80
"""
from __future__ import annotations

import argparse
import sys

from finradar.clustering import cluster_recent_articles
from finradar.tasks.collection_tasks import SyncSessionLocal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=7, help="윈도우 일수 (기본 7일)")
    ap.add_argument(
        "--threshold", type=float, default=0.80, help="cosine similarity threshold (기본 0.80)"
    )
    ap.add_argument(
        "--knn", type=int, default=30, help="per-article KNN candidate 상한 (기본 30)"
    )
    args = ap.parse_args()

    print(
        f"[cluster_now] window={args.window}일, threshold={args.threshold}, "
        f"knn={args.knn}"
    )

    with SyncSessionLocal() as session:
        result = cluster_recent_articles(
            session,
            window_days=args.window,
            threshold=args.threshold,
            knn_candidates=args.knn,
        )

    print(f"[cluster_now] 완료:")
    print(f"  대상 기사:              {result.articles_considered:,}건")
    print(f"  edge (cosine only):    {result.edges_found:,}")
    print(f"  edge after overlap:    {result.edges_after_overlap:,}")
    print(f"  2건 이상 클러스터:      {result.non_singleton_clusters}개")
    print(f"  최대 클러스터 크기:     {result.largest_cluster_size}")
    print(f"  업데이트 row:           {result.updated_rows:,}")
    print(f"  소요:                   {result.elapsed_sec:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
