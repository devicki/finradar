#!/usr/bin/env python3
"""Dry-run the clusterer with different parameters and report cluster-size
distribution.  Does NOT mutate ``news_items`` — uses ``write=False``.

Usage:
    docker compose exec celery_worker python /app/scripts/cluster_dryrun.py
    docker compose exec celery_worker python /app/scripts/cluster_dryrun.py \\
        --window-days 7 --same-lang-cosine 0.87 --overlap-min 3

Prints current (DB) distribution and the dry-run distribution side by side
so you can tell whether the new parameters actually fix the long tail.
"""
from __future__ import annotations

import argparse
from collections import Counter

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from finradar.clustering import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_KNN_CANDIDATES,
    DEFAULT_MIN_BODY_JACCARD,
    DEFAULT_SAME_LANG_COSINE,
    DEFAULT_TITLE_OVERLAP_MIN,
    DEFAULT_TITLE_OVERLAP_RATIO,
    DEFAULT_WINDOW_DAYS,
    cluster_recent_articles,
)
from finradar.config import get_settings


_settings = get_settings()
_sync_url = _settings.database_url.replace("+asyncpg", "+psycopg2")
_sync_engine = create_engine(_sync_url, pool_pre_ping=True)
SyncSession = sessionmaker(bind=_sync_engine, autoflush=False, autocommit=False)


def _current_distribution(session) -> list[tuple[int, int]]:
    rows = session.execute(
        text(
            """
            SELECT cluster_size, COUNT(*) AS n
            FROM (
              SELECT cluster_rep_id, MAX(cluster_size) AS cluster_size
              FROM news_items
              WHERE cluster_rep_id IS NOT NULL
              GROUP BY cluster_rep_id
            ) s
            GROUP BY cluster_size
            ORDER BY cluster_size DESC
            """
        )
    ).all()
    return [(r.cluster_size, r.n) for r in rows]


def _dryrun_distribution(assignments: list[dict]) -> list[tuple[int, int]]:
    by_rep: dict[int, int] = {}
    for r in assignments:
        rep = r.get("cluster_rep_id")
        if rep is None:
            continue
        by_rep[rep] = max(by_rep.get(rep, 0), r["cluster_size"])
    counter = Counter(by_rep.values())
    return sorted(counter.items(), key=lambda t: -t[0])


def _format_distribution(title: str, dist: list[tuple[int, int]]) -> str:
    lines = [f"--- {title} ---"]
    if not dist:
        lines.append("  (no non-singleton clusters)")
        return "\n".join(lines)
    largest = max(s for s, _ in dist)
    over_50 = sum(n for s, n in dist if s >= 50)
    over_100 = sum(n for s, n in dist if s >= 100)
    total_non_singleton = sum(n for _, n in dist)
    lines.append(
        f"  non-singleton clusters={total_non_singleton}  "
        f"largest={largest}  clusters(>=50)={over_50}  clusters(>=100)={over_100}"
    )
    for s, n in dist[:15]:
        lines.append(f"  size={s:5d}  clusters={n}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    p.add_argument("--threshold", type=float, default=DEFAULT_COSINE_THRESHOLD,
                   help="Baseline cosine used in the SQL KNN pull.")
    p.add_argument("--same-lang-cosine", type=float, default=DEFAULT_SAME_LANG_COSINE,
                   help="Tighter cosine floor for same-language pairs.")
    p.add_argument("--knn", type=int, default=DEFAULT_KNN_CANDIDATES)
    p.add_argument("--overlap-min", type=int, default=DEFAULT_TITLE_OVERLAP_MIN)
    p.add_argument("--overlap-ratio", type=float, default=DEFAULT_TITLE_OVERLAP_RATIO)
    p.add_argument("--body-jaccard", type=float, default=DEFAULT_MIN_BODY_JACCARD)
    p.add_argument("--no-stopwords", action="store_true")
    args = p.parse_args()

    print(
        f"Dry-run parameters: window={args.window_days}d, "
        f"cos>=({args.threshold:.2f} cross, {args.same_lang_cosine:.2f} same-lang), "
        f"overlap>={args.overlap_min} (ratio>={args.overlap_ratio:.2f}), "
        f"body_jaccard>={args.body_jaccard:.2f}, "
        f"stopwords={'off' if args.no_stopwords else 'on'}"
    )

    with SyncSession() as session:
        current = _current_distribution(session)
        print(_format_distribution("Current DB distribution", current))

        result = cluster_recent_articles(
            session,
            window_days=args.window_days,
            threshold=args.threshold,
            same_lang_cosine=args.same_lang_cosine,
            knn_candidates=args.knn,
            title_overlap_min=args.overlap_min,
            title_overlap_ratio=args.overlap_ratio,
            min_body_jaccard=args.body_jaccard,
            apply_stopwords=not args.no_stopwords,
            write=False,
        )

        print()
        print(_format_distribution("Dry-run distribution", _dryrun_distribution(result.assignments)))
        print()
        print("Summary:")
        print(f"  articles_considered   = {result.articles_considered}")
        print(f"  edges_found           = {result.edges_found}")
        print(f"  edges_after_filters   = {result.edges_after_overlap}")
        print(f"  non_singleton_clusters= {result.non_singleton_clusters}")
        print(f"  largest_cluster_size  = {result.largest_cluster_size}")
        print(f"  rejected_by_reason    = {result.rejected_by_reason}")
        print(f"  elapsed_sec           = {result.elapsed_sec:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
