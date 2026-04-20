"""전체 기사 재임베딩 스크립트.

임베딩 모델을 교체한 후 기존 DB의 `embedding` 컬럼을 새 모델로 재생성한다.
`sentiment` / `ai_summary` 등 다른 필드는 건드리지 않는다.

- settings.embedding_model 을 그대로 사용 (env로 이미 새 모델 지정됨)
- GPU가 있으면 자동으로 cuda 로드 (CPU fallback 지원)
- 임베딩 텍스트: title + summary (둘 중 있는 것 조합)
- 배치 크기: settings.embedding_batch_size (기본 64)

Usage (GPU 필요 — celery_worker 컨테이너에서):
    docker compose exec celery_worker python /app/scripts/reembed_all.py
    docker compose exec celery_worker python /app/scripts/reembed_all.py --dry-run
    docker compose exec celery_worker python /app/scripts/reembed_all.py --limit 100
"""
from __future__ import annotations

import argparse
import sys
import time

from sqlalchemy import select, update

from finradar.config import get_settings
from finradar.models.news import NewsItem
from finradar.processors.embeddings import EmbeddingGenerator
from finradar.tasks.collection_tasks import SyncSessionLocal


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="최대 처리 건수")
    p.add_argument("--dry-run", action="store_true", help="DB 쓰기 없이 샘플만 확인")
    p.add_argument(
        "--only-null",
        action="store_true",
        help="embedding이 NULL인 것만 재생성 (기본: 전체 덮어쓰기)",
    )
    return p.parse_args()


def _build_embed_text(item: NewsItem) -> str:
    """임베딩용 텍스트: title + summary."""
    parts = [item.title or ""]
    if item.summary:
        parts.append(item.summary)
    return " ".join(p for p in parts if p.strip())


def main() -> int:
    args = _parse_args()
    settings = get_settings()

    print(f"[reembed] 모델: {settings.embedding_model}")
    print(f"[reembed] 디바이스: {settings.local_model_device}")
    print(f"[reembed] 배치 크기: {settings.embedding_batch_size}")

    # --- 대상 조회 ---
    with SyncSessionLocal() as session:
        stmt = select(NewsItem).order_by(NewsItem.id)
        if args.only_null:
            stmt = stmt.where(NewsItem.embedding.is_(None))
        if args.limit:
            stmt = stmt.limit(args.limit)
        items = session.execute(stmt).scalars().all()

    if not items:
        print("[reembed] 처리할 기사 없음")
        return 0

    print(f"[reembed] 대상: {len(items)}건")

    # --- 모델 로드 ---
    t0 = time.perf_counter()
    embedder = EmbeddingGenerator(
        model_name=settings.embedding_model,
        device=settings.local_model_device,
    )
    # 워밍업 (첫 호출 시 모델 로드)
    _ = embedder.generate("warmup")
    print(f"[reembed] 모델 로드: {time.perf_counter() - t0:.1f}초")

    # --- 배치 재임베딩 ---
    batch_size = settings.embedding_batch_size
    total_embedded = 0
    total_written = 0
    t_start = time.perf_counter()

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [_build_embed_text(item) for item in batch]

        t_embed = time.perf_counter()
        vectors = embedder.generate_batch(texts)
        embed_sec = time.perf_counter() - t_embed
        total_embedded += len(batch)

        if args.dry_run:
            print(
                f"  [{i + len(batch):>5}/{len(items)}] "
                f"embed {len(batch)}건 {embed_sec*1000:.0f}ms (dry-run)"
            )
            continue

        # --- DB 업데이트 ---
        with SyncSessionLocal() as session:
            for item, vec in zip(batch, vectors):
                session.execute(
                    update(NewsItem)
                    .where(NewsItem.id == item.id)
                    .values(embedding=vec)
                )
            session.commit()
        total_written += len(batch)

        elapsed = time.perf_counter() - t_start
        rate = total_embedded / elapsed if elapsed > 0 else 0
        eta = (len(items) - total_embedded) / rate if rate > 0 else 0
        print(
            f"  [{i + len(batch):>5}/{len(items)}] "
            f"embed {embed_sec*1000:.0f}ms "
            f"rate={rate:.0f}건/s "
            f"ETA={eta:.0f}초"
        )

    total_sec = time.perf_counter() - t_start
    print(
        f"[reembed] 완료: 임베딩 {total_embedded}건 / "
        f"DB 쓰기 {total_written}건 / 총 {total_sec:.1f}초"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
