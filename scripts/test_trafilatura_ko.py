"""
한경/서울경제 본문 추출 속도 및 품질 테스트.

Usage:
    docker compose exec app python /app/scripts/test_trafilatura_ko.py
"""
from __future__ import annotations

import asyncio
import time

import httpx
import trafilatura


SAMPLE_URLS = [
    # 한국경제
    "https://www.hankyung.com/article/202604202187i",
    "https://www.hankyung.com/article/2026042018356",
    "https://www.hankyung.com/article/202604202055i",
    # 서울경제
    "https://www.sedaily.com/article/20034495",
    "https://www.sedaily.com/article/20034488",
    "https://www.sedaily.com/article/20034489",
]


async def fetch_one(
    client: httpx.AsyncClient, url: str
) -> tuple[str, float, float, int, str]:
    """Fetch URL and extract body with trafilatura.

    Returns:
        (url, fetch_ms, extract_ms, body_len, body_preview)
    """
    t0 = time.perf_counter()
    try:
        resp = await client.get(
            url,
            timeout=15.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                )
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        return url, -1.0, -1.0, 0, f"FETCH_ERROR: {exc}"
    t1 = time.perf_counter()
    fetch_ms = (t1 - t0) * 1000.0

    html = resp.text
    t2 = time.perf_counter()
    body = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    t3 = time.perf_counter()
    extract_ms = (t3 - t2) * 1000.0

    body = body or ""
    preview = body[:150].replace("\n", " ")
    return url, fetch_ms, extract_ms, len(body), preview


async def main() -> None:
    async with httpx.AsyncClient() as client:
        # Concurrent fetch (동시성 6)
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(fetch_one(client, url) for url in SAMPLE_URLS)
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n=== Trafilatura 본문 추출 테스트 ({len(SAMPLE_URLS)}건 동시) ===\n")
    print(f"{'URL':<55} {'fetch':>8} {'extract':>8} {'body':>8}")
    print("-" * 90)
    for url, fetch_ms, extract_ms, body_len, _preview in results:
        short_url = url.replace("https://www.", "")
        fetch_str = f"{fetch_ms:.0f}ms" if fetch_ms >= 0 else "ERR"
        extract_str = f"{extract_ms:.1f}ms" if extract_ms >= 0 else "ERR"
        print(f"{short_url:<55} {fetch_str:>8} {extract_str:>8} {body_len:>6}자")

    print(f"\n전체 동시 처리 시간: {total_ms:.0f}ms")
    print()

    print("=== 본문 프리뷰 (앞 150자) ===\n")
    for url, _f, _e, body_len, preview in results:
        short_url = url.replace("https://www.", "")
        print(f"[{short_url}] ({body_len}자)")
        print(f"  {preview}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
