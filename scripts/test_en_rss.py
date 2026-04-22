"""영문 경제 RSS 후보 검증.

각 후보 URL에 대해:
  1. HTTP fetch (httpx, 브라우저 UA) — 200 OK 여부
  2. feedparser 파싱 성공 여부 + 총 entry 수
  3. 평균 summary 길이 (짧으면 body-fetch fallback 필요)
  4. 최근 24시간 이내 새 기사 수 (피드가 살아있는지)
  5. 중복 URL 여부

Usage:
    docker compose exec app python /app/scripts/test_en_rss.py
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import feedparser
import httpx


# ---------------------------------------------------------------------------
# 후보 피드 (시장/경제 중심)
# ---------------------------------------------------------------------------
#
# URL이 가끔 바뀌는 매체가 많아, 동일 매체에 여러 경로를 섞어 등록했다.
# 하나라도 통과하면 사용 결정하는 전략.

CANDIDATES: list[dict[str, str]] = [
    # --- Reuters / WSJ / Barron's ---
    {"name": "WSJ Markets",          "url": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain"},
    {"name": "WSJ World News",       "url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml"},
    {"name": "MarketWatch Top",      "url": "http://feeds.marketwatch.com/marketwatch/topstories/"},
    {"name": "MarketWatch RealTime", "url": "http://feeds.marketwatch.com/marketwatch/realtimeheadlines/"},
    {"name": "Barron's Top",         "url": "https://www.barrons.com/xml/rss/3_7011.xml"},

    # --- Bloomberg sections (already have markets) ---
    {"name": "Bloomberg Technology", "url": "https://feeds.bloomberg.com/technology/news.rss"},
    {"name": "Bloomberg Politics",   "url": "https://feeds.bloomberg.com/politics/news.rss"},

    # --- CNBC sections ---
    {"name": "CNBC World Markets",   "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html"},
    {"name": "CNBC Finance",         "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html"},
    {"name": "CNBC Economy",         "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html"},

    # --- FT sections ---
    {"name": "FT Companies",         "url": "https://www.ft.com/companies?format=rss"},
    {"name": "FT Markets",           "url": "https://www.ft.com/markets?format=rss"},
    {"name": "FT Global Economy",    "url": "https://www.ft.com/global-economy?format=rss"},

    # --- Nikkei Asia / Investing / Seeking Alpha / Business Insider ---
    {"name": "Nikkei Asia Business", "url": "https://asia.nikkei.com/rss/feed/nar"},
    {"name": "Investing.com News",   "url": "https://www.investing.com/rss/news.rss"},
    {"name": "Investing.com Stock",  "url": "https://www.investing.com/rss/news_25.rss"},
    {"name": "Investing.com Forex",  "url": "https://www.investing.com/rss/news_1.rss"},
    {"name": "Seeking Alpha Currents", "url": "https://seekingalpha.com/market_currents.xml"},
    {"name": "Seeking Alpha Feed",   "url": "https://seekingalpha.com/feed.xml"},
    {"name": "Business Insider",     "url": "https://www.businessinsider.com/rss"},
    {"name": "Business Insider Markets", "url": "https://markets.businessinsider.com/rss/news"},

    # --- Reuters direct attempts (Google News proxy already works for us) ---
    {"name": "Reuters Business via GN", "url": "https://news.google.com/rss/search?q=when:24h+site:reuters.com+business&hl=en"},

    # --- AP / Associated Press ---
    {"name": "AP Business",          "url": "https://apnews.com/hub/business?utm_source=rss"},

    # --- The Economist ---
    {"name": "Economist Finance",    "url": "https://www.economist.com/finance-and-economics/rss.xml"},
    {"name": "Economist Business",   "url": "https://www.economist.com/business/rss.xml"},
]


_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _probe(url: str, timeout: float = 15.0) -> tuple[int, bytes | None, str | None]:
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
        ) as c:
            r = c.get(url)
            return r.status_code, r.content, None
    except httpx.HTTPError as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def _analyse(name: str, url: str) -> dict:
    status, raw, err = _probe(url)
    out: dict = {"name": name, "url": url, "status": status}
    if err:
        out.update(ok=False, reason=err)
        return out
    if status != 200:
        out.update(ok=False, reason=f"HTTP {status}")
        return out

    try:
        parsed = feedparser.parse(raw)
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, reason=f"parse error: {exc}")
        return out

    entries = parsed.entries or []
    out["entries"] = len(entries)
    if not entries:
        out.update(ok=False, reason="no entries")
        return out

    # Recent-ness: how many entries within 48h
    now_utc = datetime.now(tz=timezone.utc)
    cutoff = now_utc - timedelta(hours=48)
    recent = 0
    summary_lens: list[int] = []
    for e in entries[:40]:
        # published_parsed or updated_parsed
        struct = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        if struct:
            try:
                dt = datetime(*struct[:6], tzinfo=timezone.utc)
                if dt >= cutoff:
                    recent += 1
            except (TypeError, ValueError):
                pass
        summary = getattr(e, "summary", None) or getattr(e, "description", "") or ""
        summary_lens.append(len(summary))

    out["recent48h"] = recent
    out["avg_summary_len"] = int(sum(summary_lens) / len(summary_lens)) if summary_lens else 0
    out["sample_title"] = (entries[0].get("title") or "")[:70]
    out["ok"] = recent >= 1 and len(entries) >= 3
    if not out["ok"]:
        out["reason"] = f"stale (recent48h={recent}, entries={len(entries)})"
    return out


def main() -> None:
    print(f"Testing {len(CANDIDATES)} English RSS candidates...\n")
    results: list[dict] = []
    for cand in CANDIDATES:
        t0 = time.perf_counter()
        res = _analyse(cand["name"], cand["url"])
        res["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        results.append(res)
        mark = "✅" if res.get("ok") else "❌"
        if res.get("ok"):
            print(
                f"{mark} {res['name']:36s} entries={res['entries']:>3} "
                f"recent48h={res['recent48h']:>3} avg_sum={res['avg_summary_len']:>4}  "
                f"{res['elapsed_ms']:>4}ms"
            )
        else:
            print(f"{mark} {res['name']:36s} {res.get('reason', '-')}")

    print()
    ok = [r for r in results if r.get("ok")]
    print(f"✅ {len(ok)}/{len(results)} candidates usable")
    print()
    print("=== Usable feeds (copy-paste into rss_collector.py) ===")
    for r in ok:
        print(f'{{"url": "{r["url"]}", "name": "{r["name"]}", "language": "en"}},')


if __name__ == "__main__":
    main()
