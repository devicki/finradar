"""Test Korean RSS feeds to check which ones are alive."""
import asyncio
import httpx
import feedparser

KR_FEEDS = [
    # 한국경제
    {"url": "https://www.hankyung.com/feed/economy", "name": "한국경제 경제"},
    {"url": "https://www.hankyung.com/feed/finance", "name": "한국경제 증권"},
    {"url": "https://www.hankyung.com/feed/international", "name": "한국경제 국제"},
    # 매일경제
    {"url": "http://file.mk.co.kr/news/rss/rss_30000001.xml", "name": "매일경제 헤드라인"},
    {"url": "http://file.mk.co.kr/news/rss/rss_50200011.xml", "name": "매일경제 증권"},
    {"url": "http://file.mk.co.kr/news/rss/rss_30100041.xml", "name": "매일경제 경제"},
    # 조선비즈
    {"url": "http://biz.chosun.com/site/data/rss/rss.xml", "name": "조선비즈 전체"},
    {"url": "http://biz.chosun.com/site/data/rss/policybank.xml", "name": "조선비즈 정책/금융"},
    {"url": "http://biz.chosun.com/site/data/rss/market.xml", "name": "조선비즈 마켓"},
    # 서울경제
    {"url": "http://rss.hankooki.com/economy/sk_economy.xml", "name": "서울경제 경제"},
    {"url": "http://rss.hankooki.com/economy/sk_stock.xml", "name": "서울경제 증권"},
    {"url": "http://rss.hankooki.com/economy/sk_main.xml", "name": "서울경제 전체"},
    # 파이낸셜뉴스
    {"url": "http://www.fnnews.com/rss/fn_realnews_all.xml", "name": "파이낸셜뉴스 전체"},
    {"url": "http://www.fnnews.com/rss/fn_realnews_finance.xml", "name": "파이낸셜뉴스 금융"},
    {"url": "http://www.fnnews.com/rss/fn_realnews_stock.xml", "name": "파이낸셜뉴스 증권"},
]


async def test_feed(client: httpx.AsyncClient, feed: dict) -> dict:
    url = feed["url"]
    name = feed["name"]
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            return {"name": name, "status": "FAIL", "reason": f"HTTP {resp.status_code}", "count": 0}

        parsed = feedparser.parse(resp.content)
        count = len(parsed.entries)

        if count == 0:
            return {"name": name, "status": "EMPTY", "reason": "No entries", "count": 0}

        sample = parsed.entries[0].get("title", "no title")
        return {"name": name, "status": "OK", "reason": f"sample: {sample[:60]}", "count": count}

    except Exception as e:
        return {"name": name, "status": "FAIL", "reason": str(e)[:80], "count": 0}


async def main():
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        tasks = [test_feed(client, f) for f in KR_FEEDS]
        results = await asyncio.gather(*tasks)

    print(f"{'상태':<6} {'건수':>4}  {'매체':<25} {'비고'}")
    print("-" * 90)
    for r in results:
        print(f"{r['status']:<6} {r['count']:>4}  {r['name']:<25} {r['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
