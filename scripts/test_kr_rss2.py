"""Test additional Korean RSS feeds."""
import asyncio
import httpx
import feedparser

KR_FEEDS = [
    # 이데일리
    {"url": "http://rss.edaily.co.kr/edaily_news.xml", "name": "이데일리 전체"},
    {"url": "http://rss.edaily.co.kr/stock_news.xml", "name": "이데일리 증권"},
    # 머니투데이
    {"url": "http://rss.mt.co.kr/mt_news.xml", "name": "머니투데이 전체"},
    {"url": "http://rss.mt.co.kr/mt_column.xml", "name": "머니투데이 칼럼"},
    # 이투데이
    {"url": "https://rss.etoday.co.kr/eto/etoday_news_all.xml", "name": "이투데이 전체"},
    {"url": "https://rss.etoday.co.kr/eto/market_news.xml", "name": "이투데이 마켓"},
    {"url": "https://rss.etoday.co.kr/eto/finance_news.xml", "name": "이투데이 금융"},
    {"url": "https://rss.etoday.co.kr/eto/economy_news.xml", "name": "이투데이 경제"},
    # 연합뉴스경제TV
    {"url": "https://www.yonhapnewseconomytv.com/rss/allArticle.xml", "name": "연합뉴스경제TV 전체"},
    {"url": "https://www.yonhapnewseconomytv.com/rss/clickTop.xml", "name": "연합뉴스경제TV 인기"},
    # 연합뉴스 (기존 Gist에서)
    {"url": "http://www.yonhapnews.co.kr/RSS/economy.xml", "name": "연합뉴스 경제"},
    {"url": "http://www.yonhapnews.co.kr/RSS/finance.xml", "name": "연합뉴스 금융"},
    {"url": "http://www.yonhapnews.co.kr/RSS/international.xml", "name": "연합뉴스 국제"},
    # 헤럴드경제
    {"url": "http://biz.heraldcorp.com/common/rss_xml.php?ct=economy", "name": "헤럴드경제 경제"},
    {"url": "http://biz.heraldcorp.com/common/rss_xml.php?ct=stock", "name": "헤럴드경제 증권"},
    # 아시아경제
    {"url": "https://www.asiae.co.kr/rss/allnews.xml", "name": "아시아경제 전체"},
    # 전자신문 (IT/산업)
    {"url": "https://rss.etnews.com/Section901.xml", "name": "전자신문 전체"},
    # 매일경제 (새 URL 시도)
    {"url": "https://www.mk.co.kr/rss/30000001/", "name": "매일경제 헤드라인(신규)"},
    {"url": "https://www.mk.co.kr/rss/50200011/", "name": "매일경제 증권(신규)"},
    {"url": "https://www.mk.co.kr/rss/30100041/", "name": "매일경제 경제(신규)"},
    # 서울경제 (새 URL 시도)
    {"url": "https://www.sedaily.com/rss", "name": "서울경제(신규)"},
    {"url": "https://www.sedaily.com/rss/Economy", "name": "서울경제 경제(신규)"},
    # 파이낸셜뉴스 (새 URL 시도)
    {"url": "https://www.fnnews.com/rss/fn_realnews_all.xml", "name": "파이낸셜뉴스(HTTPS)"},
    # 조선비즈 (새 URL 시도)
    {"url": "https://biz.chosun.com/site/data/rss/rss.xml", "name": "조선비즈(HTTPS)"},
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
        return {"name": name, "status": "OK", "reason": f"sample: {sample[:50]}", "count": count}

    except Exception as e:
        return {"name": name, "status": "FAIL", "reason": str(e)[:80], "count": 0}


async def main():
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        tasks = [test_feed(client, f) for f in KR_FEEDS]
        results = await asyncio.gather(*tasks)

    print(f"{'상태':<6} {'건수':>4}  {'매체':<28} {'비고'}")
    print("-" * 100)
    ok_count = 0
    for r in results:
        print(f"{r['status']:<6} {r['count']:>4}  {r['name']:<28} {r['reason']}")
        if r["status"] == "OK":
            ok_count += 1
    print(f"\n살아있는 피드: {ok_count}개 / 총 {len(results)}개")


if __name__ == "__main__":
    asyncio.run(main())
