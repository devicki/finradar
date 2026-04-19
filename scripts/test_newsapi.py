"""Quick NewsAPI collection test."""
import asyncio
from finradar.config import get_settings
from finradar.collectors.newsapi_collector import NewsAPICollector


async def main():
    settings = get_settings()
    if not settings.newsapi_key:
        print("NEWSAPI_KEY가 .env에 설정되지 않았습니다.")
        return

    print(f"NewsAPI Key: {settings.newsapi_key[:8]}...{settings.newsapi_key[-4:]}")
    print()

    async with NewsAPICollector(api_key=settings.newsapi_key) as collector:
        articles = await collector.safe_collect()
        print(f"총 수집: {len(articles)}건\n")
        for a in articles[:5]:
            print(f"  [{a.source_type}] {a.title[:80]}")
            print(f"    URL: {a.url}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
