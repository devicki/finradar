"""Quick RSS collection test."""
import asyncio
from finradar.collectors.rss_collector import RSSCollector


async def main():
    async with RSSCollector() as collector:
        articles = await collector.safe_collect()
        print(f"총 수집: {len(articles)}건\n")
        for a in articles[:5]:
            print(f"  [{a.source_type}] {a.title[:80]}")
            print(f"    URL: {a.url}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
