"""Manually trigger the news collection task."""
from finradar.tasks.collection_tasks import collect_all_news

result = collect_all_news.delay()
print(f"Task submitted: {result.id}")
print("Check worker logs: docker compose logs celery_worker --tail 30")
