"""
finradar.tasks
~~~~~~~~~~~~~~

Celery application and task definitions for FinRadar.

Exports:
    celery_app       — the configured Celery application instance
    collect_all_news — periodic collection task (runs every N minutes via Beat)
    process_pending_news — AI processing task (sentiment + embeddings)
    enrich_with_llm  — per-item LLM enrichment (summary, translation, metadata)
    reconcile_pending_llm — safety-net task that re-queues orphaned LLM work
"""

from finradar.tasks.celery_app import celery_app
from finradar.tasks.collection_tasks import (
    collect_all_news,
    enrich_with_llm,
    process_pending_news,
    reconcile_pending_llm,
)

__all__ = [
    "celery_app",
    "collect_all_news",
    "process_pending_news",
    "enrich_with_llm",
    "reconcile_pending_llm",
]
