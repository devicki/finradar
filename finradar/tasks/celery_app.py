"""
finradar.tasks.celery_app
~~~~~~~~~~~~~~~~~~~~~~~~~

Celery application factory and Beat schedule for FinRadar.

The app is importable as:

    from finradar.tasks.celery_app import celery_app

Beat schedule (configured via celery_app.conf.beat_schedule):
- collect-news-every-N-min  — triggers the full collection pipeline
- process-unprocessed-news  — runs the local-GPU AI processing loop
- reconcile-pending-llm     — re-queues LLM enrichment orphaned by restarts
"""

from __future__ import annotations

import logging

from celery import Celery
from celery.schedules import crontab

from finradar.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

celery_app = Celery(
    "finradar",
    broker=settings.redis_url,
    backend=settings.redis_url,
    # Explicit include so Beat can discover tasks without loading the whole
    # package via autodiscovery (avoids importing GPU models at Beat startup).
    include=["finradar.tasks.collection_tasks"],
)

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Reliability: workers ACK only after the task completes successfully
    task_acks_late=True,
    # One task at a time per worker slot — important for GPU memory management
    worker_prefetch_multiplier=1,
    # Allow tracking "STARTED" state (useful for monitoring long-running tasks)
    task_track_started=True,
    # Expire results after 24 hours; we don't need long-term task result storage
    result_expires=86_400,
    # Soft / hard time limits (seconds).  Sentiment + embedding batch should
    # complete well within 5 min; allow 10 min hard limit as safety net.
    task_soft_time_limit=300,
    task_time_limit=600,
    # Routing: all tasks go to the default queue unless overridden
    task_default_queue="finradar",
    task_queues={
        "finradar": {"exchange": "finradar", "routing_key": "finradar"},
        "finradar.llm": {"exchange": "finradar.llm", "routing_key": "finradar.llm"},
    },
    task_routes={
        # LLM enrichment is network-bound (cloud API) — separate queue so
        # GPU-intensive tasks aren't blocked by slow LLM responses.
        "finradar.tasks.collection_tasks.enrich_with_llm": {
            "queue": "finradar.llm",
        },
    },
)

# ---------------------------------------------------------------------------
# Beat schedule (periodic tasks)
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    # Main collection pipeline — every COLLECTION_INTERVAL_MINUTES minutes.
    # Default: 15 min → 900 seconds.
    "collect-news-every-15min": {
        "task": "finradar.tasks.collection_tasks.collect_all_news",
        "schedule": settings.collection_interval_minutes * 60,
        "options": {"queue": "finradar"},
    },
    # Local AI processing (sentiment analysis + embeddings) — every 5 minutes
    # so that items inserted between collection cycles get processed quickly.
    "process-unprocessed-news": {
        "task": "finradar.tasks.collection_tasks.process_pending_news",
        "schedule": 300,  # 5 minutes
        "options": {"queue": "finradar"},
    },
    # Safety net: re-queue LLM enrichment for items whose task was lost to a
    # restart, worker crash, or broker outage.  Attempt counter + debounce
    # window (in the task itself) prevent runaway token spend.
    "reconcile-pending-llm": {
        "task": "finradar.tasks.collection_tasks.reconcile_pending_llm",
        "schedule": 600,  # 10 minutes
        "options": {"queue": "finradar"},
    },
    # Story grouping — connected components over cosine similarity.
    # Runs every 30 minutes so new articles get their cluster assignment
    # without blowing CPU on every collection cycle.
    "cluster-news-every-30min": {
        "task": "finradar.tasks.collection_tasks.cluster_news",
        "schedule": 1800,  # 30 minutes
        "options": {"queue": "finradar"},
    },
    # X (Twitter) timeline ingest — pay-as-you-go at $0.005/read.
    # The task itself checks settings.x_enabled and returns immediately when
    # disabled, so the schedule stays registered even without API credentials.
    # Adjust cadence via X_COLLECT_INTERVAL_MIN (default 10 minutes).
    "collect-x-posts": {
        "task": "finradar.tasks.collection_tasks.collect_x_posts",
        "schedule": settings.x_collect_interval_min * 60,
        "options": {"queue": "finradar"},
    },
    # YouTube community posts — smart 3-tier schedule (Beat timezone is UTC).
    # KST (UTC+9) posting pattern for US-market recap creators:
    #   05–10 KST (market close window): every 30 min   →  20–01 UTC
    #   11–23 KST (daytime):              every 2 hours  →  02–14 UTC
    #   00–04 KST (overnight):            every 4 hours  →  15–19 UTC
    "youtube-posts-intensive": {
        "task": "finradar.tasks.collection_tasks.collect_youtube_posts",
        "schedule": crontab(minute="0,30", hour="20-23,0-1"),
        "options": {"queue": "finradar"},
    },
    "youtube-posts-regular": {
        "task": "finradar.tasks.collection_tasks.collect_youtube_posts",
        "schedule": crontab(minute=0, hour="2-14/2"),
        "options": {"queue": "finradar"},
    },
    "youtube-posts-light": {
        "task": "finradar.tasks.collection_tasks.collect_youtube_posts",
        "schedule": crontab(minute=0, hour="15-19/4"),
        "options": {"queue": "finradar"},
    },
    # Breaking-news alerts — fan out significant articles to Discord.
    # Per-article, per-cluster, and hourly-cap throttling all live inside
    # the dispatcher; task itself checks settings.alerts_enabled first.
    "send-breaking-alerts": {
        "task": "finradar.tasks.collection_tasks.send_breaking_alerts",
        "schedule": settings.alerts_interval_min * 60,
        "options": {"queue": "finradar"},
    },
}

logger.info(
    "Celery app configured | broker=%s | collection_interval=%d min",
    settings.redis_url,
    settings.collection_interval_minutes,
)
