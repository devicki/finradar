"""
finradar.tasks.collection_tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Celery task definitions for the FinRadar news collection and AI processing
pipeline.

Task overview
-------------
collect_all_news      — Runs every 15 min (Beat).  Fetches articles from all
                        enabled collectors (RSS, NewsAPI), deduplicates against
                        the database, inserts new rows, then queues
                        process_pending_news.

process_pending_news  — Runs every 5 min (Beat).  Scans for news_items that
                        have no sentiment score yet, runs FinBERT sentiment
                        analysis + sentence-transformer embeddings in batches,
                        and writes results back to PostgreSQL.

enrich_with_llm       — Called on-demand (or triggered by collect_all_news for
                        top-priority items).  For a single news_id: generates
                        an AI summary, translates title+summary to Korean, and
                        extracts structured tickers/sectors via the cloud LLM.

reconcile_pending_llm — Runs every 10 min (Beat).  Safety net that re-queues
                        enrich_with_llm for items whose original task was lost
                        to a restart / crash.  Guarded by an attempt counter
                        and an in-flight debounce window to avoid runaway
                        token spend on chronically-failing articles.

Sync DB layer
-------------
Celery tasks are synchronous.  We create a dedicated sync SQLAlchemy engine
(psycopg2) rather than using the main async engine (asyncpg) to avoid the
need for a running event loop inside every task.

Async collectors
----------------
The collectors (RSSCollector, NewsAPICollector) are async.  We bridge them
into the sync Celery context with a small _run_async() helper that creates a
fresh event loop per call — safe because Celery workers have no shared loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from celery import Task, group, shared_task
from sqlalchemy import create_engine, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from finradar.collectors.base import CollectedArticle
from finradar.collectors.rss_collector import RSSCollector
from finradar.config import get_settings
from finradar.models import NewsItem
from finradar.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Sync database engine (psycopg2)
# ---------------------------------------------------------------------------
# The async engine uses asyncpg; Celery workers need a sync engine.
# We swap the driver string: postgresql+asyncpg:// → postgresql://
# The rest of the DSN (host, port, credentials, dbname) stays the same.

_SYNC_DATABASE_URL: str = settings.database_url.replace("+asyncpg", "")

_sync_engine = create_engine(
    _SYNC_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,  # recycle connections every hour
    echo=settings.db_echo,
)

SyncSessionLocal: sessionmaker[Session] = sessionmaker(
    bind=_sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# Batch / processing constants
# ---------------------------------------------------------------------------

# How many unprocessed items to load per process_pending_news run.
# Keeps GPU memory usage bounded; remaining items are caught next run.
_SENTIMENT_BATCH_SIZE: int = settings.sentiment_batch_size  # default 32
_EMBEDDING_BATCH_SIZE: int = settings.embedding_batch_size  # default 64
_MAX_ITEMS_PER_PROCESSING_RUN: int = 200

# Languages eligible for automatic LLM enrichment on the collection pipeline.
# English articles receive a full EN summary + Korean translation. Korean
# articles use a KO-only prompt that skips translation (see llm_processor
# _ENRICH_KO_SYSTEM) and produces a Korean ai_summary plus tickers/sectors.
# Other languages are enriched on-demand via the API only.
_LLM_AUTO_ENRICH_LANGUAGES: frozenset[str] = frozenset({"en", "ko"})

# Minimum absolute sentiment score to qualify for LLM enrichment.
# FinBERT is trained on financial text — non-financial articles tend to
# score near 0.0 (neutral) with low confidence.  A threshold of 0.05
# filters out most irrelevant articles while keeping genuine financial news
# that happens to be neutral.  Set to 0.0 to enrich everything.
_LLM_ENRICH_MIN_SENTIMENT_SCORE: float = 0.05

# Absolute cap on LLM enrichment calls per article.  Each enrich_with_llm
# invocation increments llm_enrich_attempts; once this limit is reached
# reconcile_pending_llm stops picking the row up, preventing runaway token
# spend on articles the LLM consistently fails to process.
_LLM_ENRICH_MAX_ATTEMPTS: int = 5

# In-flight debounce window.  After queuing (or executing) enrich_with_llm for
# a row, reconcile will not re-queue it until this many minutes have passed.
# Prevents double-processing while a Celery task is still running or retrying.
_LLM_ENRICH_DEBOUNCE_MINUTES: int = 30

# Soft cap on how many items reconcile_pending_llm claims in one run.
# Keeps a single run from flooding the broker if a large backlog accumulates.
_RECONCILE_BATCH_LIMIT: int = 100

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context.

    Creates a fresh event loop per call so that:
    - No shared loop exists between tasks (avoids subtle state leaks).
    - Works on any OS / Python version supported by the project (3.11+).

    This is intentionally simple; for high-throughput scenarios consider
    running the loop in a background thread, but for the current collection
    cadence (every 15 min) this is sufficient.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _upsert_articles(session: Session, articles: list[CollectedArticle]) -> tuple[int, int]:
    """Insert new articles; increment hit_count for duplicates.

    Uses PostgreSQL's ON CONFLICT DO UPDATE so the entire batch can be
    handled in a single round-trip.

    The unique constraint is on (topic_id, url).  Since we don't assign
    topic_id during collection (topic classification is a future step),
    topic_id is NULL for all freshly collected items.  PostgreSQL treats
    NULL as distinct from any value for uniqueness, so the constraint
    ``uq_topic_url`` won't fire for NULL topic_ids — we handle dedup on
    ``url`` alone by checking existence beforehand.

    Returns:
        (inserted_count, updated_count)
    """
    if not articles:
        return 0, 0

    inserted = 0
    updated = 0
    now = _now_utc()

    # Bulk-fetch all URLs that already exist in the DB so we can split
    # the batch into inserts vs. updates without per-row round-trips.
    incoming_urls = [a.url for a in articles]
    existing_rows = session.execute(
        select(NewsItem.url, NewsItem.id).where(NewsItem.url.in_(incoming_urls))
    ).all()
    existing_url_to_id: dict[str, int] = {row.url: row.id for row in existing_rows}

    new_articles: list[CollectedArticle] = []
    duplicate_ids: list[int] = []

    for article in articles:
        if article.url in existing_url_to_id:
            duplicate_ids.append(existing_url_to_id[article.url])
        else:
            new_articles.append(article)

    # Bulk-update hit_count and last_seen_at for duplicates
    if duplicate_ids:
        session.execute(
            update(NewsItem)
            .where(NewsItem.id.in_(duplicate_ids))
            .values(
                hit_count=NewsItem.hit_count + 1,
                last_seen_at=now,
                updated_at=now,
            )
        )
        updated = len(duplicate_ids)
        logger.debug("Updated hit_count for %d duplicate articles", updated)

    # Bulk-insert new articles
    if new_articles:
        news_item_dicts = [
            {
                "topic_id": None,
                "title": article.title,
                "summary": article.summary,
                "url": article.url,
                "source_url": article.source_url,
                "source_type": article.source_type,
                "language": article.language,
                "first_seen_at": article.published_at or now,
                "last_seen_at": article.published_at or now,
                "hit_count": 1,
                "tickers": article.tickers if article.tickers else None,
                "sectors": article.sectors if article.sectors else None,
                # AI fields left NULL — filled by process_pending_news
                "sentiment": None,
                "sentiment_label": None,
                "translated_title": None,
                "translated_summary": None,
                "ai_summary": None,
                "embedding": None,
                "search_vector": None,
                "created_at": now,
                "updated_at": now,
            }
            for article in new_articles
        ]

        # Use PostgreSQL INSERT ... ON CONFLICT DO NOTHING as a final safety
        # net in case two workers race to insert the same URL simultaneously.
        stmt = pg_insert(NewsItem).values(news_item_dicts)
        stmt = stmt.on_conflict_do_nothing()
        session.execute(stmt)
        inserted = len(new_articles)
        logger.debug("Inserted %d new articles", inserted)

    session.commit()
    return inserted, updated


# ---------------------------------------------------------------------------
# Task: collect_all_news
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="finradar.tasks.collection_tasks.collect_all_news",
    max_retries=3,
    default_retry_delay=60,  # 1 minute between retries
    queue="finradar",
)
def collect_all_news(self: Task) -> dict[str, Any]:
    """Collect news from all configured sources and persist to PostgreSQL.

    Steps
    -----
    1. Run RSSCollector (and NewsAPICollector when the key is configured).
    2. Merge all collected articles into one deduplicated list.
    3. Upsert into news_items: new rows inserted, duplicate URLs get an
       incremented hit_count and refreshed last_seen_at.
    4. Trigger process_pending_news asynchronously so newly inserted items
       get their AI features without waiting for the Beat schedule.

    Returns a summary dict suitable for Celery result inspection.
    """
    logger.info("collect_all_news: starting collection run")
    self.update_state(state="STARTED", meta={"stage": "collecting"})

    all_articles: list[CollectedArticle] = []
    collector_results: dict[str, int] = {}

    # --- RSS ---
    try:
        rss_articles: list[CollectedArticle] = _run_async(_collect_rss())
        collector_results["rss"] = len(rss_articles)
        all_articles.extend(rss_articles)
        logger.info("RSS: collected %d articles", len(rss_articles))
    except Exception as exc:
        logger.error("RSS collection failed: %s", exc, exc_info=True)
        collector_results["rss"] = 0

    # --- NewsAPI (only when enabled AND a key is configured) ---
    # NewsAPI free plan is dev-only (not for commercial use).
    # Set NEWSAPI_ENABLED=false in production.
    if settings.newsapi_enabled and settings.newsapi_key:
        try:
            newsapi_articles: list[CollectedArticle] = _run_async(_collect_newsapi())
            collector_results["newsapi"] = len(newsapi_articles)
            all_articles.extend(newsapi_articles)
            logger.info("NewsAPI: collected %d articles", len(newsapi_articles))
        except Exception as exc:
            logger.error("NewsAPI collection failed: %s", exc, exc_info=True)
            collector_results["newsapi"] = 0
    else:
        logger.debug("NewsAPI key not configured — skipping")
        collector_results["newsapi"] = 0

    if not all_articles:
        logger.warning("collect_all_news: no articles collected from any source")
        return {
            "status": "ok",
            "total_collected": 0,
            "inserted": 0,
            "updated": 0,
            "collectors": collector_results,
        }

    # --- Deduplication at Python level (same URL from multiple feeds) ---
    seen_urls: set[str] = set()
    unique_articles: list[CollectedArticle] = []
    for article in all_articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique_articles.append(article)

    logger.info(
        "collect_all_news: %d total → %d unique articles after in-memory dedup",
        len(all_articles),
        len(unique_articles),
    )

    # --- Persist ---
    self.update_state(state="STARTED", meta={"stage": "persisting"})
    try:
        with SyncSessionLocal() as session:
            inserted, updated = _upsert_articles(session, unique_articles)
    except Exception as exc:
        logger.error("Database upsert failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)

    logger.info(
        "collect_all_news: inserted=%d updated=%d",
        inserted,
        updated,
    )

    # --- Kick off AI processing immediately if we have new items ---
    if inserted > 0:
        process_pending_news.apply_async(queue="finradar")
        logger.info("collect_all_news: triggered process_pending_news for %d new items", inserted)

    return {
        "status": "ok",
        "total_collected": len(all_articles),
        "unique": len(unique_articles),
        "inserted": inserted,
        "updated": updated,
        "collectors": collector_results,
    }


async def _collect_rss() -> list[CollectedArticle]:
    """Coroutine wrapper: run the RSS collector and close the HTTP client."""
    async with RSSCollector(max_concurrent=settings.collector_concurrency) as collector:
        return await collector.safe_collect()


async def _collect_newsapi() -> list[CollectedArticle]:
    """Coroutine wrapper: run the NewsAPI collector.

    The NewsAPICollector is imported lazily here so that workers without a
    NewsAPI key (or without the module implemented yet) don't fail at import
    time.  If the class doesn't exist yet this will raise ImportError, which
    is caught by the caller.
    """
    from finradar.collectors.newsapi_collector import NewsAPICollector  # noqa: PLC0415

    async with NewsAPICollector(api_key=settings.newsapi_key, max_concurrent=settings.collector_concurrency) as collector:
        return await collector.safe_collect()


# ---------------------------------------------------------------------------
# Task: process_pending_news
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="finradar.tasks.collection_tasks.process_pending_news",
    max_retries=3,
    default_retry_delay=120,
    queue="finradar",
)
def process_pending_news(self: Task) -> dict[str, Any]:
    """Run sentiment analysis and embedding generation on unprocessed items.

    "Unprocessed" is defined as news_items where sentiment IS NULL — this
    means the row was inserted by collect_all_news but the local GPU models
    haven't been applied yet.

    Processing is done in two passes:
    1. Sentiment analysis (FinBERT) — batched per settings.sentiment_batch_size
    2. Embedding generation — batched per settings.embedding_batch_size

    Both passes update the same rows; a single DB flush happens at the end of
    each batch to minimise round-trips.

    After processing, each item is eligible for LLM enrichment.  To avoid
    spamming the cloud API, we only auto-enqueue LLM tasks for items whose
    language is in _LLM_AUTO_ENRICH_LANGUAGES (English by default).
    """
    logger.info("process_pending_news: starting AI processing run")
    self.update_state(state="STARTED", meta={"stage": "loading_models"})

    # --- Lazy-load local AI models ---
    # Models are loaded once per worker process (not per task call) because
    # they are heavy (FinBERT ~440MB, all-MiniLM ~90MB).  The module-level
    # singletons in the processor modules handle this automatically.
    try:
        from finradar.processors.embeddings import EmbeddingGenerator  # noqa: PLC0415
        from finradar.processors.sentiment import get_sentiment_analyzer  # noqa: PLC0415

        en_analyzer = get_sentiment_analyzer("en")
        ko_analyzer = get_sentiment_analyzer("ko")
        embedder = EmbeddingGenerator()
    except Exception as exc:
        logger.error("Failed to load AI models: %s", exc, exc_info=True)
        raise self.retry(exc=exc)

    # --- Fetch unprocessed batch from DB ---
    self.update_state(state="STARTED", meta={"stage": "fetching_unprocessed"})

    with SyncSessionLocal() as session:
        unprocessed: list[NewsItem] = session.execute(
            select(NewsItem)
            .where(NewsItem.sentiment.is_(None))
            .order_by(NewsItem.last_seen_at.desc())
            .limit(_MAX_ITEMS_PER_PROCESSING_RUN)
        ).scalars().all()

    if not unprocessed:
        logger.info("process_pending_news: no unprocessed items found")
        return {"status": "ok", "processed": 0}

    logger.info("process_pending_news: found %d items to process", len(unprocessed))

    # --- Sentiment analysis ---
    self.update_state(state="STARTED", meta={"stage": "sentiment", "total": len(unprocessed)})

    sentiment_results: dict[int, dict[str, Any]] = {}  # news_id → {score, label}

    # Group items by language so we use the right model
    en_items = [item for item in unprocessed if (item.language or "en") == "en"]
    ko_items = [item for item in unprocessed if (item.language or "en") == "ko"]
    other_items = [item for item in unprocessed if item not in en_items and item not in ko_items]

    for batch_items, analyzer in [
        (en_items, en_analyzer),
        (ko_items, ko_analyzer),
        (other_items, en_analyzer),  # fall back to EN model for other languages
    ]:
        if not batch_items:
            continue
        for i in range(0, len(batch_items), _SENTIMENT_BATCH_SIZE):
            batch = batch_items[i : i + _SENTIMENT_BATCH_SIZE]
            for item in batch:
                text = _build_sentiment_text(item)
                try:
                    result = analyzer.analyze(text)
                    sentiment_results[item.id] = {
                        "score": result["score"],
                        "label": result["label"],
                    }
                except Exception as exc:
                    logger.warning(
                        "Sentiment analysis failed for item %d: %s", item.id, exc
                    )
                    # Use neutral as safe default so the item isn't retried
                    # indefinitely — a score of 0.0 means "processed, neutral".
                    sentiment_results[item.id] = {"score": 0.0, "label": "neutral"}

            logger.debug(
                "Sentiment batch %d-%d done (%d items)",
                i,
                i + len(batch),
                len(batch),
            )

    # --- Embedding generation ---
    self.update_state(state="STARTED", meta={"stage": "embeddings", "total": len(unprocessed)})

    embedding_results: dict[int, list[float]] = {}  # news_id → embedding vector

    for i in range(0, len(unprocessed), _EMBEDDING_BATCH_SIZE):
        batch = unprocessed[i : i + _EMBEDDING_BATCH_SIZE]
        for item in batch:
            text = _build_embedding_text(item)
            try:
                embedding = embedder.generate(text)
                embedding_results[item.id] = embedding
            except Exception as exc:
                logger.warning(
                    "Embedding generation failed for item %d: %s", item.id, exc
                )
                # Leave embedding as None — item will remain eligible for
                # reprocessing next cycle, but sentiment will be set so it
                # won't re-run the (cheaper) sentiment pass unnecessarily.

        logger.debug(
            "Embedding batch %d-%d done (%d items)",
            i,
            i + len(batch),
            len(batch),
        )

    # --- Write results to DB ---
    self.update_state(state="STARTED", meta={"stage": "saving"})

    now = _now_utc()
    llm_enrich_ids: list[int] = []

    with SyncSessionLocal() as session:
        for item in unprocessed:
            item_id = item.id
            update_kwargs: dict[str, Any] = {"updated_at": now}

            if item_id in sentiment_results:
                sr = sentiment_results[item_id]
                update_kwargs["sentiment"] = sr["score"]
                update_kwargs["sentiment_label"] = sr["label"]

            if item_id in embedding_results:
                update_kwargs["embedding"] = embedding_results[item_id]

            if update_kwargs:
                session.execute(
                    update(NewsItem)
                    .where(NewsItem.id == item_id)
                    .values(**update_kwargs)
                )

            # Collect IDs eligible for LLM enrichment:
            # - Must be in a supported language (English by default)
            # - Must have a sentiment score above threshold (filters out
            #   non-financial articles that FinBERT scores near zero)
            if (item.language or "en") in _LLM_AUTO_ENRICH_LANGUAGES:
                sentiment_score = sentiment_results.get(item_id, {}).get("score", 0.0)
                if abs(sentiment_score) >= _LLM_ENRICH_MIN_SENTIMENT_SCORE:
                    llm_enrich_ids.append(item_id)

        # Claim eligible rows before enqueuing so that reconcile_pending_llm
        # won't pick them up again while these tasks are still in flight.
        # The enrich_with_llm task will refresh this timestamp on entry.
        if llm_enrich_ids:
            session.execute(
                update(NewsItem)
                .where(NewsItem.id.in_(llm_enrich_ids))
                .values(llm_last_attempt_at=now)
            )

        session.commit()

    logger.info(
        "process_pending_news: processed %d items "
        "(sentiment=%d, embeddings=%d)",
        len(unprocessed),
        len(sentiment_results),
        len(embedding_results),
    )

    # --- Queue LLM enrichment for eligible items ---
    if llm_enrich_ids:
        # Use a Celery group so all enrichment tasks are submitted in one
        # broker round-trip, but each runs independently.
        enrich_group = group(
            enrich_with_llm.s(news_id).set(queue="finradar.llm")
            for news_id in llm_enrich_ids
        )
        enrich_group.apply_async()
        logger.info(
            "process_pending_news: queued LLM enrichment for %d items",
            len(llm_enrich_ids),
        )

    return {
        "status": "ok",
        "processed": len(unprocessed),
        "sentiment_ok": len(sentiment_results),
        "embeddings_ok": len(embedding_results),
        "llm_queued": len(llm_enrich_ids),
    }


def _build_sentiment_text(item: NewsItem) -> str:
    """Build the text string passed to FinBERT for a NewsItem.

    FinBERT performs best on short texts (≤512 tokens).  We use the title
    only; the summary (if any) can push the input over the limit and actually
    *hurt* performance for financial sentiment classification.
    """
    return item.title


def _build_embedding_text(item: NewsItem) -> str:
    """Build the text string passed to the sentence-transformer for a NewsItem.

    Sentence-transformers benefit from richer context, so we concatenate
    title and summary (if available) separated by ". ".
    """
    parts = [item.title]
    if item.summary:
        parts.append(item.summary)
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Task: enrich_with_llm
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="finradar.tasks.collection_tasks.enrich_with_llm",
    max_retries=2,
    default_retry_delay=300,  # 5 minutes — back off from cloud API rate limits
    queue="finradar.llm",
)
def enrich_with_llm(self: Task, news_id: int) -> dict[str, Any]:
    """Enrich a single news item with a single LLM call.

    Performs summary, translation (en→ko), and metadata extraction
    (tickers/sectors) in one combined prompt to minimize API costs.

    Parameters
    ----------
    news_id:
        Primary key of the news_items row to enrich.

    Notes
    -----
    - Uses ``LLMProcessor.enrich_article()`` — one LLM call instead of 3-4.
    - If the item already has an ai_summary the task exits immediately (idempotent).
    """
    logger.info("enrich_with_llm: enriching news_id=%d", news_id)
    self.update_state(state="STARTED", meta={"news_id": news_id, "stage": "loading"})

    # --- Load the item ---
    with SyncSessionLocal() as session:
        item: NewsItem | None = session.get(NewsItem, news_id)
        if item is None:
            logger.warning("enrich_with_llm: news_id=%d not found — skipping", news_id)
            return {"status": "skipped", "reason": "not_found", "news_id": news_id}

        # Idempotency check: don't re-process already-enriched items.
        if item.ai_summary is not None:
            logger.debug(
                "enrich_with_llm: news_id=%d already enriched — skipping", news_id
            )
            return {"status": "skipped", "reason": "already_enriched", "news_id": news_id}

        # Attempt-limit check: if we've already burned the retry budget, abort
        # without calling the LLM.  reconcile_pending_llm filters by the same
        # limit, but an in-flight task scheduled before the limit was hit could
        # still arrive here — so we double-check.
        if (item.llm_enrich_attempts or 0) >= _LLM_ENRICH_MAX_ATTEMPTS:
            logger.warning(
                "enrich_with_llm: news_id=%d has %d attempts (>= cap %d) — aborting",
                news_id, item.llm_enrich_attempts, _LLM_ENRICH_MAX_ATTEMPTS,
            )
            return {
                "status": "skipped",
                "reason": "attempts_exhausted",
                "news_id": news_id,
            }

        # Claim this invocation: bump attempts + refresh timestamp atomically.
        # Committing before the LLM call ensures the counter rises even if the
        # worker is killed mid-request (protects against runaway retries).
        session.execute(
            update(NewsItem)
            .where(NewsItem.id == news_id)
            .values(
                llm_enrich_attempts=NewsItem.llm_enrich_attempts + 1,
                llm_last_attempt_at=_now_utc(),
            )
        )
        session.commit()

        # Snapshot mutable fields before the session closes
        item_id = item.id
        item_title = item.title
        item_summary = item.summary
        item_language = item.language or "en"
        item_tickers = list(item.tickers or [])
        item_sectors = list(item.sectors or [])

    # --- Load LLM processor ---
    try:
        from finradar.processors.llm_processor import LLMProcessor  # noqa: PLC0415

        llm = LLMProcessor(provider=settings.llm_provider)
    except Exception as exc:
        logger.error("Failed to initialise LLMProcessor: %s", exc, exc_info=True)
        raise self.retry(exc=exc)

    # --- Single LLM call: summary + translation + metadata ---
    self.update_state(state="STARTED", meta={"news_id": news_id, "stage": "enriching"})
    try:
        enrichment: dict[str, Any] = _run_async(
            llm.enrich_article(item_title, item_summary or "", item_language)
        )
    except Exception as exc:
        logger.error(
            "enrich_with_llm: LLM call failed for news_id=%d: %s",
            news_id, exc, exc_info=True,
        )
        raise self.retry(exc=exc, countdown=300)

    # --- Build update dict from enrichment result ---
    update_kwargs: dict[str, Any] = {"updated_at": _now_utc()}

    if enrichment.get("ai_summary"):
        update_kwargs["ai_summary"] = enrichment["ai_summary"]
    if enrichment.get("translated_title"):
        update_kwargs["translated_title"] = enrichment["translated_title"]
    if enrichment.get("translated_summary"):
        update_kwargs["translated_summary"] = enrichment["translated_summary"]
    if not item_tickers and enrichment.get("tickers"):
        update_kwargs["tickers"] = enrichment["tickers"]
    if not item_sectors and enrichment.get("sectors"):
        update_kwargs["sectors"] = enrichment["sectors"]

    # --- Persist ---
    if len(update_kwargs) <= 1:
        logger.error(
            "enrich_with_llm: LLM returned empty results for news_id=%d — will retry",
            news_id,
        )
        raise self.retry(
            exc=RuntimeError(f"LLM returned empty results for news_id={news_id}"),
            countdown=300,
        )

    try:
        with SyncSessionLocal() as session:
            session.execute(
                update(NewsItem)
                .where(NewsItem.id == item_id)
                .values(**update_kwargs)
            )
            session.commit()
    except Exception as exc:
        logger.error(
            "enrich_with_llm: DB update failed for news_id=%d: %s", news_id, exc, exc_info=True
        )
        raise self.retry(exc=exc)

    logger.info(
        "enrich_with_llm: completed for news_id=%d "
        "(summary=%s, translated_title=%s, tickers=%s, sectors=%s)",
        news_id,
        "yes" if "ai_summary" in update_kwargs else "no",
        "yes" if "translated_title" in update_kwargs else "no",
        update_kwargs.get("tickers"),
        update_kwargs.get("sectors"),
    )

    return {
        "status": "ok",
        "news_id": news_id,
        "summary": "ai_summary" in update_kwargs,
        "translated_title": "translated_title" in update_kwargs,
        "translated_summary": "translated_summary" in update_kwargs,
        "tickers": update_kwargs.get("tickers"),
        "sectors": update_kwargs.get("sectors"),
    }


# ---------------------------------------------------------------------------
# Task: reconcile_pending_llm
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="finradar.tasks.collection_tasks.reconcile_pending_llm",
    max_retries=2,
    default_retry_delay=120,
    queue="finradar",
)
def reconcile_pending_llm(self: Task) -> dict[str, Any]:
    """Re-queue LLM enrichment for items orphaned by restarts or task loss.

    Why this exists
    ---------------
    process_pending_news only queues enrich_with_llm for items it *just*
    computed sentiment for.  If the worker dies (or Redis is restarted)
    before the enrichment task runs, the row stays in a "sentiment set,
    ai_summary NULL" state and is never retried.

    This task scans for such orphans every few minutes and re-queues them,
    guarded by an attempt counter and an in-flight debounce window so that
    no article can burn tokens indefinitely.

    Guardrails
    ----------
    - Per-row cap: ``llm_enrich_attempts < _LLM_ENRICH_MAX_ATTEMPTS`` (default 5).
      Each enrich_with_llm invocation bumps the counter; once the cap is hit
      the row is left alone (an operator must reset it manually to retry).
    - Debounce: rows touched within ``_LLM_ENRICH_DEBOUNCE_MINUTES`` are
      skipped — prevents double-queueing while a task is still running.
    - Per-run cap: ``_RECONCILE_BATCH_LIMIT`` items per tick keeps the broker
      from being flooded when a large backlog accumulates.
    - ``SELECT ... FOR UPDATE SKIP LOCKED`` makes the claim race-free when
      multiple Beat / reconcile runs overlap.
    """
    logger.info("reconcile_pending_llm: scanning for orphaned LLM items")
    self.update_state(state="STARTED", meta={"stage": "scanning"})

    now = _now_utc()
    debounce_cutoff = now - timedelta(minutes=_LLM_ENRICH_DEBOUNCE_MINUTES)

    with SyncSessionLocal() as session:
        # Find candidate IDs with row-level locks so concurrent reconciles
        # don't double-claim.  The query mirrors the eligibility rules used
        # by process_pending_news, with attempt + debounce guards added.
        candidate_ids: list[int] = session.execute(
            select(NewsItem.id)
            .where(
                NewsItem.sentiment.is_not(None),
                func.abs(NewsItem.sentiment) >= _LLM_ENRICH_MIN_SENTIMENT_SCORE,
                NewsItem.language.in_(tuple(_LLM_AUTO_ENRICH_LANGUAGES)),
                NewsItem.ai_summary.is_(None),
                NewsItem.llm_enrich_attempts < _LLM_ENRICH_MAX_ATTEMPTS,
                or_(
                    NewsItem.llm_last_attempt_at.is_(None),
                    NewsItem.llm_last_attempt_at < debounce_cutoff,
                ),
            )
            .order_by(NewsItem.last_seen_at.desc())
            .limit(_RECONCILE_BATCH_LIMIT)
            .with_for_update(skip_locked=True)
        ).scalars().all()

        if not candidate_ids:
            logger.info("reconcile_pending_llm: no orphaned items found")
            return {"status": "ok", "requeued": 0}

        # Claim the rows by stamping llm_last_attempt_at.  enrich_with_llm
        # will bump llm_enrich_attempts when it actually runs — we leave the
        # counter alone here so this scan doesn't burn the budget on its own.
        session.execute(
            update(NewsItem)
            .where(NewsItem.id.in_(candidate_ids))
            .values(llm_last_attempt_at=now)
        )
        session.commit()

    logger.info(
        "reconcile_pending_llm: claimed %d items for re-enrichment",
        len(candidate_ids),
    )

    enrich_group = group(
        enrich_with_llm.s(news_id).set(queue="finradar.llm")
        for news_id in candidate_ids
    )
    enrich_group.apply_async()

    return {"status": "ok", "requeued": len(candidate_ids)}
