-- =============================================================================
-- Migration 001: LLM enrichment tracking columns
--
-- Adds two columns to news_items to prevent infinite LLM retry loops and
-- duplicate in-flight enqueueing:
--   - llm_enrich_attempts : counter incremented on each enrich_with_llm call
--   - llm_last_attempt_at : timestamp set when queued/executed (in-flight lock)
--
-- Idempotent; safe to run on any environment.
-- Run with:
--   docker compose exec db psql -U finradar -d finradar \
--     -f /docker-entrypoint-initdb.d/migrate_001_llm_tracking.sql
-- =============================================================================

ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS llm_enrich_attempts INTEGER NOT NULL DEFAULT 0;

ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS llm_last_attempt_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_news_llm_pending
    ON news_items (llm_last_attempt_at NULLS FIRST)
    WHERE ai_summary IS NULL;
