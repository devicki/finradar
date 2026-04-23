-- =============================================================================
-- Migration 006: LLM sentiment signal (Phase 3 alert quality gate)
--
-- Adds a second sentiment signal emitted by the cloud LLM during enrichment so
-- the Discord alert dispatcher can require local-sentiment + LLM agreement
-- before firing the strong_sentiment trigger.  The existing ``sentiment`` /
-- ``sentiment_label`` columns are owned by the language-branching local model
-- (ProsusAI/finbert for EN, snunlp/KR-FinBert-SC for KO) — see
-- ``finradar.processors.sentiment.get_sentiment_analyzer``.  Kept separate
-- from those columns because:
--   - coverage differs: the local model runs on every article, LLM runs only
--     on articles that passed the enrich filter (|local sentiment| >= 0.05)
--   - the gap between the two signals is a useful diagnostic when tuning
--     thresholds or reviewing misfires
--   - search / feed / dashboard ranking continues to use the existing
--     ``sentiment`` column unchanged
--
-- Idempotent; safe to run on any environment.
-- Run with:
--   docker compose exec db psql -U finradar -d finradar \
--     -f /docker-entrypoint-initdb.d/migrate_006_llm_sentiment.sql
-- =============================================================================

ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS llm_sentiment FLOAT;

ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS llm_sentiment_label VARCHAR(10);
