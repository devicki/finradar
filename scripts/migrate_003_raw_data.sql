-- =============================================================================
-- FinRadar migration 003: raw_data JSONB column
-- -----------------------------------------------------------------------------
-- Stores per-source metadata that doesn't fit the structured columns:
--
--   * RSS   : feedparser entry dict (tags, enclosures, author, etc.)
--   * X     : tweet_id, username, breaking flag, linked_url, public_metrics
--   * API   : original response payload for debugging / traceability
--
-- The column is nullable because legacy rows predate its introduction. All
-- new ingests populate it. Use JSONB (not JSON) for fast containment ops.
--
-- Idempotent: safe to re-run.
-- =============================================================================

ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS raw_data JSONB;

-- Partial index for the BREAKING filter — Phase 3 alerting will query this
-- frequently ("show me the last hour's breaking tweets") and the index keeps
-- the scan scoped to tweets only.
CREATE INDEX IF NOT EXISTS idx_news_breaking
    ON news_items ((raw_data -> 'x' ->> 'breaking'))
    WHERE source_type = 'x_feed'
      AND raw_data -> 'x' ->> 'breaking' = 'true';
