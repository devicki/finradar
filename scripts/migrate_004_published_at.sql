-- =============================================================================
-- FinRadar migration 004: dedicated published_at column
-- -----------------------------------------------------------------------------
-- Separates "source-declared publication time" from "time we first saw the
-- article" so downstream code can stop conflating them.
--
--   Before migration
--     first_seen_at   <- article.published_at OR NOW()      (mixed semantics)
--     last_seen_at    <- article.published_at OR NOW()      (same issue)
--
--   After migration
--     published_at    <- article.published_at               (nullable)
--     first_seen_at   <- NOW() at INSERT                    (row lifecycle)
--     last_seen_at    <- NOW() at INSERT or dup hit         (hit counter)
--
-- Idempotent: every step uses IF NOT EXISTS or is safe to re-run.
-- =============================================================================

-- 1. Add the column (nullable — not every source provides a timestamp).
ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;

-- 2. Backfill: for rows written before this migration, first_seen_at was
--    effectively storing the publication time when available, so copy it.
UPDATE news_items
   SET published_at = first_seen_at
 WHERE published_at IS NULL;

-- 3. Normalise first_seen_at for existing rows to match its new semantic
--    ("DB insert time"). created_at already holds that, so align the two.
--    Most rows will see a tiny shift (milliseconds); high-hit articles may
--    shift by minutes, which is acceptable for historical analytics.
UPDATE news_items
   SET first_seen_at = created_at
 WHERE first_seen_at IS DISTINCT FROM created_at;

-- 4. Index — feed / summary queries filter and sort on published_at.
CREATE INDEX IF NOT EXISTS idx_news_published_at
    ON news_items (published_at DESC);
