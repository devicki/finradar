-- =============================================================================
-- FinRadar migration 002: news clustering columns
-- -----------------------------------------------------------------------------
-- Adds:
--   cluster_rep_id     — points to the representative NewsItem of this
--                        article's cluster. NULL for singletons (cluster of 1).
--                        When cluster_rep_id = id, this row IS the rep.
--   cluster_size       — cached count of articles in this cluster (1 for
--                        singletons). Used to render "같은 스토리 N건" badges
--                        without a subquery.
--   similarity_to_rep  — cosine similarity to the representative (0..1).
--                        1.0 for the rep itself, NULL for singletons.
--
-- Idempotent: safe to re-run.
-- =============================================================================

ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS cluster_rep_id    INT REFERENCES news_items(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS cluster_size      INT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS similarity_to_rep REAL;

-- Fast sibling lookup: "find all articles in the same cluster"
CREATE INDEX IF NOT EXISTS idx_news_cluster_rep ON news_items (cluster_rep_id);

-- For dedup queries that filter "rep-only" rows: (cluster_rep_id IS NULL OR cluster_rep_id = id)
-- The predicate is expressed with a composite so the planner can use the index.
CREATE INDEX IF NOT EXISTS idx_news_cluster_size_desc ON news_items (cluster_size DESC, last_seen_at DESC);
