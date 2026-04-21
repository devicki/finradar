-- =============================================================================
-- FinRadar migration 005: multi-user-ready feedback schema
-- -----------------------------------------------------------------------------
-- Adds user_id to user_feedback so every row identifies who gave the signal.
-- Phase 3 hardcodes a single "owner" user; Phase 4 (public service + auth)
-- wires real authenticated user IDs in without schema churn.
--
-- Also enforces (user_id, news_id, action) uniqueness so repeated clicks on
-- the same button turn into idempotent no-ops instead of stacking duplicate
-- rows.
--
-- Idempotent: every step uses IF NOT EXISTS or guards against re-running.
-- =============================================================================

ALTER TABLE user_feedback
    ADD COLUMN IF NOT EXISTS user_id VARCHAR(64) NOT NULL DEFAULT 'owner';

-- Enforce NOT NULL action + news_id while we're at it (the original schema
-- marked them nullable but the app has always required them).
ALTER TABLE user_feedback
    ALTER COLUMN news_id SET NOT NULL,
    ALTER COLUMN action  SET NOT NULL;

-- De-duplicate any pre-existing identical (user_id, news_id, action) rows so
-- the unique index below can be created. Keeps the row with the lowest id.
DELETE FROM user_feedback uf
 WHERE EXISTS (
    SELECT 1 FROM user_feedback uf2
     WHERE uf2.user_id = uf.user_id
       AND uf2.news_id = uf.news_id
       AND uf2.action  = uf.action
       AND uf2.id      < uf.id
 );

CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_user_news_action
    ON user_feedback (user_id, news_id, action);

-- Fast "my bookmarks" / "my dismissed" lookups.
CREATE INDEX IF NOT EXISTS idx_feedback_user_action_created
    ON user_feedback (user_id, action, created_at DESC);
