-- =============================================================================
-- FinRadar — PostgreSQL initialisation script
-- Run automatically by the pgvector/pgvector:pg16 Docker image on first boot
-- (files placed in /docker-entrypoint-initdb.d/ are executed in alphabetical
--  order for the target database only).
--
-- Idempotent: all statements use IF NOT EXISTS / OR REPLACE so re-running is
-- safe during development resets.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: dense-vector similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram similarity for fuzzy FTS

-- ---------------------------------------------------------------------------
-- topics
-- Organises news into broad thematic buckets (Macro, Semiconductors, …).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topics (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(128) NOT NULL,
    slug        VARCHAR(128) NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_topics_name UNIQUE (name),
    CONSTRAINT uq_topics_slug UNIQUE (slug)
);

-- ---------------------------------------------------------------------------
-- news_items
-- Central fact table.  One row per unique (topic, url) pair.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_items (
    id              SERIAL PRIMARY KEY,
    topic_id        INTEGER REFERENCES topics(id) ON DELETE SET NULL,

    -- Core content
    title           TEXT NOT NULL,
    summary         TEXT,
    url             VARCHAR(2048) NOT NULL,
    source_url      VARCHAR(2048) NOT NULL,

    -- Deduplication / freshness tracking (embird pattern)
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL,
    hit_count       INTEGER NOT NULL DEFAULT 1,

    -- AI analysis — sentiment
    sentiment       FLOAT,                  -- continuous score  -1.0 … +1.0
    sentiment_label VARCHAR(10),            -- 'positive' | 'negative' | 'neutral'

    -- Provenance
    source_type     VARCHAR(20),            -- 'rss' | 'api' | 'x_feed' | 'url_report'
    language        VARCHAR(5),             -- 'en' | 'ko' | 'ja' | …

    -- AI-generated translations & summaries
    translated_title    TEXT,
    translated_summary  TEXT,
    ai_summary          TEXT,

    -- LLM enrichment bookkeeping
    llm_enrich_attempts  INTEGER NOT NULL DEFAULT 0,
    llm_last_attempt_at  TIMESTAMPTZ,

    -- Structured tags (array columns, indexed with GIN)
    tickers         VARCHAR(16)[],          -- e.g. '{AAPL,TSLA}'
    sectors         VARCHAR(64)[],          -- e.g. '{반도체,AI}'

    -- Vector search (384-dim all-MiniLM-L6-v2)
    embedding       vector(384),

    -- Full-text search (maintained by trigger below)
    search_vector   TSVECTOR,

    -- Row timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_news_topic_url UNIQUE (topic_id, url)
);

-- ---------------------------------------------------------------------------
-- Indexes on news_items
-- ---------------------------------------------------------------------------

-- ANN vector search — IVFFlat with cosine distance.
-- lists=100 is a sensible default; re-tune when row count grows past ~1 M.
CREATE INDEX IF NOT EXISTS idx_news_embedding
    ON news_items
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_news_search
    ON news_items
    USING GIN (search_vector);

-- Ticker lookup / filtering
CREATE INDEX IF NOT EXISTS idx_news_tickers
    ON news_items
    USING GIN (tickers);

-- Recency-based feed queries (most recent first)
CREATE INDEX IF NOT EXISTS idx_news_last_seen
    ON news_items (last_seen_at DESC);

-- Sentiment filtering
CREATE INDEX IF NOT EXISTS idx_news_sentiment_label
    ON news_items (sentiment_label);

-- Source-type filtering
CREATE INDEX IF NOT EXISTS idx_news_source_type
    ON news_items (source_type);

-- Topic foreign-key lookups
CREATE INDEX IF NOT EXISTS idx_news_topic_id
    ON news_items (topic_id);

-- ---------------------------------------------------------------------------
-- Migration: LLM enrichment bookkeeping (for databases created before these
-- columns were part of the initial schema).  Safe to run repeatedly.
-- ---------------------------------------------------------------------------
ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS llm_enrich_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS llm_last_attempt_at TIMESTAMPTZ;

-- Partial index to make the reconcile query fast: only indexes rows that are
-- still candidates for LLM enrichment (the vast majority eventually leave
-- this set when ai_summary is populated).
CREATE INDEX IF NOT EXISTS idx_news_llm_pending
    ON news_items (llm_last_attempt_at NULLS FIRST)
    WHERE ai_summary IS NULL;

-- ---------------------------------------------------------------------------
-- Trigger: auto-update search_vector on INSERT / UPDATE
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION news_items_search_vector_update()
RETURNS TRIGGER
LANGUAGE plpgsql AS
$$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.ai_summary, '')), 'C');
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trig_news_items_search_vector ON news_items;
CREATE TRIGGER trig_news_items_search_vector
    BEFORE INSERT OR UPDATE OF title, summary, ai_summary
    ON news_items
    FOR EACH ROW
    EXECUTE FUNCTION news_items_search_vector_update();

-- ---------------------------------------------------------------------------
-- Trigger: auto-update updated_at on row change
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql AS
$$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trig_news_items_updated_at ON news_items;
CREATE TRIGGER trig_news_items_updated_at
    BEFORE UPDATE ON news_items
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- user_feedback
-- Stores explicit user signals (bookmark, like, dislike, dismiss).
-- Phase 3 will extend this for implicit signals (click-through, dwell time).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_feedback (
    id          SERIAL PRIMARY KEY,
    news_id     INTEGER NOT NULL REFERENCES news_items(id) ON DELETE CASCADE,
    action      VARCHAR(20) NOT NULL,       -- 'bookmark' | 'like' | 'dislike' | 'dismiss'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_feedback_action CHECK (
        action IN ('bookmark', 'like', 'dislike', 'dismiss')
    )
);

CREATE INDEX IF NOT EXISTS idx_feedback_news_id
    ON user_feedback (news_id);

CREATE INDEX IF NOT EXISTS idx_feedback_created_at
    ON user_feedback (created_at DESC);

-- ---------------------------------------------------------------------------
-- Default topics
-- ---------------------------------------------------------------------------
INSERT INTO topics (name, slug, description) VALUES
    ('Macro Economy',    'macro-economy',  '거시경제 — 금리, 인플레이션, GDP, 중앙은행 정책'),
    ('Semiconductors',   'semiconductors', '반도체 산업 — TSMC, 삼성, NVIDIA, 공급망'),
    ('AI Technology',    'ai-technology',  'AI/ML 기술 — LLM, 칩, 빅테크, 스타트업'),
    ('Energy',           'energy',         '에너지 — 원유, 천연가스, 신재생, OPEC'),
    ('Crypto',           'crypto',         '암호화폐 — BTC, ETH, DeFi, 규제'),
    ('Forex',            'forex',          '외환시장 — USD, KRW, JPY, EUR, 달러인덱스')
ON CONFLICT (slug) DO NOTHING;
