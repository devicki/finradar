"""
finradar.config
~~~~~~~~~~~~~~~

Application settings loaded from environment variables / .env file.
All settings are validated by Pydantic at startup time.

Usage
-----
    from finradar.config import get_settings

    settings = get_settings()
    print(settings.database_url)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration object for FinRadar.

    Values are read (in priority order) from:
      1. Actual environment variables
      2. `.env` file in the project root
      3. The defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = Field(default="FinRadar", description="Human-readable application name")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False, description="Enable debug mode (verbose logging, reload)")

    # -------------------------------------------------------------------------
    # Database (PostgreSQL + pgvector)
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://finradar:finradar_dev@db:5432/finradar",
        description="Async SQLAlchemy DSN for PostgreSQL",
    )
    db_password: str = Field(
        default="finradar_dev",
        description="PostgreSQL password (also used in docker-compose env substitution)",
    )
    db_pool_size: int = Field(default=10, ge=1, le=50)
    db_max_overflow: int = Field(default=20, ge=0, le=100)
    db_pool_timeout: int = Field(default=30, ge=1)
    db_echo: bool = Field(default=False, description="Log all SQL statements")

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL (used for caching and Celery broker/backend)",
    )

    # -------------------------------------------------------------------------
    # News APIs
    # -------------------------------------------------------------------------
    newsapi_key: str = Field(default="", description="NewsAPI.org API key")
    newsapi_enabled: bool = Field(
        default=False,
        description="Enable NewsAPI.org collector. Set to false in production (free plan is dev-only)",
    )
    polygon_api_key: str = Field(default="", description="Polygon.io API key")

    # -------------------------------------------------------------------------
    # X (Twitter) API — pay-as-you-go at $0.005 per read
    # -------------------------------------------------------------------------
    x_enabled: bool = Field(
        default=False,
        description=(
            "Enable X (Twitter) collector. When false OR x_bearer_token is empty, "
            "the X Celery task skips itself gracefully — no API calls, no cost."
        ),
    )
    x_bearer_token: str = Field(
        default="",
        description="X API v2 bearer token (App-only context is sufficient for timeline reads)",
    )
    x_tracked_accounts: str = Field(
        default="markets,business,BloombergTV",
        description=(
            "Comma-separated X usernames to follow (without @). Each account is polled "
            "on every collect_x_posts run; new tweets since last_seen_id are ingested."
        ),
    )
    x_collect_interval_min: int = Field(
        default=10,
        ge=1,
        description="Minutes between collect_x_posts runs (lower = lower latency, higher cost)",
    )
    x_max_tweets_per_account: int = Field(
        default=50,
        ge=5,
        le=100,
        description="Max tweets to fetch per account per run (X API limits this to 100)",
    )
    x_monthly_budget_usd: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "Hard monthly spend cap in USD. When the Redis-tracked running total "
            "reaches this threshold, collect_x_posts stops making API calls until "
            "the next calendar month."
        ),
    )
    x_cost_per_read_usd: float = Field(
        default=0.0053,
        ge=0.0,
        description=(
            "Current X API billed price per tweet resource. The posted X pricing is "
            "$0.005/resource but observed billing on small test runs is ~$0.0053 because "
            "requested tweet fields (entities, referenced_tweets, ...) count as additional "
            "resources. Update empirically when X changes pricing."
        ),
    )

    # -------------------------------------------------------------------------
    # AI — Cloud LLM
    # -------------------------------------------------------------------------
    anthropic_api_key: str = Field(default="", description="Anthropic Claude API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    grok_api_key: str = Field(default="", description="xAI Grok API key")
    llm_provider: Literal["anthropic", "openai", "grok"] = Field(
        default="anthropic",
        description="Which cloud LLM provider to use for summarization / translation",
    )

    # Model identifiers for cloud usage
    anthropic_model: str = Field(default="claude-opus-4-6")
    openai_model: str = Field(default="gpt-4o-mini")
    grok_model: str = Field(default="grok-3-mini-fast")
    grok_base_url: str = Field(default="https://api.x.ai/v1")

    # Maximum tokens the LLM may generate per summary/translation call
    llm_max_tokens: int = Field(default=512, ge=64, le=4096)

    # -------------------------------------------------------------------------
    # AI — Local Models  (RTX 3080 Ti)
    # -------------------------------------------------------------------------
    local_model_device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="PyTorch device for local inference (cuda or cpu)",
    )
    sentiment_model_en: str = Field(
        default="ProsusAI/finbert",
        description="HuggingFace model ID for English financial sentiment analysis",
    )
    sentiment_model_ko: str = Field(
        default="snunlp/KR-FinBert-SC",
        description="HuggingFace model ID for Korean financial sentiment analysis",
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "SentenceTransformers model ID for 384-dim embeddings. "
            "Default is a multilingual model that handles Korean + English "
            "both monolingually and cross-lingually. Swap to all-MiniLM-L6-v2 "
            "for English-only speedups if cross-lingual search is unused."
        ),
    )
    embedding_dim: int = Field(
        default=384,
        description="Output dimensionality of the embedding model (must match pgvector column)",
    )

    # Batch sizes for GPU inference — tune based on available VRAM
    sentiment_batch_size: int = Field(default=32, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1)

    # -------------------------------------------------------------------------
    # Collection / scheduling
    # -------------------------------------------------------------------------
    collection_interval_minutes: int = Field(
        default=15,
        ge=1,
        description="How often (in minutes) Celery Beat triggers the news collection pipeline",
    )

    # Maximum concurrent HTTP requests during RSS / API polling
    collector_concurrency: int = Field(default=10, ge=1, le=50)

    # Per-request timeout for external HTTP calls (seconds)
    http_timeout: float = Field(default=30.0, ge=1.0)

    # -------------------------------------------------------------------------
    # Search / ranking weights
    # -------------------------------------------------------------------------
    rank_weight_bm25: float = Field(default=0.6, ge=0.0, le=1.0)
    rank_weight_cosine: float = Field(default=0.3, ge=0.0, le=1.0)
    rank_weight_recency: float = Field(default=0.1, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("rank_weight_bm25", "rank_weight_cosine", "rank_weight_recency")
    @classmethod
    def weights_sum_to_one(cls, v: float, info) -> float:  # noqa: ANN001
        # Individual weight validation only; cross-field sum check would require
        # a model_validator but we keep it simple — weights are adjusted at runtime.
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Ranking weight must be in [0, 1], got {v}")
        return v

    # -------------------------------------------------------------------------
    # Derived helpers
    # -------------------------------------------------------------------------
    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url

    @property
    def is_cuda(self) -> bool:
        return self.local_model_device == "cuda"

    def active_llm_api_key(self) -> str:
        """Return the API key for the currently configured LLM provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        if self.llm_provider == "grok":
            return self.grok_api_key
        return self.openai_api_key

    def active_llm_model(self) -> str:
        """Return the model identifier for the currently configured LLM provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        if self.llm_provider == "grok":
            return self.grok_model
        return self.openai_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    The result is cached so that environment variables and .env are parsed
    only once per process.  In tests, call ``get_settings.cache_clear()``
    before patching env vars.
    """
    return Settings()
