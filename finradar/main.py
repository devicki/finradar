"""
finradar.main
~~~~~~~~~~~~~

FastAPI application factory for the FinRadar API.

Startup sequence
----------------
1. Build the FastAPI app with metadata.
2. Register CORS middleware (open in development; tighten in production).
3. Mount API routers under /api/v1/.
4. Expose a /health liveness probe.

Running locally
---------------
    uvicorn finradar.main:app --reload --host 0.0.0.0 --port 8000

Or via the project script entry-point:
    finradar
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finradar.api.routes import feed, news, search
from finradar.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinRadar",
    description=(
        "Global Economic News Intelligence Platform.\n\n"
        "Collects financial news from multiple sources, enriches it with "
        "AI-powered sentiment analysis (FinBERT), automatic summarisation "
        "and translation (Claude/GPT-4o-mini), and exposes a hybrid "
        "search API backed by PostgreSQL FTS + pgvector."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production to known origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(news.router, prefix="/api/v1/news", tags=["news"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(feed.router, prefix="/api/v1/feed", tags=["feed"])

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    tags=["meta"],
    summary="Liveness probe",
    description="Returns HTTP 200 and the current application version.  "
    "Use this endpoint for container health checks.",
)
async def health_check() -> dict:
    return {"status": "ok", "version": settings.app_version}


# ---------------------------------------------------------------------------
# Script entry-point  (pyproject.toml: finradar = "finradar.main:main")
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the Uvicorn server (convenience entry-point)."""
    import uvicorn

    uvicorn.run(
        "finradar.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
