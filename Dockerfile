# =============================================================================
# FinRadar — Multi-stage Dockerfile
# Base: Python 3.11-slim
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies into a virtual env
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System dependencies required to build psycopg2/asyncpg and other native exts
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first so Docker can cache this layer
COPY pyproject.toml ./

# Create virtual environment and install all project dependencies.
# We install in "no-deps" for the project itself at the end so we don't
# pull in a stale editable install.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install \
        # Core web framework
        "fastapi>=0.104.0" \
        "uvicorn[standard]>=0.24.0" \
        "pydantic>=2.5.0" \
        "pydantic-settings>=2.1.0" \
        # Database
        "sqlalchemy[asyncio]>=2.0.23" \
        "asyncpg>=0.29.0" \
        "psycopg2-binary>=2.9.9" \
        "alembic>=1.13.0" \
        "pgvector>=0.2.4" \
        # HTTP & RSS
        "httpx>=0.25.0" \
        "feedparser>=6.0.0" \
        "tweepy>=4.14.0" \
        # AI - Local models (CPU wheel; swap to GPU wheel via pip install --extra-index-url)
        "transformers>=4.36.0" \
        "torch>=2.1.0" \
        "sentence-transformers>=2.2.0" \
        # AI - Cloud LLMs
        "anthropic>=0.8.0" \
        "openai>=1.6.0" \
        "langchain>=0.1.0" \
        "langchain-anthropic>=0.1.0" \
        "langchain-openai>=0.1.0" \
        # Task queue
        "celery[redis]>=5.3.0" \
        "redis>=5.0.0" \
        # Utilities
        "python-dotenv>=1.0.0" \
        "trafilatura>=1.6.0" \
        "pdfplumber>=0.10.0" \
        # Dashboard (dev/debugging UI)
        "streamlit>=1.30.0" \
        "plotly>=5.18.0" \
        "pandas>=2.1.0"

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system libraries only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 finradar && \
    useradd --uid 1001 --gid finradar --shell /bin/bash --create-home finradar

WORKDIR /app

# Bring the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate venv for all subsequent RUN / CMD instructions
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

# Copy project source
COPY --chown=finradar:finradar . .

# Install the project package itself (editable during dev via volume mount)
RUN pip install --no-deps -e .

USER finradar

EXPOSE 8000

# Default command — can be overridden in docker-compose.yml
CMD ["uvicorn", "finradar.main:app", "--host", "0.0.0.0", "--port", "8000"]
