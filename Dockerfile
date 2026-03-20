# ─────────────────────────────────────────────────────────────────────────────
# runtime-base: system libs only — rebuilt only when system deps change
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips-dev \
    libjxl-dev \
    libwebp-dev \
    libexif-dev \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# deps-builder: compiles Python packages (pyvips needs gcc for cffi bindings)
# This layer is cached as long as pyproject.toml doesn't change.
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS deps-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml ./
# Install all deps (prod + dev) into /opt/venv — no source code needed yet
RUN uv sync --no-install-project

# ─────────────────────────────────────────────────────────────────────────────
# dev: for daily development
#   - pre-built /opt/venv copied from deps-builder (fast, no recompile)
#   - build tools kept so `uv sync` works when deps change inside the container
#   - source code is volume-mounted at runtime (never baked in)
#   - uvicorn runs with --reload for instant hot-reload on file save
#
# Usage:
#   docker compose -f docker-compose.dev.yml up --build   # first time (~2 min)
#   docker compose -f docker-compose.dev.yml up           # subsequent starts (~5 s)
#   docker compose -f docker-compose.dev.yml run --rm app uv run pytest tests/ -v
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=deps-builder /opt/venv /opt/venv

WORKDIR /app
# pyproject.toml is needed by `uv run`; the volume mount will overlay it at
# runtime, but having it here avoids an error on first `uv run`.
COPY pyproject.toml ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONPATH=/app \
    VIPS_WARNING=0

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/app"]

# ─────────────────────────────────────────────────────────────────────────────
# app (production): no build tools, code baked in
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=deps-builder /opt/venv /opt/venv

WORKDIR /app
COPY pyproject.toml ./
COPY app/       ./app/
COPY tests/     ./tests/
COPY conftest.py ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONPATH=/app \
    VIPS_WARNING=0

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

