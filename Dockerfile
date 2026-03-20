# ─────────────────────────────────────────────────────────────────────────────
# runtime-base: ONLY runtime .so libraries — no headers, no static libs
#   Kept minimal so the production image inherits nothing extra.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips42t64 \
    libjxl0.11 \
    libwebp7 \
    libwebpmux3 \
    libwebpdemux2 \
    libexif12 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# build-base: adds -dev headers + compiler tools on top of runtime-base.
#   Nothing from this stage leaks into the production image.
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS build-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips-dev \
    libjxl-dev \
    libwebp-dev \
    libexif-dev \
    build-essential \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ─────────────────────────────────────────────────────────────────────────────
# deps-prod: production venv only (no pytest, no dev tools)
# ─────────────────────────────────────────────────────────────────────────────
FROM build-base AS deps-prod

WORKDIR /build
COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-install-project --no-dev

# ─────────────────────────────────────────────────────────────────────────────
# deps-dev: all deps including pytest/httpx (for dev & CI)
# Separate stage so shebang paths are always /opt/venv/bin/*
# ─────────────────────────────────────────────────────────────────────────────
FROM build-base AS deps-dev

WORKDIR /build
COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-install-project

# ─────────────────────────────────────────────────────────────────────────────
# dev: daily development
#   - pre-built venv copied from deps (fast rebuild on dep change)
#   - build tools kept so `uv add` / `uv sync` work inside the container
#   - source code is volume-mounted — never baked in
# ─────────────────────────────────────────────────────────────────────────────
FROM build-base AS dev

COPY --from=deps-dev /opt/venv /opt/venv
WORKDIR /app
COPY pyproject.toml ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONPATH=/app \
    VIPS_WARNING=0

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/app"]

# ─────────────────────────────────────────────────────────────────────────────
# test: CI test runner
#   - runtime-base only (no build tools, no uv binary)
#   - dev venv baked in (has pytest, httpx, etc.)
#   - full source including tests baked in
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS test

COPY --from=deps-dev /opt/venv /opt/venv
WORKDIR /app
COPY pyproject.toml ./
COPY app/       ./app/
COPY tests/     ./tests/
COPY conftest.py ./

ENV PYTHONPATH=/app \
    VIPS_WARNING=0

CMD ["/opt/venv/bin/pytest", "tests/", "-v"]

# ─────────────────────────────────────────────────────────────────────────────
# app: production
#   - runtime-base only (no build tools, no uv binary)
#   - prod venv only (no pytest, no dev tools)
#   - only app/ source code baked in (no tests)
#   - CMD uses venv path directly — no uv needed
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime-base AS app

COPY --from=deps-prod /opt/venv /opt/venv
WORKDIR /app
COPY app/ ./app/

ENV PYTHONPATH=/app \
    VIPS_WARNING=0

EXPOSE 8000
CMD ["/opt/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

