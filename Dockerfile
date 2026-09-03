# syntax=docker/dockerfile:1.7
# Single-container, air-gapped deployment. No external vector database:
# the index is a memory-mapped file under /data and the demo document store
# is in-process. Postgres is optional and configured via PERIMETER_DATABASE_URL.

FROM node:22-alpine AS admin
WORKDIR /admin
COPY admin/package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; else echo "no admin yet"; fi
COPY admin/ ./
RUN if [ -f package.json ]; then npm run build; else mkdir -p dist; fi

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY perimeter ./perimeter
COPY README.md ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
RUN useradd --create-home --uid 10001 perimeter
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/perimeter /app/perimeter
COPY --from=admin /admin/dist /app/admin/dist
ENV PATH="/app/.venv/bin:$PATH" \
    PERIMETER_DATA_DIR=/data \
    PERIMETER_ADMIN_DIST=/app/admin/dist \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
USER perimeter
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["perimeter", "serve", "--host", "0.0.0.0", "--port", "8000"]
