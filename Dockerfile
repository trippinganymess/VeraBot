# syntax=docker/dockerfile:1.6

# --- Build stage --------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# --- Runtime stage ------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VERA_HOST=0.0.0.0 \
    VERA_PORT=8080

# Non-root user for prod safety
RUN groupadd --system vera && useradd --system --gid vera --home /app vera

WORKDIR /app

# Copy installed packages from the build stage
COPY --from=builder /install /usr/local

# Copy the application source. Tests, datasets, and dev-only assets are
# excluded via .dockerignore.
COPY --chown=vera:vera bot.py semantic_matcher.py llm_pool.py ./

USER vera

EXPOSE 8080

# Liveness probe matches what the judge harness polls every 60s
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:'+__import__('os').getenv('VERA_PORT','8080')+'/v1/healthz', timeout=3).status==200 else 1)"

CMD ["sh", "-c", "uvicorn bot:app --host ${VERA_HOST} --port ${VERA_PORT} --workers 1 --proxy-headers --forwarded-allow-ips=*"]
