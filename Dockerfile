# syntax=docker/dockerfile:1.4

########################
# Build stage: download wheels for deps
########################
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages needed at build time (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN python -m pip install --upgrade pip setuptools wheel

# Pre-build dependency wheels to speed up installs and keep runtime slim
RUN mkdir -p /wheels && \
    pip wheel --wheel-dir /wheels \
      fastapi \
      "uvicorn[standard]" \
      lightgbm \
      numpy \
      Click \
      h11 \
      starlette \
      pydantic-core \
      pydantic \
      typing-extensions \
      typing-inspection \
      annotated-types \
      anyio \
      sniffio \
      scipy

########################
# Runtime stage
########################
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Install minimal runtime deps: libgomp for LightGBM, git available in container, tini as init
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      git \
      ca-certificates \
      tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from prebuilt wheels
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels /wheels/*.whl

# Copy application code
COPY . /app

# Create non-root user and make sure key dirs exist
RUN addgroup --system app && adduser --system --ingroup app appuser && \
    mkdir -p /app/data /app/logs /app/models && \
    chown -R appuser:app /app

USER appuser

# Expose default port
EXPOSE 8000

# Environment defaults (can be overridden)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=/app/models/model.txt \
    LOG_LEVEL=INFO

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini","--"]

# Start the API (single process by default to avoid duplicate schedulers)
CMD ["python","-m","uvicorn","ml_server.app.main:app","--host","0.0.0.0","--port","8000"]
