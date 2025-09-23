FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ARTIFACTS_DIR=model_artifacts

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy app source and artifacts
COPY . /app

# (Optional) basic healthcheck hitting /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-10000}/health || exit 1

# Use shell form so $PORT expands on Render; default to 10000 locally
CMD sh -c 'uvicorn mta_models:app --host 0.0.0.0 --port ${PORT:-10000}'
