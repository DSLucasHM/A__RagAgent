# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy requirements and install with uv (caching pip downloads)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

# Copy only installed deps and uv binary
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ /app/src/


# Copy .documentsend
COPY ./src/documents /app/documents/

# Create non-root user, cache dir, and set ownership
RUN addgroup --system app \
 && adduser --system --ingroup app --home /app app \
 && mkdir -p /app/.cache/huggingface \
 && chown -R app:app /app

# Point HF and Transformers caches to a writable location
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

USER app

# Run the CLI application
CMD ["python", "-m", "src.main"]
