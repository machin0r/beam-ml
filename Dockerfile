# Multi-stage Dockerfile for LPBF Density Prediction API
# Stage 1: Builder - Install dependencies
# Stage 2: Runtime - Slim production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.12-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and README (needed for package metadata)
COPY pyproject.toml README.md ./

# Copy source code
COPY src/ ./src/
COPY api/ ./api/

# Install dependencies
RUN uv pip install --system ".[api]"

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src /app/src
COPY --from=builder /app/api /app/api

# Copy model artifacts and feature statistics
COPY models/feature_schema.json /app/models/feature_schema.json
COPY models/production/ /app/models/production/
COPY reports/feature_space_stats.json /app/reports/feature_space_stats.json

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/production

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/api/v1/health')" || exit 1

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
