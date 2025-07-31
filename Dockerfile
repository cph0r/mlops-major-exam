# Multi-stage Docker build for California Housing Price Predictor
# Optimized for production deployment with security and performance considerations

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim-buster as base

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev \
        make \
        curl \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Stage 2: Dependencies installation
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application build
FROM dependencies as builder

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/output

# Set proper permissions
RUN chown -R mluser:mluser /app

# Stage 4: Production image
FROM python:3.9-slim-buster as production

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data /app/output /app/models && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "src/predict.py"]

# Stage 5: Development image (optional)
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Set development environment
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/output /app/models

# Set proper permissions
RUN chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Development command
CMD ["python", "-m", "pytest", "tests/", "-v"]