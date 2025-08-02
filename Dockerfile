# Multi-stage container build for Real Estate Valuation System
# Optimized for production deployment with enhanced security and performance

# Stage 1: Foundation image with Python and system dependencies
FROM python:3.9-slim-buster as foundation

# Configure Python environment for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and security patches
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

# Create non-privileged user for enhanced security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Stage 2: Dependency installation
FROM foundation as deps

# Set application directory
WORKDIR /application

# Copy dependency specification
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application compilation
FROM deps as compiler

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create required directories
RUN mkdir -p /application/logs /application/data /application/output

# Set appropriate permissions
RUN chown -R appuser:appuser /application

# Stage 4: Production container
FROM python:3.9-slim-buster as prod

# Copy Python packages from compiler stage
COPY --from=compiler /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=compiler /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=compiler /application /application

# Create non-privileged user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set application directory
WORKDIR /application

# Configure environment variables
ENV PYTHONPATH=/application/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create required directories and set permissions
RUN mkdir -p /application/logs /application/data /application/output /application/models && \
    chown -R appuser:appuser /application

# Switch to non-privileged user
USER appuser

# Health monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default execution command
CMD ["python", "src/inference.py"]

# Stage 5: Development container (optional)
FROM deps as dev

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy entire codebase
COPY . .

# Set development environment
ENV PYTHONPATH=/application/src \
    PYTHONUNBUFFERED=1

# Create required directories
RUN mkdir -p /application/logs /application/data /application/output /application/models

# Set appropriate permissions
RUN chown -R appuser:appuser /application

# Switch to non-privileged user
USER appuser

# Development execution command
CMD ["python", "-m", "pytest", "tests/", "-v"]