# ==============================================================================
# VoltEcho AI - Production Docker Container
# Lightweight, hardened Python 3.12-slim runtime for Acoustic RL Control Platform
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output for live logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for healthchecks and C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (compact ~150MB wheel instead of 800MB CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files, models, and frontend assets into container
COPY . .

# Ensure contacts log file exists with write permissions
RUN touch contacts_inquiries.json && chmod 666 contacts_inquiries.json

# Expose HTTP port
EXPOSE 8000

# Container Healthcheck verifying FastAPI and model readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Start production ASGI server with uvicorn listening to dynamic $PORT
CMD ["sh", "-c", "uvicorn web.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
