# Export Service Dockerfile

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim as base

# Re-declare ARG after FROM to make it available in this stage
ARG COMPUTE_TYPE=cpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies for weasyprint
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Copy common module
COPY ./common/ /app/common/

# Copy service files
COPY ./services/export/ /app/

# Install common requirements
RUN pip install --no-cache-dir -r /app/common/requirements/requirements.txt && \
    pip install --no-cache-dir -r /app/common/requirements/requirements.celery.txt

# Install service-specific requirements
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["celery", "-A", "worker.export.tasks", "worker", "--loglevel=info", "--queues=export"]
