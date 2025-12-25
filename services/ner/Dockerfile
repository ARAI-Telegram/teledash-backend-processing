# Named Entity Recognition Service Dockerfile

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

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy common module
COPY ./common/ /app/common/

# Copy service files
COPY ./services/ner/ /app/

# Install common requirements
RUN pip install --no-cache-dir -r /app/common/requirements/requirements.txt && \
    pip install --no-cache-dir -r /app/common/requirements/requirements.celery.txt

# Install service-specific requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy German language model
RUN python -m pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl

# Default command
CMD ["celery", "-A", "worker.ner.tasks", "worker", "--loglevel=info", "--queues=ner"]
