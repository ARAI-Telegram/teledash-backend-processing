# Sentiment Analysis Service Dockerfile

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
COPY ./services/sentiment/ /app/

# Install common requirements
RUN pip install --no-cache-dir -r /app/common/requirements/requirements.txt && \
    pip install --no-cache-dir -r /app/common/requirements/requirements.celery.txt

# Install service-specific requirements based on compute type
RUN pip install --no-cache-dir -r requirements.${COMPUTE_TYPE}.txt

# Create directory for model cache
RUN mkdir -p /app/models

# Set Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/app/models

# Download sentiment model at build time (optional - can also download at runtime)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('oliverguhr/german-sentiment-bert'); \
    AutoModelForSequenceClassification.from_pretrained('oliverguhr/german-sentiment-bert')"

# Default command
CMD ["celery", "-A", "worker.sentiment.tasks", "worker", "--loglevel=info", "--queues=sentiment"]
