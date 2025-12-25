ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy common module and requirements first
COPY common /workspace/common

# Copy service-specific requirements
ARG COMPUTE_TYPE=cpu
COPY services/topic_modeling/requirements.${COMPUTE_TYPE}.txt /workspace/requirements.txt

# Install Python dependencies
# First install common requirements, then service-specific ones
RUN pip install --no-cache-dir -r /workspace/common/requirements/requirements.txt && \
    pip install --no-cache-dir -r /workspace/common/requirements/requirements.celery.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/topic_modeling /workspace

# Create model directory
RUN mkdir -p /tmp/topic_models && chmod 755 /tmp/topic_models

ENV PYTHONPATH=/workspace
ENV CONTAINER_TYPE=worker

CMD ["celery", "-A", "main", "worker", "--loglevel=INFO", "--concurrency=1", "-Q", "topic_modeling"]
