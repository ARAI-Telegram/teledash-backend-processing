# Evaluate build arguments first
ARG GPU_USE=false
ARG CONTAINER_TYPE=worker

# Multi-stage build: select base image based on GPU_USE
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base-true
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

FROM python:3.10-slim AS base-false

# Select the appropriate base (base-true for GPU, base-false for CPU)
FROM base-${GPU_USE} AS final

# Re-declare ARGs for this build stage
ARG GPU_USE=false
ARG CONTAINER_TYPE=worker

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

COPY ./common/ /app/common/
COPY ./services/asr/ /app/

RUN python -m pip install -r common/requirements/requirements.txt \
    && python -m pip install -r common/requirements/requirements.celery.txt

# Install ML dependencies only for worker container
# (CUDA libraries are provided by base image when GPU_USE=true)
RUN if [ "$CONTAINER_TYPE" = "worker" ]; then \
        echo "Installing ML requirements for worker..." && \
        pip install --no-cache-dir -r requirements.txt; \
    elif [ "$CONTAINER_TYPE" = "beat" ]; then \
        echo "Skipping ML dependencies for beat container"; \
    else \
        echo "CONTAINER_TYPE not set, no special requirements will be installed."; \
    fi

# Create tmp directory structure with proper permissions
RUN mkdir -p /app/tmp/.cache && chmod -R 755 /app/tmp

# Create non-root user and ensure ownership
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

# Set 'appuser' as the default user
USER appuser
