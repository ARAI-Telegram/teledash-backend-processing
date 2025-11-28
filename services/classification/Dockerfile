FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Evaluate build arguments and set default values
ARG GPU_USE=false
ARG CONTAINER_TYPE=worker

# Install shared pip requirements
COPY ./common/ /app/common/
COPY ./services/classification/ /app/

RUN python -m pip install -r /app/common/requirements/requirements.txt \
    && python -m pip install -r /app/common/requirements/requirements.celery.txt

# Install ML dependencies only for worker container and based on GPU_USE
RUN if [ "$CONTAINER_TYPE" = "worker" ]; then \
        if [ "$GPU_USE" = "true" ]; then \
            echo "Installing GPU requirements for worker..." && \
            pip install --no-cache-dir -r requirements.gpu.txt && \
            export NVIDIA_VISIBLE_DEVICES=all && \
            export NVIDIA_DRIVER_CAPABILITIES=compute,utility; \
        else \
            echo "Installing CPU requirements for worker..." && \
            pip install --no-cache-dir -r requirements.cpu.txt; \
        fi; \
    elif [ "$CONTAINER_TYPE" = "beat" ]; then \
        echo "Skipping ML dependencies for beat container"; \
    else \
        echo "CONTAINER_TYPE not set, no special requirements will be installed."; \
    fi

# Create tmp directory structure with proper permissions
RUN mkdir -p /app/tmp/model && chmod -R 755 /app/tmp

# Create non-root user and ensure ownership
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

USER appuser