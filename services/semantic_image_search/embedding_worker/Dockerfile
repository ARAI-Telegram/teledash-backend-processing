FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Evaluate build arguments and set default value
ARG CONTAINER_TYPE=worker

COPY ./common/ /app/common/
COPY ./services/semantic_image_search/embedding_worker /app/
COPY ./services/semantic_image_search/shared/ /app/shared/

# Install pip requirements
RUN python -m pip install -r /app/common/requirements/requirements.txt \
    && python -m pip install -r /app/common/requirements/requirements.celery.txt

# Install ML dependencies only for worker container 
RUN if [ "$CONTAINER_TYPE" = "worker" ]; then \
        echo "Installing requirements for worker..." && \
        pip install --no-cache-dir -r /app/shared/requirements.cpu.txt; \
    elif [ "$CONTAINER_TYPE" = "beat" ]; then \
        echo "Skipping ML dependencies for beat container"; \
    else \
        echo "CONTAINER_TYPE not set, no special requirements will be installed."; \
    fi

# Create tmp directory structure with proper permissions
RUN mkdir -p /app/tmp/model && chmod -R 755 /app/tmp

# Create non-root user and ensure ownership
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

# Set 'appuser' as the default user
USER appuser