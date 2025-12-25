FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy common module
COPY common /workspace/common

# Copy service requirements
COPY services/network_analysis/requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /workspace/common/requirements/requirements.txt && \
    pip install --no-cache-dir -r /workspace/common/requirements/requirements.celery.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/network_analysis /workspace

ENV PYTHONPATH=/workspace
ENV CONTAINER_TYPE=worker

CMD ["celery", "-A", "main", "worker", "--loglevel=INFO", "--concurrency=2", "-Q", "network_analysis"]
