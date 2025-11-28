# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install gcc (necessary to build uvicorn)
RUN apt-get update && apt-get install wget gcc -y

WORKDIR /app

COPY ./common/ /app/common/
COPY ./services/semantic_image_search/api/ /app/
COPY ./services/semantic_image_search/shared/ /app/shared/

# Install pip requirements
RUN python -m pip install -r /app/common/requirements/requirements.txt \
    && python -m pip install -r /app/common/requirements/requirements.celery.txt \
    && python -m pip install -r /app/common/requirements/requirements.fastapi.txt \
    && python -m pip install -r /app/shared/requirements.cpu.txt

# clean up build dependencies
RUN apt-get remove -y wget gcc && apt-get autoremove -y && apt-get clean

# Create tmp directory structure with proper permissions
RUN mkdir -p /app/tmp/model && chmod -R 755 /app/tmp

# Create non-root user and ensure ownership
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

# Set 'appuser' as the default user
USER appuser
