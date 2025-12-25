FROM python:3.11-slim

WORKDIR /app

# Copy common module
COPY common/ /app/common/

# Copy monitoring service
COPY services/monitoring/requirements.txt /app/requirements.txt
COPY services/monitoring/api/ /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8003

# Run FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
