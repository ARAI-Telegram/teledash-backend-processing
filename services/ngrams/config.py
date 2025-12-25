"""
Celery configuration for N-grams analysis service.
"""

from datetime import timedelta
from common.settings import settings

# Redis broker configuration
broker_url = f"redis://redis-processing:{settings.redis_port}/0"

# Task routing
task_routes = {
    "ngrams.*": {"queue": "ngrams"},
}

# Result backend
result_backend = broker_url

# Task serialization
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "UTC"
enable_utc = True

# Celery Beat Schedule (optional periodic tasks)
beat_schedule = {
    # Periodic n-gram generation could be added here if needed
    # Example:
    # "generate-ngrams": {
    #     "task": "ngrams.generate_ngrams",
    #     "schedule": timedelta(hours=6),
    # }
}
