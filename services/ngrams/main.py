"""
Main module for N-grams analysis Celery application.

This module initializes the Celery app and loads configuration.
All tasks should import the app instance from this module.
"""

from celery import Celery

# Create Celery app instance
app = Celery("teledash-processing-ngrams")

# Load configuration from config module
app.config_from_object("config")

# Initialize database connection (if needed for worker startup)
try:
    from worker.database import Database
    database = Database()
except Exception as e:
    # Database connection will be established per-task if needed
    database = None

# Import tasks to register them with Celery
import worker.ngrams.tasks  # noqa: F401
