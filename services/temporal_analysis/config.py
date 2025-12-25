"""Configuration for Temporal Analysis service."""

import os

from common.settings import settings

broker_url = f"redis://redis-processing:{settings.redis_port}/0"

# Only import task and its dependencies in worker container
if os.getenv("CONTAINER_TYPE", "worker") == "worker":
    imports = []  # Add task imports when tasks are created
else:
    print("Beat container - skipping task import")

# Send task-related events so that tasks can be monitored using tools like flower.
worker_send_task_events = True

task_track_started = True
broker_connection_retry_on_startup = True

# maps task names set in @task decorator
task_routes = {
    "temporal_analysis.*": {"queue": "temporal_analysis"},
}

timezone = "UTC"

# Temporal Analysis specific settings
TEMPORAL_ANOMALY_CONTAMINATION = float(os.getenv("TEMPORAL_ANOMALY_CONTAMINATION", "0.1"))
TEMPORAL_FORECAST_PERIODS = int(os.getenv("TEMPORAL_FORECAST_PERIODS", "7"))
TEMPORAL_TIME_GRANULARITY = os.getenv("TEMPORAL_TIME_GRANULARITY", "hour")
