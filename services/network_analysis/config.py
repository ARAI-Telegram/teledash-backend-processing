"""Configuration for Network Analysis service."""

import os
from datetime import timedelta

from common.settings import settings

broker_url = f"redis://redis-processing:{settings.redis_port}/0"

# Only import task and its dependencies in worker container
if os.getenv("CONTAINER_TYPE", "worker") == "worker":
    imports = ["worker.network_analysis.tasks"]
else:
    print("Beat container - skipping task import")

# Send task-related events so that tasks can be monitored using tools like flower.
worker_send_task_events = True

task_track_started = True
broker_connection_retry_on_startup = True

# maps task names set in @task decorator
task_routes = {
    "network_analysis.*": {"queue": "network_analysis"},
}

timezone = "UTC"

# Network Analysis specific settings
NETWORK_MIN_EDGE_WEIGHT = int(os.getenv("NETWORK_MIN_EDGE_WEIGHT", "2"))
NETWORK_UPDATE_INTERVAL_MINUTES = int(os.getenv("NETWORK_UPDATE_INTERVAL_MINUTES", "240"))
NETWORK_EXPORT_FORMAT = os.getenv("NETWORK_EXPORT_FORMAT", "graphml")
NETWORK_COORDINATION_TIME_WINDOW = int(os.getenv("NETWORK_COORDINATION_TIME_WINDOW", "3600"))
