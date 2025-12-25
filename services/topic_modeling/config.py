import os
from datetime import timedelta

from common.settings import settings

broker_url = f"redis://redis-processing:{settings.redis_port}/0"

# Only import task and its dependencies in worker container
if os.getenv("CONTAINER_TYPE", "worker") == "worker":
    imports = ["worker.topic_modeling.tasks"]
else:
    print("Beat container - skipping task import")

# Send task-related events so that tasks can be monitored using tools like flower.
worker_send_task_events = True

task_track_started = True
broker_connection_retry_on_startup = True

# maps task names set in @task decorator
task_routes = {
    "topic_modeling.*": {"queue": "topic_modeling"},
}

beat_schedule = {
    "fit-topic-models": {
        "task": "topic_modeling.init_topic_modeling",
        "schedule": timedelta(minutes=settings.topic_update_interval_minutes),
    }
}

timezone = "UTC"
