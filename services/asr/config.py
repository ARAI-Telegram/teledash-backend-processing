import os
from datetime import timedelta

from common.settings import settings

broker_url = f"redis://redis-processing:{settings.redis_port}/0"

# Only import task and its dependencies in worker container
if os.getenv("CONTAINER_TYPE", "worker") == "worker":
    imports = ["worker.tasks"]
else:
    print("Beat container - skipping task import")

# maps task names set in @task decorator
task_routes = {
    "asr.*": {"queue": "asr"},
}

# Send task-related events so that tasks can be monitored
# using tools like flower.
worker_send_task_events = True

task_track_started = True
broker_connection_retry_on_startup = True

beat_schedule = {
    "transcribe-speech": {
        "task": "asr.init_asr",
        "schedule": timedelta(minutes=settings.asr_interval_minutes),
    }
}

timezone = "UTC"
