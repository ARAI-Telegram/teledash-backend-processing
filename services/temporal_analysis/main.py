"""Main Celery app for Temporal Analysis service."""

from celery import Celery
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

app = Celery("teledash-processing-temporal-analysis")
app.config_from_object("config")

if __name__ == "__main__":
    app.start()
