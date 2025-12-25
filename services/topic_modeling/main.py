from celery import Celery
from celery.utils.log import get_task_logger
from worker.database import Database

logger = get_task_logger(__name__)

app = Celery("teledash-processing-topic-modeling")
app.config_from_object("config")

try:
    database = Database()
    if not database.client.ping():
        raise ConnectionError("Elasticsearch ping failed.")
    logger.info("Connected to the database.")
except Exception as e:
    logger.error(f"Error during database connection: {e}", exc_info=True)

if __name__ == "__main__":
    app.start()
