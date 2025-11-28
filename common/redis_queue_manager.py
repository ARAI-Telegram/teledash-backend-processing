import redis
from celery.utils.log import get_task_logger

from common.settings import settings

logger = get_task_logger(__name__)


class QueueChecker:
    """
    Simple queue checker to prevent flooding Celery queues.
    Checks if queues are empty before adding more tasks.
    """

    def __init__(self):
        self.redis_client = redis.Redis(
            host="redis-processing", port=settings.redis_port, decode_responses=True
        )

    def is_queue_empty(self, queue_name: str) -> bool:
        """
        Check if a Celery queue is empty.

        Args:
            queue_name: Name of the queue to check (e.g., 'text_embeddings', 'classification', 'asr')

        Returns:
            True if queue is empty, False if it has pending tasks
        """
        try:
            queue_length = self.redis_client.llen(queue_name)
            logger.debug(f"Queue '{queue_name}' has {queue_length} pending tasks")
            return queue_length == 0
        except Exception as e:
            logger.error(f"Error checking queue '{queue_name}': {e}")
            # If we can't check, assume queue is not empty to be safe
            return False

    def get_queue_length(self, queue_name: str) -> int:
        """
        Get the number of pending tasks in a queue.
        This is a synchronous method that returns the length of the queue.
        Args:
            queue_name: Name of the queue to check

        Returns:
            Number of pending tasks, or -1 if error
        """
        try:
            return self.redis_client.llen(queue_name)  # type: ignore[return-value] # the client is sync so no Awaitable will be returned
        except Exception as e:
            logger.error(f"Error getting queue length for '{queue_name}': {e}")
            return -1
