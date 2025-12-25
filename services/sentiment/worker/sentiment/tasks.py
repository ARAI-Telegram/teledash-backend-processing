"""
Celery tasks for sentiment analysis processing.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from celery import Task
from celery.utils.log import get_task_logger

# Note: These imports will need to be adapted based on actual common module structure
# from common.database.index_alias import IndexAlias
# from common.redis_queue_manager import QueueChecker
# from common.settings import settings
# from common.utils import get_chat_id_from_index_name
# from worker.database import Database
from worker.model_init import init_sentiment_model
from worker.sentiment.analysis import (
    analyze_sentiment_batch as _analyze_sentiment,
    create_sentiment_result_document
)

logger = get_task_logger(__name__)


class SentimentTask(Task):
    """
    Abstraction of Celery's Task class to support loading sentiment model.
    Model is loaded once and shared across all task instances in the worker process.
    """

    abstract = True
    _model = None  # Class-level variable shared across instances

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e., first task processed).
        Avoids the need to load model on each task request.
        """
        if SentimentTask._model is None:
            logger.info("Loading sentiment analysis model...")
            try:
                SentimentTask._model = init_sentiment_model()
                logger.info("Sentiment model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load sentiment model: {e}")
                logger.error("Critical error: Cannot proceed without model.")
                raise
        return self.run(*args, **kwargs)

    @property
    def model(self):
        """Get the shared model instance."""
        return SentimentTask._model


# Import Celery app from main module (configured with Redis broker)
from main import app


@app.task(
    name="sentiment.init_sentiment_analysis",
    autoretry_for=(Exception,),
    retry_kwargs={
        "max_retries": 5,
        "countdown": 300,
    },  # Retry every 5 minutes, max 5 times
)
def init_sentiment_analysis(
    chat_ids: Optional[List[str]] = None,
    message_ids: Optional[List[str]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    reanalyze: bool = False
) -> None:
    """
    Initialize sentiment analysis process for messages in the database.

    This task identifies unanalyzed messages and enqueues them for sentiment analysis.
    Skips if there are already pending tasks in the sentiment queue.

    Args:
        chat_ids: Optional list of specific chat IDs to analyze
        message_ids: Optional list of specific message IDs to analyze
        date_from: Optional start date for filtering messages
        date_to: Optional end date for filtering messages
        reanalyze: If True, reanalyze already processed messages
    """
    logger.info("Starting sentiment analysis initialization...")
    logger.info(f"Parameters: chat_ids={chat_ids}, message_ids={message_ids}, "
                f"date_from={date_from}, date_to={date_to}, reanalyze={reanalyze}")

    # Validate model availability before queuing any tasks
    logger.info("Validating sentiment model availability...")
    try:
        init_sentiment_model(load_to_ram=False)
        logger.info("Sentiment model validation successful.")
    except Exception as e:
        logger.error(f"Sentiment model validation failed: {e}")
        logger.error("Cannot initialize sentiment analysis - model unavailable.")
        raise

    # Connect to database
    from worker.database import Database
    db = Database()
    es = db.client

    # Build query for messages
    must_conditions = []
    must_not_conditions = []

    # Filter by chat_ids if provided
    if chat_ids:
        must_conditions.append({"terms": {"chat_id": chat_ids}})

    # Filter by message_ids if provided
    if message_ids:
        must_conditions.append({"terms": {"_id": message_ids}})

    # Filter by date range if provided
    if date_from or date_to:
        date_range = {}
        if date_from:
            date_range["gte"] = date_from.isoformat()
        if date_to:
            date_range["lte"] = date_to.isoformat()
        must_conditions.append({"range": {"date": date_range}})

    # Ensure text field exists
    must_conditions.append({"exists": {"field": "text"}})

    # Skip messages that already have sentiment (unless reanalyze=True)
    if not reanalyze:
        must_not_conditions.append({"exists": {"field": "sentiment"}})

    query = {
        "bool": {
            "must": must_conditions if must_conditions else [{"match_all": {}}],
            "must_not": must_not_conditions
        }
    }

    # Count messages to process
    try:
        count_response = es.count(index="messages", query=query)
        total_messages = count_response["count"]
        logger.info(f"Found {total_messages} messages to analyze")

        if total_messages == 0:
            logger.info("No messages to analyze, exiting")
            return

        # Fetch messages in batches and enqueue tasks
        batch_size = 100  # Process 100 messages per task
        tasks_queued = 0

        # Use scroll API for large result sets
        response = es.search(
            index="messages",
            query=query,
            size=batch_size,
            scroll="5m",
            _source=["text", "date", "chat_id", "language"]
        )

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        while hits:
            # Prepare batch
            messages = []
            for hit in hits:
                msg_id = hit["_id"]
                source = hit["_source"]
                messages.append({
                    "id": msg_id,
                    "text": source.get("text", ""),
                    "date": source.get("date"),
                    "chat_id": source.get("chat_id"),
                    "language": source.get("language", "de")
                })

            # Group by chat_id for efficient processing
            from collections import defaultdict
            by_chat = defaultdict(list)
            for msg in messages:
                by_chat[msg["chat_id"]].append(msg)

            # Enqueue batch tasks
            for chat_id, chat_messages in by_chat.items():
                message_batch = {
                    "chat_id": chat_id,
                    "messages": chat_messages
                }
                analyze_sentiment_batch.delay(message_batch)
                tasks_queued += 1

            # Get next batch
            if scroll_id:
                response = es.scroll(scroll_id=scroll_id, scroll="5m")
                scroll_id = response.get("_scroll_id")
                hits = response["hits"]["hits"]
            else:
                break

        # Clear scroll
        if scroll_id:
            es.clear_scroll(scroll_id=scroll_id)

        logger.info(f"Sentiment analysis initialization complete: {tasks_queued} batch tasks queued for {total_messages} messages")

    except Exception as e:
        logger.error(f"Error during sentiment analysis initialization: {e}", exc_info=True)
        raise


@app.task(
    name="sentiment.analyze_batch",
    bind=True,
    base=SentimentTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
    default_retry_delay=60 * 60,  # Retry after 60 minutes
)
def analyze_sentiment_batch(self, message_batch: Dict[str, Any]) -> None:
    """
    Analyze sentiment for a batch of messages.

    Args:
        message_batch: Dictionary with structure:
            {
                "chat_id": str,
                "messages": [
                    {"id": str, "text": str, "date": str, "language": str},
                    ...
                ]
            }
    """
    chat_id = message_batch.get("chat_id")
    messages = message_batch.get("messages", [])

    if not messages:
        logger.warning("Empty message batch received, skipping.")
        return

    logger.info(f"Analyzing sentiment for {len(messages)} messages from chat {chat_id}")

    try:
        # Extract texts for batch processing
        texts = [msg["text"] for msg in messages if msg.get("text")]

        if not texts:
            logger.warning("No valid texts found in batch, skipping.")
            return

        # Run sentiment analysis
        sentiment_results = _analyze_sentiment(texts, self.model)

        # Create Elasticsearch documents
        sentiment_docs = []
        message_updates = []

        for msg, sentiment in zip(messages, sentiment_results):
            # Create sentiment result document
            sentiment_doc = create_sentiment_result_document(
                message_id=msg["id"],
                chat_id=chat_id,
                date=msg.get("date", datetime.utcnow()),
                text_snippet=msg["text"][:200] if msg.get("text") else None,
                language=msg.get("language", "de"),
                sentiment=sentiment["sentiment"],
                confidence=sentiment["confidence"],
                scores=sentiment["scores"]
            )
            sentiment_docs.append(sentiment_doc)

            # Prepare message update (add sentiment field to message doc)
            message_updates.append({
                "message_id": msg["id"],
                "sentiment": {
                    "label": sentiment["sentiment"],
                    "confidence": sentiment["confidence"]
                }
            })

        # Bulk write to Elasticsearch
        from worker.database import Database
        db = Database()
        es = db.client

        # 1. Index sentiment_docs to sentiment_results index
        from elasticsearch.helpers import bulk

        sentiment_actions = [
            {
                "_index": "sentiment_results",
                "_id": doc["message_id"],  # Use message_id as document ID
                "_source": doc
            }
            for doc in sentiment_docs
        ]

        if sentiment_actions:
            success_count, errors = bulk(es, sentiment_actions, raise_on_error=False)
            if errors:
                logger.warning(f"Some sentiment results failed to index: {errors}")
            logger.info(f"Indexed {success_count} sentiment results to sentiment_results index")

        # 2. Update message documents with sentiment field
        # Use per-chat index instead of alias (messages_{chat_id})
        message_index = f"messages_{chat_id}" if chat_id else "messages"
        update_actions = [
            {
                "_op_type": "update",
                "_index": message_index,
                "_id": update["message_id"],
                "doc": {"sentiment": update["sentiment"]},
                "doc_as_upsert": False
            }
            for update in message_updates
        ]

        if update_actions:
            success_count, errors = bulk(es, update_actions, raise_on_error=False)
            if errors:
                logger.warning(f"Some message updates failed: {errors}")
            logger.info(f"Updated {success_count} messages with sentiment data")

        logger.info(f"Successfully analyzed {len(sentiment_docs)} messages")
        logger.info(f"Sentiment distribution: {_count_sentiments(sentiment_docs)}")

    except Exception as e:
        logger.error(f"Error analyzing sentiment batch: {e}", exc_info=True)
        raise


@app.task(
    name="sentiment.compute_stats",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
)
def compute_sentiment_stats(
    chat_ids: Optional[List[str]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Compute aggregate sentiment statistics.

    Uses Elasticsearch aggregations for efficient computation.

    Args:
        chat_ids: Optional filter by chat IDs
        date_from: Optional start date
        date_to: Optional end date

    Returns:
        Dictionary with sentiment statistics
    """
    logger.info("Computing sentiment statistics...")

    # TODO: Implement Elasticsearch aggregation queries
    # This is a placeholder return structure

    return {
        "total_messages": 0,
        "sentiment_breakdown": {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        },
        "average_score": 0.0,
        "average_confidence": 0.0,
        "by_chat": []
    }


def _count_sentiments(sentiment_docs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Helper function to count sentiment distribution in a batch."""
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for doc in sentiment_docs:
        sentiment = doc.get("sentiment", "neutral")
        if sentiment in counts:
            counts[sentiment] += 1
    return counts
