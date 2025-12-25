"""
Celery tasks for N-gram analysis processing.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from celery import Task
from celery.utils.log import get_task_logger

from worker.ngrams.generator import (
    generate_ngrams_from_corpus,
    compute_word_frequency
)

logger = get_task_logger(__name__)


# Import Celery app from main module (configured with Redis broker)
from main import app


@app.task(
    name="ngrams.generate_ngrams",
    autoretry_for=(Exception,),
    retry_kwargs={
        "max_retries": 3,
        "countdown": 300,
    },
)
def generate_ngrams(
    chat_ids: Optional[List[str]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    n_values: List[int] = [2, 3],
    min_frequency: int = 5,
    exclude_stopwords: bool = True,
    stopwords_language: str = "german"
) -> Dict[str, Any]:
    """
    Generate n-grams for specified corpus.

    This task fetches messages from Elasticsearch, generates n-grams,
    and stores the results in the ngrams index.

    Args:
        chat_ids: Optional filter by chat IDs
        date_from: Optional start date
        date_to: Optional end date
        n_values: List of n values (1, 2, 3)
        min_frequency: Minimum frequency threshold
        exclude_stopwords: Whether to exclude stopwords
        stopwords_language: Language for stopwords

    Returns:
        Summary of n-grams generated
    """
    logger.info("Starting n-gram generation...")
    logger.info(f"Parameters: chat_ids={chat_ids}, date_from={date_from}, date_to={date_to}")
    logger.info(f"N-values: {n_values}, Min frequency: {min_frequency}")

    try:
        # Connect to database
        from worker.database import Database
        db = Database()
        es = db.client

        # Build query for messages
        must_conditions = []

        # Filter by chat_ids if provided
        if chat_ids:
            must_conditions.append({"terms": {"chat_id": chat_ids}})

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

        query = {
            "bool": {
                "must": must_conditions if must_conditions else [{"match_all": {}}]
            }
        }

        # Count messages to process
        count_response = es.count(index="messages", query=query)
        total_messages = count_response["count"]
        logger.info(f"Found {total_messages} messages for n-gram generation")

        if total_messages == 0:
            logger.info("No messages found for n-gram generation")
            return {
                "status": "completed",
                "ngrams_generated": 0,
                "message": "No messages found"
            }

        # Fetch messages using scroll API for large result sets
        messages = []
        batch_size = 1000
        response = es.search(
            index="messages",
            query=query,
            size=batch_size,
            scroll="5m",
            _source=["text", "chat_id"]
        )

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        while hits:
            for hit in hits:
                messages.append({
                    "id": hit["_id"],
                    "text": hit["_source"].get("text", ""),
                    "chat_id": hit["_source"].get("chat_id")
                })

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

        logger.info(f"Fetched {len(messages)} messages for n-gram analysis")

        # Generate n-grams
        ngrams = generate_ngrams_from_corpus(
            messages,
            n_values=n_values,
            min_frequency=min_frequency,
            exclude_stopwords=exclude_stopwords,
            stopwords_language=stopwords_language
        )

        if not ngrams:
            logger.info("No n-grams generated (all below frequency threshold)")
            return {
                "status": "completed",
                "ngrams_generated": 0,
                "message": "No n-grams met minimum frequency threshold"
            }

        # Bulk write n-grams to Elasticsearch ngrams index
        from elasticsearch.helpers import bulk

        ngram_actions = [
            {
                "_index": "ngrams",
                "_id": ngram["ngram_id"],
                "_source": ngram
            }
            for ngram in ngrams
        ]

        if ngram_actions:
            success_count, errors = bulk(es, ngram_actions, raise_on_error=False)
            if errors:
                logger.warning(f"Some n-grams failed to index: {errors}")
            logger.info(f"Indexed {success_count} n-grams to ngrams index")

        logger.info(f"Successfully generated {len(ngrams)} n-grams")

        # Return summary
        return {
            "status": "completed",
            "ngrams_generated": len(ngrams),
            "n_values": n_values,
            "messages_processed": len(messages),
            "distribution": _count_by_n(ngrams)
        }

    except Exception as e:
        logger.error(f"Error generating n-grams: {e}", exc_info=True)
        raise


@app.task(
    name="ngrams.word_frequency",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
)
def compute_word_frequency_task(
    chat_ids: Optional[List[str]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_frequency: int = 5,
    exclude_stopwords: bool = True,
    stopwords_language: str = "german",
    limit: int = 1000
) -> Dict[str, Any]:
    """
    Compute word frequency distribution for corpus.

    Args:
        chat_ids: Optional filter by chat IDs
        date_from: Optional start date
        date_to: Optional end date
        min_frequency: Minimum frequency threshold
        exclude_stopwords: Whether to exclude stopwords
        stopwords_language: Language for stopwords
        limit: Maximum words to return

    Returns:
        Word frequency distribution
    """
    logger.info("Computing word frequencies...")

    try:
        # TODO: Query Elasticsearch for messages
        messages = []

        if not messages:
            logger.info("No messages found for word frequency")
            return {
                "status": "completed",
                "words": [],
                "total_words": 0
            }

        # Compute frequencies
        word_frequencies = compute_word_frequency(
            messages,
            min_frequency=min_frequency,
            exclude_stopwords=exclude_stopwords,
            stopwords_language=stopwords_language,
            limit=limit
        )

        logger.info(f"Computed frequencies for {len(word_frequencies)} words")

        return {
            "status": "completed",
            "words": word_frequencies,
            "total_words": len(word_frequencies),
            "messages_processed": len(messages)
        }

    except Exception as e:
        logger.error(f"Error computing word frequencies: {e}", exc_info=True)
        raise


@app.task(
    name="ngrams.get_ngram_messages",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
)
def get_ngram_messages(
    ngram_text: str,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get messages containing a specific n-gram.

    Args:
        ngram_text: N-gram text to search for
        limit: Maximum messages to return

    Returns:
        List of messages containing the n-gram
    """
    logger.info(f"Fetching messages for n-gram: {ngram_text}")

    try:
        # TODO: Query Elasticsearch for messages containing ngram_text
        # Use phrase query for exact match

        messages = []

        return {
            "ngram": ngram_text,
            "messages": messages,
            "total": len(messages)
        }

    except Exception as e:
        logger.error(f"Error fetching n-gram messages: {e}", exc_info=True)
        raise


def _count_by_n(ngrams: List[Dict[str, Any]]) -> Dict[int, int]:
    """Helper to count n-grams by n value."""
    counts = {}
    for ngram in ngrams:
        n = ngram.get("n", 0)
        counts[n] = counts.get(n, 0) + 1
    return counts
