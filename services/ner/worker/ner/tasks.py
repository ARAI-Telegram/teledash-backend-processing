"""
Celery tasks for Named Entity Recognition processing.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from celery import Task
from celery.utils.log import get_task_logger

from worker.model_init import init_ner_model
from worker.ner.extraction import (
    extract_entities_batch,
    compute_cooccurrences
)
from worker.ner.network_builder import build_entity_network

logger = get_task_logger(__name__)


class NERTask(Task):
    """
    Abstraction of Celery's Task class to support loading NER model.
    Model is loaded once and shared across all task instances in the worker process.
    """

    abstract = True
    _model = None  # Class-level variable shared across instances

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e., first task processed).
        Avoids the need to load model on each task request.
        """
        if NERTask._model is None:
            logger.info("Loading NER model...")
            try:
                NERTask._model = init_ner_model()
                logger.info("NER model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
                logger.error("Critical error: Cannot proceed without model.")
                raise
        return self.run(*args, **kwargs)

    @property
    def model(self):
        """Get the shared model instance."""
        return NERTask._model


# Import Celery app from main module (configured with Redis broker)
from main import app


@app.task(
    name="ner.init_ner_extraction",
    autoretry_for=(Exception,),
    retry_kwargs={
        "max_retries": 5,
        "countdown": 300,
    },  # Retry every 5 minutes, max 5 times
)
def init_ner_extraction(
    chat_ids: Optional[List[str]] = None,
    message_ids: Optional[List[str]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    reanalyze: bool = False
) -> None:
    """
    Initialize NER extraction process for messages in the database.

    This task identifies unprocessed messages and enqueues them for entity extraction.
    Skips if there are already pending tasks in the NER queue.

    Args:
        chat_ids: Optional list of specific chat IDs to process
        message_ids: Optional list of specific message IDs to process
        date_from: Optional start date for filtering messages
        date_to: Optional end date for filtering messages
        reanalyze: If True, reprocess already analyzed messages
    """
    logger.info("Starting NER extraction initialization...")
    logger.info(f"Parameters: chat_ids={chat_ids}, message_ids={message_ids}, "
                f"date_from={date_from}, date_to={date_to}, reanalyze={reanalyze}")

    # Validate model availability before queuing any tasks
    logger.info("Validating NER model availability...")
    try:
        init_ner_model(load_to_ram=False)
        logger.info("NER model validation successful.")
    except Exception as e:
        logger.error(f"NER model validation failed: {e}")
        logger.error("Cannot initialize NER - model unavailable.")
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

    # Skip messages that already have entities (unless reanalyze=True)
    if not reanalyze:
        must_not_conditions.append({"exists": {"field": "entities"}})

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
        logger.info(f"Found {total_messages} messages to process for NER")

        if total_messages == 0:
            logger.info("No messages to process, exiting")
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
                extract_entities_batch_task.delay(message_batch)
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

        logger.info(f"NER extraction initialization complete: {tasks_queued} batch tasks queued for {total_messages} messages")

    except Exception as e:
        logger.error(f"Error during NER extraction initialization: {e}", exc_info=True)
        raise


@app.task(
    name="ner.extract_entities_batch",
    bind=True,
    base=NERTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
    default_retry_delay=60 * 60,  # Retry after 60 minutes
)
def extract_entities_batch_task(self, message_batch: Dict[str, Any]) -> None:
    """
    Extract named entities from a batch of messages.

    Args:
        message_batch: Dictionary with structure:
            {
                "chat_id": str,
                "messages": [
                    {"id": str, "text": str, "date": str},
                    ...
                ]
            }
    """
    chat_id = message_batch.get("chat_id")
    messages = message_batch.get("messages", [])

    if not messages:
        logger.warning("Empty message batch received, skipping.")
        return

    logger.info(f"Extracting entities from {len(messages)} messages from chat {chat_id}")

    try:
        # Extract entities
        entity_mentions, unique_entities = extract_entities_batch(messages, self.model)

        if not entity_mentions:
            logger.info("No entities found in batch, skipping.")
            return

        # Compute co-occurrences
        cooccurrences = compute_cooccurrences(entity_mentions, min_cooccurrence=2)

        # Add co-occurrence data to entities
        for entity in unique_entities:
            entity_id = entity["entity_id"]
            entity["co_occurring_entities"] = cooccurrences.get(entity_id, [])

        # Bulk write to Elasticsearch
        from worker.database import Database
        from elasticsearch.helpers import bulk
        db = Database()
        es = db.client

        # 1. Index entity_mentions to entity_mentions index
        mention_actions = [
            {
                "_index": "entity_mentions",
                "_source": mention
            }
            for mention in entity_mentions
        ]

        if mention_actions:
            success_count, errors = bulk(es, mention_actions, raise_on_error=False)
            if errors:
                logger.warning(f"Some entity mentions failed to index: {errors}")
            logger.info(f"Indexed {success_count} entity mentions")

        # 2. Upsert entities to entities index (merge with existing)
        entity_actions = [
            {
                "_op_type": "update",
                "_index": "entities",
                "_id": entity["entity_id"],
                "doc": entity,
                "doc_as_upsert": True
            }
            for entity in unique_entities
        ]

        if entity_actions:
            success_count, errors = bulk(es, entity_actions, raise_on_error=False)
            if errors:
                logger.warning(f"Some entities failed to upsert: {errors}")
            logger.info(f"Upserted {success_count} unique entities")

        # 3. Update message documents with entities field
        # Group entities by message
        from collections import defaultdict
        entities_by_message = defaultdict(list)
        for mention in entity_mentions:
            entities_by_message[mention["message_id"]].append({
                "text": mention["text"],
                "type": mention["type"],
                "start": mention.get("start"),
                "end": mention.get("end")
            })

        update_actions = [
            {
                "_op_type": "update",
                "_index": "messages",
                "_id": msg_id,
                "doc": {"entities": entities},
                "doc_as_upsert": False
            }
            for msg_id, entities in entities_by_message.items()
        ]

        if update_actions:
            success_count, errors = bulk(es, update_actions, raise_on_error=False)
            if errors:
                logger.warning(f"Some message updates failed: {errors}")
            logger.info(f"Updated {success_count} messages with entity data")

        logger.info(f"Successfully extracted {len(entity_mentions)} entity mentions")
        logger.info(f"Entity type distribution: {_count_entity_types(entity_mentions)}")

    except Exception as e:
        logger.error(f"Error extracting entities: {e}", exc_info=True)
        raise


@app.task(
    name="ner.build_entity_network",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
)
def build_entity_network_task(
    entity_ids: Optional[List[str]] = None,
    chat_ids: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    min_cooccurrence: int = 2,
    max_nodes: int = 100
) -> Dict[str, Any]:
    """
    Build entity co-occurrence network for visualization.

    Args:
        entity_ids: Optional filter by specific entity IDs
        chat_ids: Optional filter by chat IDs
        entity_types: Optional filter by entity types
        min_cooccurrence: Minimum co-occurrence count for edges
        max_nodes: Maximum nodes in network

    Returns:
        Network graph with nodes and edges
    """
    logger.info("Building entity network...")

    # TODO: Query entities from Elasticsearch with filters
    # Placeholder: return empty network
    entities = []

    network = build_entity_network(
        entities,
        entity_types=entity_types,
        min_cooccurrence=min_cooccurrence,
        max_nodes=max_nodes
    )

    return network


@app.task(
    name="ner.consolidate_entities",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
)
def consolidate_entities() -> None:
    """
    Consolidate duplicate entities using fuzzy matching.

    This task identifies and merges similar entities (e.g., "Angela Merkel" and "A. Merkel").
    Uses Levenshtein distance for fuzzy matching.
    """
    logger.info("Starting entity consolidation...")

    # TODO: Implement fuzzy matching logic
    # 1. Query all entities grouped by type
    # 2. For each type, compute pairwise similarity
    # 3. Merge entities above similarity threshold
    # 4. Update entity_mentions to point to consolidated entity_id
    # 5. Delete old entity documents

    logger.info("Entity consolidation placeholder - implementation pending")


def _count_entity_types(entity_mentions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Helper function to count entity type distribution."""
    counts = {}
    for mention in entity_mentions:
        entity_type = mention.get("entity_type", "MISC")
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts
