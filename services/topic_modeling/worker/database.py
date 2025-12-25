"""
Database operations for Topic Modeling service.

This module handles Elasticsearch operations for storing and retrieving
topic modeling results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from celery.utils.log import get_task_logger
from elasticsearch import Elasticsearch

from common.database.index_alias import IndexAlias
from common.settings import settings

logger = get_task_logger(__name__)


class Database:
    """Database client for topic modeling operations."""

    def __init__(self):
        """Initialize Elasticsearch client."""
        self.client = Elasticsearch(
            [
                {
                    "host": settings.elastic_host,
                    "port": settings.elastic_port,
                    "scheme": "http",
                }
            ],
            timeout=300,
            max_retries=10,
            retry_on_timeout=True,
        )

    def ensure_topics_index(self) -> None:
        """Create topics index if it doesn't exist."""
        index_name = "topics"

        if self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists")
            return

        # Define mapping for topics
        mapping = {
            "mappings": {
                "properties": {
                    "topic_id": {"type": "keyword"},
                    "topic_number": {"type": "integer"},
                    "top_words": {"type": "keyword"},
                    "top_words_scores": {"type": "float"},
                    "representative_docs": {"type": "keyword"},
                    "size": {"type": "integer"},
                    "coherence_score": {"type": "float"},
                    "first_seen": {"type": "date"},
                    "last_seen": {"type": "date"},
                    "metadata": {
                        "properties": {
                            "primary_channels": {"type": "keyword"},
                            "extremism_category": {"type": "keyword"},
                            "language": {"type": "keyword"},
                        }
                    },
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                }
            }
        }

        self.client.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index: {index_name}")

    def store_topics(self, topics: List[Dict[str, Any]]) -> None:
        """
        Store topic information in Elasticsearch.

        Args:
            topics: List of topic dictionaries to store
        """
        self.ensure_topics_index()

        for topic in topics:
            doc_id = topic["topic_id"]
            topic["updated_at"] = datetime.utcnow().isoformat()

            if not topic.get("created_at"):
                topic["created_at"] = topic["updated_at"]

            self.client.index(
                index="topics",
                id=doc_id,
                document=topic
            )

        logger.info(f"Stored {len(topics)} topics in Elasticsearch")

    def update_message_topics(
        self,
        chat_id: str,
        message_topics: List[Dict[str, Any]]
    ) -> None:
        """
        Update messages with their topic assignments.

        Args:
            chat_id: Chat ID
            message_topics: List of dicts with message_id and topic assignments
        """
        index_name = f"messages_{chat_id}"

        if not self.client.indices.exists(index=index_name):
            logger.warning(f"Index {index_name} does not exist")
            return

        # Bulk update messages
        bulk_operations = []
        for msg_topic in message_topics:
            bulk_operations.append({
                "update": {"_index": index_name, "_id": msg_topic["message_id"]}
            })
            bulk_operations.append({
                "doc": {"topics": msg_topic["topics"]}
            })

        if bulk_operations:
            self.client.bulk(operations=bulk_operations, refresh=True)
            logger.info(f"Updated {len(message_topics)} messages with topic assignments")

    def get_messages_for_topic_modeling(
        self,
        chat_id: Optional[str] = None,
        min_length: int = 50,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages suitable for topic modeling.

        Args:
            chat_id: Optional chat ID to filter by
            min_length: Minimum character length for messages
            limit: Maximum number of messages to retrieve

        Returns:
            List of message documents
        """
        # Build query
        query = {
            "bool": {
                "must": [
                    {"exists": {"field": "text"}},
                ],
                "should": [],
                "filter": []
            }
        }

        # Add minimum length filter
        query["bool"]["filter"].append({
            "script": {
                "script": {
                    "source": "doc['text.keyword'].value.length() >= params.min_length",
                    "params": {"min_length": min_length}
                }
            }
        })

        # Determine which indices to search
        if chat_id:
            indices = [f"messages_{chat_id}"]
        else:
            # Get all message indices
            indices = self.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)

        if not indices:
            logger.warning("No message indices found")
            return []

        # Search across indices
        results = []
        for index in indices:
            try:
                response = self.client.search(
                    index=index,
                    query=query,
                    size=min(limit, 10000),
                    _source=["id", "text", "date", "chat_id"],
                    sort=[{"date": "desc"}]
                )

                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    results.append({
                        "id": source.get("id"),
                        "text": source.get("text"),
                        "date": source.get("date"),
                        "chat_id": source.get("chat_id"),
                    })

                if len(results) >= limit:
                    break

            except Exception as e:
                logger.error(f"Error searching index {index}: {e}")
                continue

        logger.info(f"Retrieved {len(results)} messages for topic modeling")
        return results[:limit]

    def get_all_indices_by_alias(self, alias: IndexAlias) -> List[str]:
        """
        Get all indices matching an alias pattern.

        Args:
            alias: Index alias to match

        Returns:
            List of index names
        """
        try:
            aliases = self.client.cat.aliases(format="json")
            matching_indices = [
                a["index"] for a in aliases if a["alias"] == alias.value
            ]
            return matching_indices
        except Exception as e:
            logger.error(f"Error getting indices for alias {alias}: {e}")
            return []

    def get_topic_by_id(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a topic by its ID.

        Args:
            topic_id: Topic identifier

        Returns:
            Topic document or None if not found
        """
        try:
            response = self.client.get(index="topics", id=topic_id)
            return response["_source"]
        except Exception as e:
            logger.warning(f"Topic {topic_id} not found: {e}")
            return None

    def get_all_topics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all topics.

        Args:
            limit: Maximum number of topics to return

        Returns:
            List of topic documents
        """
        try:
            response = self.client.search(
                index="topics",
                size=limit,
                sort=[{"size": "desc"}]
            )

            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error retrieving topics: {e}")
            return []

    def get_messages_by_topic(
        self,
        topic_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get messages assigned to a specific topic.

        Args:
            topic_id: Topic identifier
            limit: Maximum number of messages to return

        Returns:
            List of message documents
        """
        query = {
            "nested": {
                "path": "topics",
                "query": {
                    "term": {"topics.topic_id": topic_id}
                }
            }
        }

        indices = self.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)

        results = []
        for index in indices:
            try:
                response = self.client.search(
                    index=index,
                    query=query,
                    size=min(limit - len(results), 100),
                    _source=["id", "text", "date", "chat_id", "topics"]
                )

                for hit in response["hits"]["hits"]:
                    results.append(hit["_source"])

                if len(results) >= limit:
                    break

            except Exception as e:
                logger.error(f"Error searching index {index}: {e}")
                continue

        return results
