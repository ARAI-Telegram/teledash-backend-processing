"""
Celery tasks for Topic Modeling service.
"""

from datetime import datetime
from typing import Optional

from celery import Task
from celery.utils.log import get_task_logger

from common.redis_queue_manager import QueueChecker
from common.settings import settings
from main import app
from worker.database import Database
from worker.model_init import create_topic_model, init_topic_model, save_topic_model
from worker.topic_modeling.topic_model import (
    assign_topics_to_messages,
    extract_topic_info,
    fit_topic_model_on_messages,
)

logger = get_task_logger(__name__)


class TopicModelingTask(Task):
    """
    Abstraction of Celery's Task class to support loading topic model.
    """

    abstract = True
    _model = None  # Class-level variable shared across instances

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed).
        Avoids the need to load model on each task request.
        """
        if TopicModelingTask._model is None:
            logger.info("Initializing topic model...")
            try:
                TopicModelingTask._model = init_topic_model()
                logger.info("Topic model initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to load existing topic model: {e}")
                logger.info("Will create new model when needed")
        return self.run(*args, **kwargs)

    @property
    def model(self):
        """Get the shared model instance."""
        return TopicModelingTask._model

    @model.setter
    def model(self, value):
        """Set the shared model instance."""
        TopicModelingTask._model = value


@app.task(
    name="topic_modeling.init_topic_modeling",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 600},
)
def init_topic_modeling(
    min_topic_size: int = None,
    max_topics: Optional[int] = None,
    min_message_length: int = None,
    message_limit: int = None
) -> None:
    """
    Initialize topic modeling process.
    This task identifies messages for topic modeling and enqueues processing.

    Args:
        min_topic_size: Minimum number of messages to form a topic (default from settings)
        max_topics: Maximum number of topics to discover (default auto)
        min_message_length: Minimum character length for messages (default from settings)
        message_limit: Maximum messages to process in one batch (default 10000)
    """
    # Use defaults from settings if not provided
    min_topic_size = min_topic_size if min_topic_size is not None else settings.topic_min_size
    min_message_length = min_message_length if min_message_length is not None else settings.topic_min_char_length
    message_limit = message_limit if message_limit is not None else 10000

    logger.info(f"Starting topic modeling initialization with config: "
                f"min_topic_size={min_topic_size}, max_topics={max_topics}, "
                f"min_message_length={min_message_length}, message_limit={message_limit}")

    # Validate database connectivity
    try:
        db = Database()
        if not db.client.ping():
            raise ConnectionError("Elasticsearch ping failed.")
        logger.info("Connected to database for topic modeling initialization")
    except Exception as e:
        logger.error(f"Error during database connection: {e}")
        raise

    queue_checker = QueueChecker()

    # Check if queue already has pending tasks
    if not queue_checker.is_queue_empty("topic_modeling"):
        pending_count = queue_checker.get_queue_length("topic_modeling")
        logger.info(
            f"Skipping init_topic_modeling - queue already has {pending_count} pending tasks"
        )
        return

    logger.info("Topic modeling queue is empty, starting initialization...")

    # Get messages for topic modeling
    try:
        messages = db.get_messages_for_topic_modeling(
            chat_id=None,  # Process all chats
            min_length=min_message_length,
            limit=message_limit
        )

        if not messages:
            logger.info("No messages found for topic modeling")
            return

        logger.info(f"Found {len(messages)} messages for topic modeling")

        # Enqueue topic modeling task with configuration
        fit_and_store_topics.apply_async(
            kwargs={
                "messages": messages,
                "min_topic_size": min_topic_size,
                "max_topics": max_topics
            },
            queue="topic_modeling"
        )

        logger.info("Topic modeling task enqueued successfully")

    except Exception as e:
        logger.error(f"Error during topic modeling initialization: {e}")
        raise


@app.task(
    name="topic_modeling.fit_and_store_topics",
    base=TopicModelingTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 1800},
)
def fit_and_store_topics(
    messages: list,
    min_topic_size: int = None,
    max_topics: Optional[int] = None
) -> dict:
    """
    Fit topic model on messages and store results.

    Args:
        messages: List of message dictionaries
        min_topic_size: Minimum number of messages to form a topic
        max_topics: Maximum number of topics to discover

    Returns:
        Dictionary with processing results
    """
    if not messages:
        logger.warning("No messages provided for topic modeling")
        return {"status": "skipped", "reason": "no_messages"}

    logger.info(f"Fitting topic model on {len(messages)} messages with "
                f"min_topic_size={min_topic_size}, max_topics={max_topics}")

    # Get or create topic model
    topic_model = TopicModelingTask._model

    if topic_model is None:
        logger.info("No existing model found, creating new model")
        use_gpu = settings.gpu_use
        topic_model = create_topic_model(
            use_gpu=use_gpu,
            min_topic_size=min_topic_size,
            max_topics=max_topics
        )
        TopicModelingTask._model = topic_model

    try:
        # Fit topic model
        topic_model, topics, probs = fit_topic_model_on_messages(
            messages, topic_model
        )

        # Update class-level model
        TopicModelingTask._model = topic_model

        # Extract topic information
        topic_info = extract_topic_info(topic_model, messages, topics)

        # Store topics in database
        db = Database()
        db.store_topics(topic_info)

        # Assign topics to messages
        message_topics = assign_topics_to_messages(messages, topics, probs)

        # Group by chat_id and update messages
        chat_message_map = {}
        for mt in message_topics:
            chat_id = mt.get("chat_id")
            if chat_id:
                if chat_id not in chat_message_map:
                    chat_message_map[chat_id] = []
                chat_message_map[chat_id].append(mt)

        # Update each chat's messages
        for chat_id, chat_messages in chat_message_map.items():
            db.update_message_topics(chat_id, chat_messages)

        # Save model to disk
        save_topic_model(topic_model)

        logger.info(
            f"Topic modeling complete. Discovered {len(topic_info)} topics, "
            f"assigned to {len(message_topics)} messages"
        )

        return {
            "status": "success",
            "num_topics": len(topic_info),
            "num_messages_assigned": len(message_topics),
            "num_messages_processed": len(messages),
        }

    except Exception as e:
        logger.error(f"Error during topic modeling: {e}")
        raise


@app.task(
    name="topic_modeling.update_topic_assignments",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 300},
)
def update_topic_assignments(chat_id: Optional[str] = None) -> dict:
    """
    Update topic assignments for existing messages using current model.

    Args:
        chat_id: Optional chat ID to update. If None, updates all chats.

    Returns:
        Dictionary with update results
    """
    logger.info(f"Updating topic assignments for chat_id={chat_id}")

    # Get topic model
    topic_model = TopicModelingTask._model

    if topic_model is None:
        logger.warning("No topic model available for updating assignments")
        return {"status": "skipped", "reason": "no_model"}

    try:
        # Get messages
        db = Database()
        messages = db.get_messages_for_topic_modeling(
            chat_id=chat_id,
            min_length=settings.topic_min_char_length,
            limit=10000
        )

        if not messages:
            logger.info("No messages found for update")
            return {"status": "skipped", "reason": "no_messages"}

        # Transform (predict) topics without refitting
        texts = [msg["text"] for msg in messages]
        topics, probs = topic_model.transform(texts)

        # Assign topics to messages
        message_topics = assign_topics_to_messages(messages, topics, probs)

        # Update database
        chat_message_map = {}
        for mt in message_topics:
            c_id = mt.get("chat_id")
            if c_id:
                if c_id not in chat_message_map:
                    chat_message_map[c_id] = []
                chat_message_map[c_id].append(mt)

        for c_id, chat_messages in chat_message_map.items():
            db.update_message_topics(c_id, chat_messages)

        logger.info(f"Updated {len(message_topics)} message topic assignments")

        return {
            "status": "success",
            "num_messages_updated": len(message_topics),
        }

    except Exception as e:
        logger.error(f"Error updating topic assignments: {e}")
        raise


@app.task(
    name="topic_modeling.compute_topic_evolution",
    base=TopicModelingTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 600},
)
def compute_topic_evolution(
    topic_id: str,
    time_granularity: str = "week"
) -> dict:
    """
    Compute how a topic evolves over time.

    Args:
        topic_id: Topic identifier (e.g., "topic_0001")
        time_granularity: Time granularity ('day', 'week', 'month')

    Returns:
        Dictionary with evolution data
    """
    logger.info(f"Computing evolution for topic {topic_id} with granularity {time_granularity}")

    # Get topic model
    topic_model = TopicModelingTask._model

    if topic_model is None:
        logger.warning("No topic model available")
        return {"status": "error", "reason": "no_model"}

    try:
        # Get all messages assigned to this topic
        db = Database()

        # Query messages with this topic assignment
        query = {
            "nested": {
                "path": "topics",
                "query": {
                    "term": {"topics.topic_id": topic_id}
                }
            }
        }

        # Search with large limit to get all messages
        response = db.client.search(
            index="messages_*",
            query=query,
            size=10000,
            _source=["id", "text", "date", "chat_id", "topics"],
            sort=[{"date": {"order": "asc"}}]
        )

        messages = [hit["_source"] for hit in response["hits"]["hits"]]

        if not messages:
            logger.info(f"No messages found for topic {topic_id}")
            return {"status": "success", "evolution": []}

        # Extract topic number from topic_id
        topic_number = int(topic_id.split("_")[-1])

        # Create topics list (all same topic)
        topics = [topic_number] * len(messages)

        # Track topics over time
        from worker.topic_modeling.topic_model import track_topics_over_time
        evolution_df = track_topics_over_time(
            topic_model, messages, topics, granularity=time_granularity
        )

        if evolution_df.empty:
            logger.warning("No evolution data generated")
            return {"status": "success", "evolution": []}

        # Convert to list of dicts for storage
        evolution_data = []
        for _, row in evolution_df.iterrows():
            if row.get("Topic") == topic_number:
                evolution_data.append({
                    "timestamp": row["Timestamp"].isoformat() if hasattr(row["Timestamp"], "isoformat") else str(row["Timestamp"]),
                    "frequency": int(row.get("Frequency", 0)),
                    "top_words": row.get("Words", "").split(", ")[:5] if isinstance(row.get("Words"), str) else []
                })

        # Store evolution data in topic document
        db.client.update(
            index="topics",
            id=topic_id,
            body={
                "doc": {
                    "evolution": evolution_data,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )

        logger.info(f"Stored {len(evolution_data)} evolution points for topic {topic_id}")

        return {
            "status": "success",
            "topic_id": topic_id,
            "evolution_points": len(evolution_data)
        }

    except Exception as e:
        logger.error(f"Error computing topic evolution: {e}")
        raise


@app.task(
    name="topic_modeling.build_topic_hierarchy",
    base=TopicModelingTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 900},
)
def build_topic_hierarchy() -> dict:
    """
    Build hierarchical topic structure using clustering.

    Uses topic embeddings to create a hierarchy of similar topics.

    Returns:
        Dictionary with hierarchy data
    """
    logger.info("Building topic hierarchy")

    # Get topic model
    topic_model = TopicModelingTask._model

    if topic_model is None:
        logger.warning("No topic model available")
        return {"status": "error", "reason": "no_model"}

    try:
        import numpy as np
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import pdist

        # Get all topics from database
        db = Database()
        response = db.client.search(
            index="topics",
            query={"match_all": {}},
            size=1000,
            _source=["topic_id", "topic_number", "top_words", "size"]
        )

        topics = [hit["_source"] for hit in response["hits"]["hits"]]

        if len(topics) < 2:
            logger.info("Not enough topics for hierarchy (need at least 2)")
            return {"status": "success", "hierarchy": None}

        # Get topic embeddings from model
        topic_embeddings = []
        topic_ids = []

        for topic in topics:
            topic_num = topic["topic_number"]

            # Get embedding for this topic
            try:
                # BERTopic stores topic embeddings internally
                if hasattr(topic_model, 'c_tf_idf_'):
                    embedding = topic_model.c_tf_idf_[topic_num + 1].toarray()[0]  # +1 because -1 is outliers
                    topic_embeddings.append(embedding)
                    topic_ids.append(topic["topic_id"])
            except:
                logger.warning(f"Could not get embedding for topic {topic_num}")
                continue

        if len(topic_embeddings) < 2:
            logger.warning("Not enough topic embeddings for hierarchy")
            return {"status": "success", "hierarchy": None}

        # Compute pairwise distances
        embeddings_array = np.array(topic_embeddings)
        distances = pdist(embeddings_array, metric='cosine')

        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')

        # Convert to tree structure
        def build_tree_dict(node, topic_ids):
            """Recursively build tree dictionary."""
            if node.is_leaf():
                return {
                    "id": topic_ids[node.id],
                    "type": "leaf",
                    "distance": 0
                }
            else:
                return {
                    "type": "branch",
                    "distance": float(node.dist),
                    "children": [
                        build_tree_dict(node.left, topic_ids),
                        build_tree_dict(node.right, topic_ids)
                    ]
                }

        root_node = to_tree(linkage_matrix)
        hierarchy_tree = build_tree_dict(root_node, topic_ids)

        # Store hierarchy in a special document
        hierarchy_doc = {
            "hierarchy_id": "topic_hierarchy_v1",
            "created_at": datetime.utcnow().isoformat(),
            "num_topics": len(topic_ids),
            "tree": hierarchy_tree
        }

        # Store in a separate index or as a document
        db.client.index(
            index="topic_metadata",
            id="topic_hierarchy_v1",
            document=hierarchy_doc
        )

        logger.info(f"Built topic hierarchy with {len(topic_ids)} topics")

        return {
            "status": "success",
            "num_topics": len(topic_ids),
            "hierarchy_id": "topic_hierarchy_v1"
        }

    except Exception as e:
        logger.error(f"Error building topic hierarchy: {e}")
        raise


@app.task(
    name="topic_modeling.merge_topics",
    base=TopicModelingTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 600},
)
def merge_topics(
    topic_ids: list,
    new_label: Optional[str] = None
) -> dict:
    """
    Merge multiple similar topics into one.

    Args:
        topic_ids: List of topic IDs to merge
        new_label: Optional label for merged topic

    Returns:
        Dictionary with merge results
    """
    logger.info(f"Merging topics: {topic_ids}")

    if len(topic_ids) < 2:
        logger.warning("Need at least 2 topics to merge")
        return {"status": "error", "reason": "insufficient_topics"}

    try:
        db = Database()

        # Get all topics to merge
        topics_to_merge = []
        for topic_id in topic_ids:
            response = db.client.get(index="topics", id=topic_id)
            topics_to_merge.append(response["_source"])

        # Create merged topic
        # Keep the first topic as base and merge others into it
        primary_topic = topics_to_merge[0]
        primary_topic_id = primary_topic["topic_id"]

        # Aggregate data from all topics
        total_size = sum(t["size"] for t in topics_to_merge)

        # Combine top words (deduplicated)
        all_words = []
        all_scores = []
        for topic in topics_to_merge:
            for word, score in zip(topic["top_words"], topic["top_words_scores"]):
                if word not in all_words:
                    all_words.append(word)
                    all_scores.append(score)

        # Keep top 10
        word_score_pairs = list(zip(all_words, all_scores))
        word_score_pairs.sort(key=lambda x: x[1], reverse=True)
        merged_words, merged_scores = zip(*word_score_pairs[:10]) if word_score_pairs else ([], [])

        # Update primary topic
        merged_topic_data = {
            "size": total_size,
            "top_words": list(merged_words),
            "top_words_scores": list(merged_scores),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": primary_topic.get("metadata", {})
        }

        if new_label:
            merged_topic_data["metadata"]["custom_label"] = new_label

        # Mark as merged
        merged_topic_data["metadata"]["merged_from"] = [t["topic_id"] for t in topics_to_merge[1:]]

        # Update primary topic
        db.client.update(
            index="topics",
            id=primary_topic_id,
            body={"doc": merged_topic_data}
        )

        # Update all messages assigned to merged topics
        # Reassign them to the primary topic
        for topic_id in topic_ids[1:]:
            # Find all messages with this topic
            query = {
                "nested": {
                    "path": "topics",
                    "query": {
                        "term": {"topics.topic_id": topic_id}
                    }
                }
            }

            # Update in batches
            response = db.client.search(
                index="messages_*",
                query=query,
                size=1000,
                _source=["id", "chat_id", "topics"]
            )

            messages_to_update = []
            for hit in response["hits"]["hits"]:
                msg = hit["_source"]
                # Replace old topic_id with primary_topic_id
                for topic_assignment in msg.get("topics", []):
                    if topic_assignment["topic_id"] == topic_id:
                        topic_assignment["topic_id"] = primary_topic_id

                messages_to_update.append({
                    "message_id": msg["id"],
                    "chat_id": msg["chat_id"],
                    "topics": msg["topics"]
                })

            # Group by chat and update
            chat_message_map = {}
            for mt in messages_to_update:
                chat_id = mt["chat_id"]
                if chat_id not in chat_message_map:
                    chat_message_map[chat_id] = []
                chat_message_map[chat_id].append(mt)

            for chat_id, chat_messages in chat_message_map.items():
                db.update_message_topics(chat_id, chat_messages)

            # Delete the merged topic
            db.client.delete(index="topics", id=topic_id)

        logger.info(f"Merged {len(topic_ids)} topics into {primary_topic_id}")

        return {
            "status": "success",
            "merged_topic_id": primary_topic_id,
            "num_topics_merged": len(topic_ids),
            "new_size": total_size
        }

    except Exception as e:
        logger.error(f"Error merging topics: {e}")
        raise
