from datetime import datetime
from typing import List

from celery import Task
from celery.utils.log import get_task_logger
from main import app
from shared.model_init import init_embedding_model
from worker.database import Database

from common.database.index_alias import IndexAlias
from common.database.message import TextSearchMessage, VectorizedTextMessage
from common.redis_queue_manager import QueueChecker
from common.settings import settings
from common.utils import build_index_name, get_chat_id_from_index_name

logger = get_task_logger(__name__)

TEXT_EMBEDDINGS_QUEUE = "text-embeddings"


class EmbeddingTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """

    abstract = True
    _model = None  # Class-level variable shared across instances within this process

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if EmbeddingTask._model is None:
            logger.info("Loading embedding model for this worker process...")
            try:
                EmbeddingTask._model = init_embedding_model()
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                logger.error("Critical error: Cannot proceed without model.")
                raise  # Let Celery handle the error naturally
        return self.run(*args, **kwargs)

    @property
    def model(self):
        """Get the shared model instance."""
        return EmbeddingTask._model


@app.task(
    name="text_embeddings.generate_embeddings",
    bind=True,
    base=EmbeddingTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
    default_retry_delay=60 * 60,  # Retry after 60 minutes
)
def generate_embeddings(
    self,
    chat_id: str,
    message_list: List[dict],
) -> None:
    """
    Generate embeddings for a list of messages from a specific chat.

    Args:
        message_list (List[dict]): A list of dictionaries where each dictionary represents
        a serialized Message object.

    Returns:
        None

    The function performs the following steps:
    1. Converts the list of message dictionaries to a list of Message objects.
    2. Extracts text fields specified in the settings for embedding.
    3. Generates embeddings for the extracted text fields.
    4. Maps each embedding back to its corresponding message and field.
    5. Creates VectorizedMessage objects with the generated embeddings.
    6. Stores the VectorizedMessage objects in the database.
    7. Logs the completion of the batch embedding process.

    If an error occurs during the conversion of message dictionaries to Message objects,
    it logs the error and returns early.
    """
    db = Database()
    try:
        messages: List[TextSearchMessage] = [
            TextSearchMessage(**msg) for msg in message_list
        ]
    except Exception as e:
        logger.error(f"Error converting message list to Message objects: {e}")
        raise

    if not messages:
        logger.warning("No messages to embed.")
        return

    logger.info(f"Processing {len(messages)} messages for chat {chat_id}")

    embedding_fields = settings.text_embedding_fields
    texts = []
    message_meta_list = []  # To map embeddings back to messages and fields
    embedded_messages = []

    for message in messages:
        for field in embedding_fields:
            text_to_embed = getattr(message, field, None)
            if text_to_embed not in ["", None]:
                texts.append(text_to_embed)
                message_meta_list.append({"message_id": message.id, "field": field})

    if not texts:
        logger.info(f"No texts to embed for chat {chat_id}")
        return

    embedding_model = self.model
    embeddings = embedding_model.encode(
        sentences=texts,
        show_progress_bar=False,
        batch_size=settings.text_embedding_model_batch_size,
    )

    if not len(embeddings) == len(texts) == len(message_meta_list):
        logger.error("Mismatch in lengths of embeddings, texts, and message_meta_list.")
        return

    # Create VectorizedMessage objects
    for i, embedding in enumerate(embeddings):
        message_info = message_meta_list[i]
        message_id = message_info["message_id"]
        field = message_info["field"]

        original_message = next(msg for msg in messages if msg.id == message_id)

        if isinstance(embedding, list):
            embedding_value = embedding
        else:
            embedding_value = embedding.tolist()
        embedded_message = VectorizedTextMessage.create_from_message(
            message=original_message, embedding=(field, embedding_value)
        )

        embedded_messages.append(embedded_message)

    # Store embedded messages in db
    try:
        actions = db.create_index_actions_per_chat(chat_id, embedded_messages)
        db.bulk_write(
            actions=actions,
        )
        logger.info(
            f"Stored {len(embedded_messages)} embedded messages for chat {chat_id}"
        )
    except Exception as e:
        logger.error(f"Failed to store embedded messages for chat {chat_id}: {e}")
        raise


@app.task(
    name="text_embeddings.init_embeddings",
    autoretry_for=(Exception,),
    retry_kwargs={
        "max_retries": 5,
        "countdown": 300,
    },  # Retry every 5 minutes, max 5 times
)
def init_embeddings() -> None:
    """
    Initialize the embedding process for messages in the database.
    Skips if there are already pending tasks in the classification queue.

    This function retrieves messages, chat by chat, that need to be embedded,
    processes them in batches, and enqueues the batches for embedding generation.

    The function performs the following main steps:
    - Per chat, retrieves messages in batches from oldest to newest and always using date of the newest message in the batch as start date for the next batch,
        until no more messages are found.
    - Converts each message to a dictionary for serialization.
    - Enqueues each batch of messages for embedding generation.


    Raises:
        ValueError: If the retrieved messages are not in a list format.
        TypeError: If the date type of the message is unsupported.
    """
    # Validate model availability before queuing any tasks
    logger.info("Validating text embedding model availability...")
    try:
        init_embedding_model(
            load_to_ram=False
        )  # Lightweight validation - doesn't load the model into memory
        logger.info("Model validation successful.")
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        logger.error(
            "Cannot initialize text embeddings - model unavailable. Will retry later."
        )
        raise  # Raise to trigger retry mechanism

    # Validate database connectivity
    try:
        db = Database()
        if not db.client.ping():
            raise ConnectionError("Elasticsearch ping failed.")
    except Exception as e:
        logger.error(f"Error during database connection: {e}")
        raise  # Raise to trigger retry mechanism

    queue_checker = QueueChecker()

    # Check if queue already has pending tasks
    if not queue_checker.is_queue_empty(TEXT_EMBEDDINGS_QUEUE):
        pending_count = queue_checker.get_queue_length(TEXT_EMBEDDINGS_QUEUE)
        logger.info(
            f"Skipping init_embeddings - queue already has {pending_count} pending tasks"
        )
        return
    logger.info("Text embeddings queue is empty, starting initialization...")

    batch_size = settings.text_embedding_fetch_batch_size
    db = Database()
    total_message_count = 0

    messages_indices = db.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)
    if not messages_indices:
        logger.info("No messages indices found. Exiting init_embeddings.")
        return

    # loop through all messages indices
    for index in messages_indices:
        chat_id = get_chat_id_from_index_name(index)
        vectorized_index_name = build_index_name(
            IndexAlias.TEXT_VECTOR_INDEX_ALIAS, chat_id
        )
        if not db.index_exists(vectorized_index_name):
            try:
                db.create_embedding_index(chat_id)
                start_message_date = None
            except Exception as e:
                logger.error(f"Failed to create index for chat_id {chat_id}: {e}")
                break
        else:
            start_message_date = db.get_most_recent_doc_date(vectorized_index_name)

        # create batch tasks
        while True:
            try:
                messages = db.get_messages_to_embed(
                    chat_id=chat_id,
                    size=batch_size,
                    start_message_date=start_message_date,
                )
            except Exception as e:
                logger.error(f"Error retrieving messages for chat {chat_id}: {e}")
                break

            if not messages:
                logger.info(f"No more messages found to embed for chat {chat_id}.")
                break

            start_message_date = (
                datetime.fromisoformat(messages[-1].date)
                if isinstance(messages[-1].date, str)
                else messages[-1].date
            )
            batch = []
            for msg in messages:  # Convert to dict for serialization
                msg_dict = msg.dict()
                msg_dict["id"] = msg.id
                batch.append(msg_dict)
            total_message_count += len(batch)
            logger.info(f"Enqueuing {len(batch)} messages for chat {chat_id}")
            generate_embeddings.delay(
                chat_id=chat_id, message_list=batch
            )  # Enqueue the batch as a task
    logger.info(
        f"Enqueued {total_message_count} messages for embedding generation across all chats."
    )
