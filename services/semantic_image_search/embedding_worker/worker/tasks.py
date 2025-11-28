from datetime import datetime
from pathlib import Path
from typing import List, Optional

from celery import Task
from celery.utils.log import get_task_logger
from elasticsearch.dsl import AttrList
from main import app
from PIL import Image
from shared.model_init import init_image_embedding_model

from common.database.index_alias import IndexAlias
from common.database.message import ImageSearchMessage as Message
from common.database.message import VectorizedImageMessage
from common.redis_queue_manager import QueueChecker
from common.settings import settings
from common.storage import Storage, StorageBucketNames
from common.utils import build_index_name, get_chat_id_from_index_name
from worker.database import Database

IMAGE_EMBEDDING_QUEUE = "image-embeddings"  # TODO: store queue names in one place?

TMP_PATH = Path().cwd().joinpath("tmp")

logger = get_task_logger(__name__)


def get_attachment_storage_ref(doc, bucket: StorageBucketNames) -> Optional[str]:
    """
    Retrieve the storage reference object path for a given bucket.
    """
    if hasattr(doc, "attachment") and doc.attachment:
        attachment = doc.attachment
        if "storage_refs" in attachment:
            storage_refs_value = attachment["storage_refs"]

            if isinstance(storage_refs_value, AttrList) or isinstance(
                storage_refs_value, list
            ):
                for item in storage_refs_value:
                    # Compare against enum value, not string
                    if item["bucket"] == bucket.value:
                        return item["object"]
    return None


class ImageEmbeddingTask(Task):
    """
    Celery Task base class that loads the CLIP model and processor
    once per worker process and shares them across tasks.
    """

    abstract = True

    # Class-level shared variables
    _model = None
    _processor = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if ImageEmbeddingTask._model is None:
            logger.info("Loading embedding model for this worker process...")
            try:
                ImageEmbeddingTask._model, ImageEmbeddingTask._processor = (
                    init_image_embedding_model()
                )
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                logger.error("Critical error: Cannot proceed without model.")
                raise  # Let Celery handle the error naturally
        return self.run(*args, **kwargs)

    @property
    def model(self):
        """Return the shared model instance."""
        return ImageEmbeddingTask._model

    @property
    def processor(self):
        """Return the shared processor instance."""
        return ImageEmbeddingTask._processor


@app.task(
    name="image_embeddings.generate_embedding",
    queue=IMAGE_EMBEDDING_QUEUE,
    bind=True,
    base=ImageEmbeddingTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
    default_retry_delay=60 * 60,  # Retry after 60 minutes
)
def generate_image_embeddings(self, chat_id: str, message_list: List[dict]) -> None:
    """
    Generate image embeddings for a list of messages.

    Args:
        chat_id (str): The chat ID for which images are being embedded.
        message_list (List[dict]): A list of dictionaries representing serialized Message objects.

    Returns:
        None
    """
    storage_client = Storage()
    db = Database()

    try:
        messages: List[Message] = [Message(**msg) for msg in message_list]
        logger.info(f"Loaded {len(messages)} messages for embedding.")
    except Exception as e:
        logger.error(f"Error converting messages to Message objects: {e}")
        raise

    embedded_messages: dict[str, list[VectorizedImageMessage]] = {}
    images, texts, message_map = [], [], {}

    logger.info(
        f"Processing {len(messages)} messages from chat {chat_id} for image embedding."
    )

    for message in messages:
        if message.attachment and isinstance(message.attachment, dict):
            attachment_type = message.attachment.get("type", "photo")
        else:
            attachment_type = "photo"  # default

        if chat_id not in embedded_messages:
            embedded_messages[chat_id] = []

        file_name = get_attachment_storage_ref(message, StorageBucketNames.photos)

        if not file_name:
            logger.error(f"No image file path found for message id: {message.id}")
            continue

        local_media_path = TMP_PATH / Path(file_name).name
        try:
            storage_client.fget_object(
                StorageBucketNames.photos.value,
                file_name,
                str(local_media_path),
            )

        except Exception as e:
            logger.error(
                f"Failed to download {file_name} to {local_media_path}: {e}",
                exc_info=True,
            )
            return  # Stop processing if download fails

        try:
            image = Image.open(local_media_path)
        except Exception as e:
            logger.error(f"Failed to open image {local_media_path}: {e}", exc_info=True)
            continue

        text = message.caption or ""  # TODO: maybe replace with default caption

        images.append(image)
        texts.append(text)
        message_map[len(images) - 1] = {
            "message": message,
            "attachment_type": attachment_type,
            "chat_id": chat_id,
            "local_path": local_media_path,
        }

        # Process batch if batch size reached
        if len(images) == settings.image_embedding_model_batch_size:
            _process_image_batch(self, images, texts, message_map, embedded_messages)

            # Reset batch containers
            images, texts, message_map = [], [], {}

    # Process any leftover images
    if images:
        logger.info(f"Processing final batch of {len(images)} images for embedding.")
        _process_image_batch(self, images, texts, message_map, embedded_messages)

    # Ensure vector indices exist
    for chat_id, _ in embedded_messages.items():
        index_name = build_index_name(IndexAlias.IMAGE_VECTOR_INDEX_ALIAS, chat_id)
        if not db.index_exists(index_name):
            try:
                db.create_vectorized_image_index(index_name)
                logger.info(f"Created index for chat_id {chat_id}.")
            except Exception as e:
                logger.error(f"Failed to create index for chat_id {chat_id}: {e}")

    # Store all vectorized messages in DB
    for chat_id, vec_messages in embedded_messages.items():
        try:
            actions = db.create_index_actions_per_chat(chat_id, vec_messages)
            db.bulk_write(
                actions=actions,
            )
            logger.info(
                f"Stored {len(vec_messages)} vectorized messages for chat_id {chat_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to store vectorized messages for chat_id {chat_id}: {e}"
            )
            raise

    logger.info("Batch image embedding completed successfully.")


def _process_image_batch(
    self, images: list, texts: list, message_map: dict, embedded_messages: dict
) -> None:
    """
    Helper function to process a batch of images and store embeddings in embedded_messages.
    """
    logger.info(f"Processing batch of {len(images)} images and texts for embedding.")

    inputs = self.processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    outputs = self.model(**inputs)

    for idx, embedding in enumerate(outputs.image_embeds):
        msg_info = message_map[idx]
        vec_message = VectorizedImageMessage.create_from_message(
            msg_info["message"],
            embedding=embedding.cpu().detach().numpy().tolist(),
        )
        vec_message.attachment_type = msg_info["attachment_type"]
        vec_message.chat_id = msg_info["chat_id"]

        embedded_messages[msg_info["chat_id"]].append(vec_message)

    # Delete local files for this batch
    for info in message_map.values():
        local_path = info.get("local_path")
        if local_path and local_path.exists():
            try:
                local_path.unlink()
                logger.info(f"Deleted temporary file {local_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary file {local_path}: {e}")


@app.task(
    name="image_embeddings.init_image_embedding",
    autoretry_for=(Exception,),
    retry_kwargs={
        "max_retries": 5,
        "countdown": 300,
    },  # Retry every 5 minutes, max 5 times
)
def init_image_embedding() -> None:  # TODO: common method
    """
    Initialize the image embedding process for messages in the database,
    processing new images per chat in chronological order.

    The function:
    - Skips if there are already pending tasks in the image_embeddings queue.
    - Per chat, retrieves messages with photo attachments that have not yet been vectorized.
    - Processes messages in batches, always using the last message date as start_date for the next batch.
    - Enqueues each batch for embedding generation.
    """
    # Validate model availability before queuing any tasks
    logger.info("Validating image embedding model availability...")
    try:
        init_image_embedding_model(
            load_to_ram=False
        )  # Lightweight validation - doesn't load the model into memory
        logger.info("Model validation successful.")
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        logger.error(
            "Cannot initialize image embeddings - model unavailable. Will retry later."
        )
        raise  # Raise to trigger retry mechanism

    # Validate database connectivity
    try:
        db = Database()
        if not db.client.ping():
            raise ConnectionError("Elasticsearch ping failed.")
        logger.info("Connected to the database for image embeddings initialization.")
    except Exception as e:
        logger.error(f"Error during database connection: {e}")
        raise  # Raise to trigger retry mechanism

    # Validate storage connectivity
    try:
        _ = Storage()  # Test storage connection
        logger.info("Storage validation successful.")
    except Exception as e:
        logger.error(f"Storage validation failed: {e}")
        logger.error(
            "Cannot initialize image embeddings - storage unavailable. Will retry later."
        )
        raise  # Raise to trigger retry mechanism

    # db is already instantiated and validated above
    batch_size = settings.image_fetch_batch_size
    queue_checker = QueueChecker()

    # Early exit if queue already has pending tasks
    if not queue_checker.is_queue_empty(IMAGE_EMBEDDING_QUEUE):
        pending_count = queue_checker.get_queue_length(IMAGE_EMBEDDING_QUEUE)
        logger.info(
            f"Skipping init_image_embeddings - queue already has {pending_count} pending tasks"
        )
        return

    # Get message indices that actually have new data/photos
    indices_with_images = db.get_all_message_indices_with_new_images(
        attachment_type="photo"
    )
    if not indices_with_images:
        logger.info("No chats with new images found. Exiting init_image_embeddings.")
        return

    total_enqueued = 0

    # Process each chat index individually
    for index_name, last_vectorized_date in indices_with_images:
        chat_id = get_chat_id_from_index_name(index_name)  # not really necessary
        logger.info(f"Processing chat {chat_id} (messages index: {index_name})")

        # Ensure vectorized image index exists
        if not db.index_exists(index_name):
            try:
                db.create_vectorized_image_index(index_name)
                last_vectorized_date = None
                logger.info(f"Created vectorized image index for chat {chat_id}")
            except Exception as e:
                logger.error(
                    f"Failed to create vectorized index for chat {chat_id}: {e}"
                )
                continue

        start_date = last_vectorized_date

        # Batch-wise fetching and enqueuing
        while True:
            try:
                messages = db.get_images_to_vectorize_docs(
                    chat_id=chat_id,
                    size=batch_size,
                    start_date=start_date,
                )
            except Exception as e:
                logger.error(f"Error retrieving images for chat {chat_id}: {e}")
                break

            # Break if no messages returned
            if not messages:
                logger.info(f"No more new images found for chat {chat_id}.")
                break

            # Update start_date to last message's date to avoid duplicates in next batch
            last_message_date = messages[-1].date
            start_date = (
                datetime.fromisoformat(last_message_date)
                if isinstance(last_message_date, str)
                else last_message_date
            )
            logger.info(
                f"Retrieved {len(messages)} messages for chat {chat_id} starting from {start_date}."
            )

            # Prepare batch for task queue
            batch = []
            for msg in messages:
                msg_dict = msg.dict()
                msg_dict["id"] = msg.id
                batch.append(msg_dict)

            # Enqueue batch
            generate_image_embeddings.delay(chat_id=chat_id, message_list=batch)
            total_enqueued += len(batch)
            logger.info(f"Enqueued {len(batch)} images for chat {chat_id}")

    logger.info(f"Total {total_enqueued} images enqueued across all chats.")
