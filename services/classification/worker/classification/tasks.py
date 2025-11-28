from typing import Tuple

import pandas as pd
from celery import Task
from celery.utils.log import get_task_logger

from common.database.index_alias import IndexAlias
from common.redis_queue_manager import QueueChecker
from common.settings import settings
from common.utils import get_chat_id_from_index_name
from main import app
from worker.classification.classification import (
    classify_batch,
    create_failed_classification_results,
)
from worker.classification.preprocessing import preprocess_df
from worker.database import Database
from worker.model_init import init_classification_model

logger = get_task_logger(__name__)


class ClassificationTask(Task):
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
        if ClassificationTask._model is None:
            logger.info("Loading classification model...")
            try:
                ClassificationTask._model = init_classification_model()
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load classification model: {e}")
                logger.error("Critical error: Cannot proceed without model.")
                raise
        return self.run(*args, **kwargs)

    @property
    def model(self):
        """Get the shared model instance."""
        return ClassificationTask._model


@app.task(
    name="classification.init_classification",
    autoretry_for=(Exception,),
    retry_kwargs={
        "max_retries": 5,
        "countdown": 300,
    },  # Retry every 5 minutes, max 5 times
)
def init_classification() -> None:
    """
    Initialize the classification process for documents in the database.
    This task identifies unclassified documents for each chat and enqueues batches for classification.
    Skips if there are already pending tasks in the classification queue.
    """
    # Validate model availability before queuing any tasks
    logger.info("Validating classification model availability...")
    try:
        init_classification_model(
            load_to_ram=False
        )  # Model validation - doesn't load the model into memory
        logger.info("Model validation successful.")
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        logger.error(
            "Cannot initialize classification - model unavailable. Will retry later."
        )
        raise

    # Validate database connectivity
    try:
        db = Database()
        if not db.client.ping():
            raise ConnectionError("Elasticsearch ping failed.")
        logger.info("Connected to the database for classification initialization.")
    except Exception as e:
        logger.error(f"Error during database connection: {e}")
        raise

    queue_checker = QueueChecker()

    # Check if queue already has pending tasks
    if not queue_checker.is_queue_empty("classification"):
        pending_count = queue_checker.get_queue_length("classification")
        logger.info(
            f"Skipping init_classification - queue already has {pending_count} pending tasks"
        )
        return

    logger.info("Classification queue is empty, starting initialization...")

    messages_indices = db.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)
    if not messages_indices:
        logger.info("No message indices found. Exiting classification initialization.")
        return

    try:
        unclassified_doc_count = db.get_unclassified_docs_count(
            chat_id=None, last_processed_date=None
        )
        logger.info(
            "Number of unclassified documents: %d. Classification batch size: %d",
            unclassified_doc_count,
            settings.classification_fetch_batch_size,
        )
    except Exception as e:
        logger.error(f"Error retrieving unclassified documents: {e}")
        raise

    # Track enqueued tasks and messages
    total_tasks_enqueued = 0
    total_messages_enqueued = 0

    # loop through all message indices
    for msg_index in messages_indices:
        chat_id = get_chat_id_from_index_name(msg_index)
        last_processed_date = None

        while True:
            unclassified_docs = db.get_unclassified_docs(
                size=settings.classification_fetch_batch_size,
                chat_id=chat_id,
                last_processed_date=last_processed_date,
            )

            if not unclassified_docs:
                logger.debug(
                    f"No more unclassified documents found for chat with id {chat_id}."
                )
                break

            last_processed_date = max(doc.date for doc in unclassified_docs if doc.date)
            texts_ids: list[Tuple[str, str]] = [
                result
                for doc in unclassified_docs
                if (result := doc.get_text_and_id()) is not None
            ]
            if texts_ids:
                batch = {
                    "chat_id": chat_id,
                    "documents": [
                        {"text": text, "id": doc_id} for text, doc_id in texts_ids
                    ],
                }
            else:
                logger.warning(
                    "No valid documents found with both 'text' and 'id' fields."
                )
                continue
            logger.info(
                f"Enqueuing {len(batch['documents'])} messages for classification."
            )

            classify_documents.delay(batch)
            total_tasks_enqueued += 1
            total_messages_enqueued += len(batch['documents'])

    logger.info(
        f"Initialization complete: Enqueued {total_tasks_enqueued} tasks with {total_messages_enqueued} messages for classification."
    )


@app.task(
    name="classification.classify_docs",
    bind=True,
    base=ClassificationTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},  # idempotent task, retry on failure
    default_retry_delay=60 * 60,  # Retry after 60 minutes
)
def classify_documents(self, document_batch: dict) -> None:
    """
    Classify a batch of documents from the provided dictionary.

    Args:
        document_batch (dict): A dictionary with "chat_id" and "documents" keys, where documents is a list of {"text": <text>, "id": <id>} dicts
    """

    chat_id = document_batch["chat_id"]
    documents = document_batch["documents"]

    for i, document in enumerate(documents):
        if not isinstance(document, dict):
            logger.error(f"Item at index {i} is not a dictionary: {document}")
            continue

        if "id" not in document or "text" not in document:
            logger.error(
                f"Document at index {i} is missing required keys 'id' or 'text': {document}"
            )
            continue

    try:
        database = Database()
        df = pd.DataFrame(documents)
        df["error"] = None  # --> df with columns id, text, error

        logger.info("Classifying %d documents...", len(df))

        # Preprocess the documents
        df_preprocessed = preprocess_df(df)
        df_not_classifiable = df_preprocessed[df_preprocessed.error.notna()]
        df_filtered = df_preprocessed[df_preprocessed.error.isna()]

        # Classify documents
        classified_messages = []
        if not df_filtered.empty:
            # Convert DataFrame to list of dicts for classify_batch
            documents = [
                {"id": str(row["id"]), "text": str(row["text"])}
                for _, row in df_filtered[["id", "text"]].iterrows()
            ]
            classified_messages = classify_batch(documents, self.model)

        # Handle documents that can not be classified
        unclassified_messages = []
        if not df_not_classifiable.empty:
            # Convert DataFrame to list of dicts
            unclassifiable_docs = [
                {"id": str(row["id"]), "error": row["error"]}
                for _, row in df_not_classifiable[["id", "error"]].iterrows()
            ]
            unclassified_messages = create_failed_classification_results(
                unclassifiable_docs
            )

        # Bulk update all results to database
        all_results = classified_messages + unclassified_messages
        if all_results:
            actions = database.create_update_actions_per_chat(chat_id, all_results)
            database.bulk_write(actions)
            logger.info(
                f"Updated {len(all_results)} documents of chat {chat_id} in database."
            )
        else:
            logger.info("No classification results to save")

        logger.info("Batch classification completed successfully.")
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        raise
