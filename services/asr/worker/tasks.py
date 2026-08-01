import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from celery import Task
from celery.utils.log import get_task_logger
from elasticsearch.dsl import AttrList
from faster_whisper import WhisperModel, available_models

from common.database.index_alias import IndexAlias
from common.redis_queue_manager import QueueChecker
from common.settings import settings
from common.storage import Storage
from common.utils import build_index_name, get_chat_id_from_index_name
from main import app
from worker.database import Database

ASR_QUEUE_NAME = "asr"
MODEL_CACHE_PATH = "/app/tmp/.cache/faster-whisper"
TMP_PATH = Path("/app/tmp")

logger = get_task_logger(__name__)


def get_attachment_duration_seconds(doc) -> Optional[float]:
    """
    Extract the media duration reported by Telegram from a document.

    The duration is already part of the attachment metadata fetched from
    Elasticsearch, so it can be checked before downloading the media file.

    Args:
        doc: The document (Message object) to process.

    Returns:
        Duration in seconds, or None if the document carries no duration metadata.
    """
    attachment = getattr(doc, "attachment", None)
    if not attachment:
        return None

    # 'attachment' is a plain dict, but the nested 'raw' is an AttrList/AttrDict
    # without a .get() method, so membership has to be checked explicitly.
    raw = attachment.get("raw")
    if raw is None or "duration" not in raw:
        return None

    duration = raw["duration"]
    if not isinstance(duration, (int, float)):
        logger.warning(
            "Unexpected duration value %r for doc id %s; ignoring.", duration, doc.id
        )
        return None

    return float(duration)


def get_attachment_storage_refs_and_id(doc) -> Optional[Tuple[list, str]]:
    """
    Extract storage references and ID from a document if it has a relevant attachment.

    Args:
        doc: The document (Message object) to process.

    Returns:
        Tuple containing the storage_refs list and document ID if a relevant
        attachment is found, otherwise None.
    """
    if not hasattr(doc, "id"):
        logger.warning("Skipping document due to missing 'id': %s", doc)
        return None

    if hasattr(doc, "attachment") and doc.attachment:
        attachment = doc.attachment
        if "storage_refs" in attachment:
            storage_refs_value = attachment["storage_refs"]

            # Ensure storage_refs is a list (or convert if necessary, though it should be)
            if isinstance(storage_refs_value, AttrList):
                # Convert AttrList to list to match the type hint
                return list(storage_refs_value), doc.id
            else:
                logger.warning(
                    "Attachment storage_refs is not a list for doc id %s: %s",
                    doc.id,
                    storage_refs_value,
                )
        else:
            logger.warning(
                "Attachment found but missing 'storage_refs' for doc id %s", doc.id
            )
    return None


class TranscribeTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    # Reference: https://towardsdatascience.com/deploying-ml-models-in-production-with-fastapi-and-celery-7063e539a5db # noqa
    """

    abstract = True

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            try:
                self.model, _ = load_model(settings.asr_model)
            except Exception as e:
                logger.error(f"Failed to load ASR model: {e}")
                raise
        return self.run(*args, **kwargs)


def load_model(
    model_size: str,
    load_to_ram: bool = True,
) -> Tuple[Optional[WhisperModel], float]:
    """
    Load model via faster-whisper.

    Loads whisper model from cache or downloads it from Hugging Face
    if not available locally. Available models: https://huggingface.co/Systran

    Args:
        model_size: The size/name of the model to load
        load_to_ram: If True, returns the loaded (model, duration) tuple. If False, just ensures model
                    is cached and returns (None, duration).

    Returns:
        When load_to_ram=True: (WhisperModel, duration)
        When load_to_ram=False: (None, duration) - validates/downloads model only
    """
    if model_size not in available_models():
        logger.error(
            f"Model size '{model_size}' is not available",
            exc_info=True,
        )
        raise RuntimeError(f"Model size '{model_size}' is not available")

    if not load_to_ram:
        cache_base_path = Path(MODEL_CACHE_PATH)

        # Look for any folder that contains the model size in its name
        # This handles different HuggingFace repository naming patterns
        model_cache_path = None
        if cache_base_path.exists():
            for folder in cache_base_path.iterdir():
                if folder.is_dir() and model_size in folder.name:
                    model_cache_path = folder
                    break

        # Check if model directory exists (consistent with other services)
        if model_cache_path and model_cache_path.exists():
            logger.info(
                f"Model validation: Found ASR model locally at {model_cache_path}"
            )
            return None, 0.0
        else:
            logger.info(
                f"Model validation: ASR model not cached locally, downloading {model_size}..."
            )

    # Determine device based on GPU settings
    device = "cuda" if settings.gpu_use else "cpu"
    device_index = settings.gpu_device if settings.gpu_use else 0

    options = {
        "device": device,
        "device_index": device_index,
        "model_size_or_path": model_size,
        "download_root": MODEL_CACHE_PATH,
        # "cpu_threads": settings.asr_cpu_threads, # ToDo
        # "num_workers": settings.asr_num_workers, # ToDo
    }

    logger.info(
        f"Loading model '{model_size}' [faster-whisper] on {device.upper()} "
        f"(device_index={device_index}, gpu_use={settings.gpu_use})"
    )

    try:
        start_time = time.time()
        model = WhisperModel(**options)
        end_time = round(time.time() - start_time, 2)

        logger.info(
            f"Model '{model_size}' loaded successfully in {end_time}s on {device.upper()}"
        )

        if load_to_ram:
            return model, end_time
        else:
            # For validation: release from memory and return success
            del model
            logger.info("ASR model released from memory after caching to disk")
            return None, end_time

    except Exception as e:
        # Provide helpful error message if GPU initialization fails
        if settings.gpu_use and "CUDA" in str(e):
            error_msg = (
                f"Failed to initialize model on GPU: {e}\n\n"
                "Possible causes:\n"
                "1. NVIDIA Container Toolkit is not installed on the host\n"
                "2. NVIDIA_VISIBLE_DEVICES not set correctly in .env\n"
                "3. No compatible GPU available\n"
                "4. CUDA libraries (cuBLAS, cuDNN) not accessible\n\n"
                "Install NVIDIA Container Toolkit: "
                "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html\n\n"
                "Or set GPU_USE=false in .env to use CPU mode."
            )
            logger.error(error_msg)
        else:
            logger.error(
                f"Loading model '{model_size}' on {device.upper()} failed",
                exc_info=True,
            )
        raise RuntimeError(f"Failed to load ASR model '{model_size}': {e}")


# ToDo: seperate this task into different queue to avoid blocking other tasks?


@app.task(
    name="asr.init_asr",
    autoretry_for=(Exception,),
    retry_kwargs={
        "max_retries": 5,
        "countdown": 300,
    },  # Retry every 5 minutes, max 5 times
)
def init_asr() -> None:
    """
    Initialize the automatic speech recognition (asr) process for documents
    in the database.
    """
    # Validate model availability before queuing any tasks
    logger.info("Validating ASR model availability...")
    try:
        _, duration = load_model(settings.asr_model, load_to_ram=False)
        logger.info(f"Model validation successful in {duration}s.")
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        logger.error("Cannot initialize ASR - model unavailable. Will retry later.")
        raise  # Raise to trigger retry mechanism

    # Validate database connectivity
    try:
        database = Database()
        if not database.client.ping():
            raise ConnectionError("Elasticsearch ping failed.")
        logger.info("Connected to the database for asr initialization.")
    except Exception as e:
        logger.error(f"Error during database connection: {e}", exc_info=True)
        raise

    # Validate storage connectivity
    try:
        _ = Storage()  # Test storage connection
        logger.info("Storage validation successful.")
    except Exception as e:
        logger.error(f"Storage validation failed: {e}")
        logger.error("Cannot initialize ASR - storage unavailable. Will retry later.")
        raise  # Raise to trigger retry mechanism

    queue_checker = QueueChecker()

    # Safety check: if the ASR queue is empty but messages are still marked as PENDING,
    # reset their status. This prevents documents from getting stuck in PENDING forever
    # (e.g., after a crash when tasks were lost).
    if queue_checker.is_queue_empty(ASR_QUEUE_NAME):
        database.clear_pending_states()
        logger.warning(
            """ASR queue was empty, but found documents with pending status.
            Reset pending states to allow reprocessing."""
        )

    messages_indices = database.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)
    if not messages_indices:
        logger.info("No message indices found. Exiting asr initialization.")
        return

    last_processed_date: Optional[datetime] = None

    try:
        unprocessed_audio_count = database.get_unprocessed_audio_count()
        logger.info(
            "Number of unprocessed media attachments: %d (types considered: %s)",  # Use %s for list
            unprocessed_audio_count,
            settings.asr_attachment_types,
        )
    except Exception as e:
        logger.error(f"Error retrieving unprocessed media attachments: {e}")
        raise

    # loop through all message indices
    for msg_index in messages_indices:
        chat_id = get_chat_id_from_index_name(msg_index)
        last_processed_date: Optional[datetime] = None

        while True:
            unclassified_docs = database.get_unprocessed_audio_docs(
                size=100,
                chat_id=chat_id,
                last_processed_date=last_processed_date,
            )

            if not unclassified_docs:
                logger.info(
                    f"No (more) unclassified documents found for chat {chat_id}"
                )
                break

            logger.info(
                f"Processing chat {chat_id} with {len(unclassified_docs)} unclassified documents..."
            )

            # Update last_processed_date using the date of the last document fetched
            # Ensure dates exist before finding max
            valid_dates = [
                doc.date
                for doc in unclassified_docs
                if hasattr(doc, "date") and doc.date
            ]

            if not valid_dates:
                logger.warning(
                    f"No valid dates found in the current batch of documents from chat {chat_id}."
                )
                break

            last_processed_date = max(valid_dates)

            # Skip media that is too long to transcribe. Very long files exhaust
            # the worker's memory and kill the container, so they are filtered out
            # here - before the file is downloaded - and marked as processed so
            # they are not picked up again on the next run.
            # Note: this runs after 'last_processed_date' is derived from the full
            # batch, so pagination keeps advancing even if every doc is skipped.
            max_duration = settings.asr_max_duration_seconds
            if max_duration > 0:
                within_duration_limit = []

                for doc in unclassified_docs:
                    duration = get_attachment_duration_seconds(doc)

                    if duration is None:
                        logger.debug(
                            "No duration metadata for doc %s, processing it.", doc.id
                        )
                        within_duration_limit.append(doc)
                        continue

                    if duration <= max_duration:
                        within_duration_limit.append(doc)
                        continue

                    logger.warning(
                        "Skipping doc %s: duration %.0fs exceeds ASR_MAX_DURATION_SECONDS (%ds).",
                        doc.id,
                        duration,
                        max_duration,
                    )

                    try:
                        database.update_one_retry(
                            index_name=build_index_name(
                                IndexAlias.MESSAGE_INDEX_ALIAS, chat_id
                            ),
                            doc_id=doc.id,
                            update_doc_body={
                                "attachment": {
                                    "transcription_status": "SUCCESS",
                                    "transcription": None,
                                    "transcription_skip_reason": (
                                        f"duration {duration:.0f}s exceeds limit "
                                        f"{max_duration}s"
                                    ),
                                }
                            },
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to mark doc_id {doc.id} as skipped: {e}",
                            exc_info=True,
                        )

                unclassified_docs = within_duration_limit

            # Extract storage_refs and ids
            storage_refs_and_ids: list[Tuple[list, str]] = [
                result
                for doc in unclassified_docs
                if (result := get_attachment_storage_refs_and_id(doc)) is not None
            ]

            if not storage_refs_and_ids:
                logger.info(
                    "No documents with relevant attachments and storage_refs found in this batch."
                )
                continue

            # Trigger tasks for each attachment found
            logger.info(
                f"Triggering {len(storage_refs_and_ids)} transcription tasks for chat {chat_id}."
            )
            for storage_refs, doc_id in storage_refs_and_ids:
                # Find the actual media file reference (e.g., the .mp4 or .ogg)
                # This assumes the relevant media file doesn't have 'thumbnail' in its bucket name
                media_ref = next(
                    (
                        ref
                        for ref in storage_refs
                        if hasattr(ref, "bucket")
                        and "thumbnail" not in ref.bucket.lower()
                    ),
                    None,
                )

                if media_ref and hasattr(media_ref, "object"):
                    media_identifier = f"{media_ref.bucket}/{media_ref.object}"

                    try:
                        database.update_one_retry(
                            index_name=build_index_name(
                                IndexAlias.MESSAGE_INDEX_ALIAS, chat_id
                            ),
                            doc_id=doc_id,
                            update_doc_body={
                                "attachment": {
                                    "transcription_status": "PENDING",
                                }
                            },
                        )

                        # Trigger the transcription task
                        logger.info(
                            f"Triggering transcription task for media '{media_identifier}' (doc_id: {doc_id})"
                        )
                        transcribe.delay(
                            media=media_identifier, chat_id=chat_id, doc_id=doc_id
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to update database and trigger transcription for doc_id {doc_id}: {e}",
                            exc_info=True,
                        )
                        raise

                else:
                    logger.warning(
                        f"Could not determine media file from storage_refs for doc_id {doc_id}: {storage_refs}"
                    )

    logger.info("Finished processing loop for documents with relevant attachments.")


@app.task(name="asr.transcribe", base=TranscribeTask, bind=True)
def transcribe(
    self,
    media: str,  # This will now be something like "bucket/object.mp4"
    chat_id: str,
    doc_id: str,
) -> None:
    """
    Transcribe media file identified by 'media' (e.g., "bucket/object.mp4")
    and save the result associated with 'doc_id' to the database.

    'self' is the task instance (TranscribeTask) due to bind=True.
    """
    logger.info(f"Starting transcription task for media '{media}' (doc_id: {doc_id})")

    storage_client = Storage()
    database = Database()
    local_media_path = None  # Initialize to None
    index_name = build_index_name(IndexAlias.MESSAGE_INDEX_ALIAS, chat_id)

    try:
        bucket, object_name = media.split("/", 1)
        local_media_path = TMP_PATH / Path(object_name).name
        logger.info(
            f"Attempting to download {bucket}/{object_name} to {local_media_path}"
        )
        storage_client.fget_object(bucket, object_name, str(local_media_path))
        logger.info(f"Downloaded {media} to {local_media_path}")
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Failed to download {media}: {e}", exc_info=True)
        raise

    try:
        logger.info(f"Transcribing file '{local_media_path}'")
        segments, info = self.model.transcribe(
            str(local_media_path),
            beam_size=5,
            language=None,  # auto-detect language
            vad_filter=settings.asr_vad_filter,  # Use VAD filter if specified
        )

        detected_language = info.language
        language_probability = info.language_probability
        transcription_text = "".join(segment.text for segment in segments)

        if not transcription_text:
            transcription_text = None
            logger.info(
                f"No transcription text found for {media}. Language: {detected_language} (Prob: {language_probability:.2f})"
            )
        else:
            logger.info(
                f"Transcription complete for {media}. Language: {detected_language} (Prob: {language_probability:.2f}): {transcription_text[:10]}..."
            )

    except IndexError as e:
        # Handle files with no audio stream or invalid audio stream
        logger.warning(f"File {media} has no audio stream or invalid audio format: {e}")

        # Mark as SUCCESS but with no transcription (video might have no audio)
        database.update_one_retry(
            index_name=index_name,
            doc_id=doc_id,
            update_doc_body={
                "attachment": {
                    "transcription_status": "SUCCESS",
                    "transcription": None,
                }
            },
        )
        logger.info(
            f"Marked {media} as SUCCESS with no transcription (no audio stream)"
        )

        # Remove downloaded file
        if local_media_path and local_media_path.exists():
            local_media_path.unlink()
        return  # Exit successfully without raising

    except Exception as e:
        database.update_one_retry(
            index_name=index_name,
            doc_id=doc_id,
            update_doc_body={
                "attachment": {
                    "transcription_status": "FAILURE",
                }
            },
        )
        logger.error(f"Transcription of {media} failed: {e}", exc_info=True)
        # Remove downloaded file even if transcription fails
        if local_media_path and local_media_path.exists():
            local_media_path.unlink()
        raise

    # Update the database document with the transcription result
    try:
        update_body = {"attachment": {"transcription_status": "SUCCESS"}}
        if transcription_text:
            update_body["attachment"].update(
                {
                    "transcription": transcription_text.strip(),
                    "transcription_language": detected_language,
                    "transcription_language_probability": language_probability,
                }
            )

        database.update_one_retry(
            index_name=index_name,
            doc_id=doc_id,
            update_doc_body=update_body,
        )
        logger.info(f"Successfully updated database for doc_id {doc_id}")
    except Exception as e:
        logger.error(
            f"Failed to update database for doc_id {doc_id}: {e}", exc_info=True
        )
        raise

    finally:
        if local_media_path is not None and local_media_path.exists():
            try:
                local_media_path.unlink()
                logger.info(f"Deleted temporary file {local_media_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary file {local_media_path}: {e}")
