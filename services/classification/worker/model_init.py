from pathlib import Path
from typing import Optional

from celery.utils.log import get_task_logger
from requests import HTTPError, Timeout
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TextClassificationPipeline as pipeline

from common.settings import settings

model_name = settings.classification_model
model_path = Path(f"/app/tmp/model/{model_name}")

logger = get_task_logger(__name__)


def init_classification_model(load_to_ram: bool = True) -> Optional[pipeline]:
    """
    This function checks if the model is available locally.
    If not, it downloads the model from Hugging Face.

    Args:
        load_to_ram: If True, returns the loaded pipeline. If False, just ensures model
                    is downloaded and returns None.
    """
    if model_path.exists() and model_path.is_dir():
        logger.info(f"Loading model from local path: {model_path}")
        if not load_to_ram:
            logger.info("Model found locally, no need to load to RAM")
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, clean_up_tokenization_spaces=True, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except OSError as e:
            raise RuntimeError(f"Error loading model from local path: {e}")
    elif settings.load_models_from_huggingface:
        logger.info(
            f"Model folder not found. Downloading model from Huggingface: {model_name}"
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, clean_up_tokenization_spaces=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Save model and tokenizer for future use
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info(f"Model downloaded and saved to {model_path}")

            if not load_to_ram:
                # Release from memory and return success
                del model, tokenizer
                logger.info("Model released from memory after saving to disk")
                return
        except (OSError, HTTPError, ConnectionError, Timeout) as e:
            raise RuntimeError(f"Error downloading model: {e}")
    else:
        raise RuntimeError("Model not found locally and downloading is disabled.")

    # Only executed when load_to_ram=True
    device = settings.gpu_device if settings.gpu_use else -1
    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,
        device=device,
    )
