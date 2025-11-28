import logging
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from common.settings import settings

logger = logging.getLogger(__name__)

model_name = settings.text_embedding_model
model_path = Path(f"/app/tmp/model/{model_name}")


def init_embedding_model(load_to_ram: bool = True) -> Optional[SentenceTransformer]:
    """
    This function checks if the model is available locally. If not, it downloads the model
    from Hugging Face, if allowed by user.

    Args:
        load_to_ram: If True, returns the loaded model. If False, just ensures model
                    is downloaded and returns None.
    """
    # Check if the model exists locally
    if model_path.exists() and model_path.is_dir():
        logger.info(f"Loading model from local path: {model_path}")
        if not load_to_ram:
            logger.info("Model found locally, no need to load to RAM")
            return
        try:
            model = SentenceTransformer(str(model_path))
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model from local path: {e}")
    elif settings.load_models_from_huggingface:
        logger.info(
            f"Model folder not found. Downloading model from Huggingface: {model_name}"
        )
        try:
            # Download the model from Hugging Face
            model = SentenceTransformer(model_name)
            model.save(str(model_path))  # Save for future use
            logger.info(f"Model downloaded and saved to {model_path}")

            if load_to_ram:
                return model
            else:
                # Release from memory and return success
                del model
                logger.info("Model released from memory after saving to disk")
                return
        except Exception as e:
            raise RuntimeError(f"Error downloading model: {e}")
    else:
        logger.error(
            "The desired embedding model could not be found locally, and downloading is disabled."
        )
        raise RuntimeError("Model not found locally and downloading is disabled.")
