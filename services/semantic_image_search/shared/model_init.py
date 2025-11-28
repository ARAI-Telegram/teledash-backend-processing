import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from transformers import CLIPProcessor, PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPModel

from common.settings import settings

logger = logging.getLogger(__name__)

model_name = settings.image_embedding_model.value
model_path = Path(f"/app/tmp/model/{model_name}")


def init_image_embedding_model(
    load_to_ram: bool = True,
) -> Optional[Tuple[PreTrainedModel, CLIPProcessor]]:
    """
    Load the image embedding model from local path or download from Hugging Face if allowed.

    Args:
        load_to_ram: If True, returns the loaded (model, processor) tuple. If False, just ensures model
                    is downloaded and returns None.
    """
    if model_path.exists() and model_path.is_dir():
        logger.info(f"Loading image embedding model from local path: {model_path}")
        if not load_to_ram:
            logger.info("Model found locally, no need to load to RAM")
            return
        try:
            processor_result = CLIPProcessor.from_pretrained(
                os.fspath(model_path), tokenizer_fast=False
            )
            model = CLIPModel.from_pretrained(os.fspath(model_path))
            # Handle case where processor might return tuple (processor, dict) or just processor
            if isinstance(processor_result, tuple):
                processor = processor_result[0]  # Extract processor from tuple
            else:
                processor = processor_result

            if isinstance(processor, CLIPProcessor):
                logger.info(f"Model loaded successfully from {model_path}")
                return model, processor
            else:
                raise RuntimeError("Failed to load valid CLIPProcessor")
        except Exception as e:
            raise RuntimeError(
                f"Error loading image embedding model from local path: {e}"
            )
    elif settings.load_models_from_huggingface:
        logger.info(
            f"Model folder not found. Downloading {model_name} from Hugging Face..."
        )
        logger.info(f"Model with name {model_name} will be saved to {model_path}")
        try:
            processor_result = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            if isinstance(processor_result, tuple):
                processor = processor_result[0]
            else:
                processor = processor_result
            if isinstance(processor, CLIPProcessor):
                processor.save_pretrained(os.fspath(model_path))
                model.save_pretrained(os.fspath(model_path))
                logger.info(f"Model downloaded and saved to {model_path}")

                if load_to_ram:
                    return model, processor
                else:
                    # Release from memory and return success
                    del model, processor
                    logger.info("Model released from memory after saving to disk")
                    return
            else:
                raise RuntimeError("Failed to load valid CLIPProcessor")
        except Exception as e:
            raise RuntimeError(f"Error downloading image embedding model: {e}")
    else:
        raise RuntimeError("Model not found locally and downloading is disabled.")
