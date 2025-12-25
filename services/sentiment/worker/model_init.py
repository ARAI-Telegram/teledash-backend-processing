"""
Initialize sentiment analysis model.
"""

import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


def init_sentiment_model(load_to_ram: bool = True):
    """
    Initialize the sentiment analysis model.

    Uses oliverguhr/german-sentiment-bert for German sentiment analysis.
    Falls back to multilingual models if needed.

    Args:
        load_to_ram: If True, loads model into memory. If False, only validates availability.

    Returns:
        Hugging Face pipeline for sentiment analysis or None if load_to_ram=False
    """
    model_name = "oliverguhr/german-sentiment-bert"

    try:
        if not load_to_ram:
            # Just validate model availability without loading
            logger.info(f"Validating sentiment model availability: {model_name}")
            AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Sentiment model {model_name} is available")
            return None

        # Load model with GPU support if available
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"

        logger.info(f"Loading sentiment model {model_name} on {device_name}...")

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=512
        )

        logger.info(f"Sentiment model loaded successfully on {device_name}")
        return sentiment_pipeline

    except Exception as e:
        logger.error(f"Error initializing sentiment model: {e}")
        raise
