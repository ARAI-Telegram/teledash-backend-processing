"""
Topic Model Initialization

This module handles loading and initializing BERTopic models with GPU support.
"""

import os
from pathlib import Path
from typing import Optional

from bertopic import BERTopic
from celery.utils.log import get_task_logger
from sentence_transformers import SentenceTransformer

from common.settings import settings

logger = get_task_logger(__name__)


def get_topic_model_path() -> Path:
    """Get the path where topic models should be stored."""
    model_dir = Path("/tmp/topic_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def init_topic_model(load_to_ram: bool = True) -> Optional[BERTopic]:
    """
    Initialize BERTopic model with GPU support if available.

    Args:
        load_to_ram: If True, loads model into memory. If False, just validates model exists.

    Returns:
        BERTopic model instance if load_to_ram=True, None otherwise
    """
    model_path = get_topic_model_path() / "bertopic_model"

    # Check if we should use GPU
    use_gpu = settings.gpu_use and os.path.exists('/dev/nvidia0')

    if use_gpu:
        logger.info("GPU detected - using GPU-accelerated topic modeling")
        device = f"cuda:{settings.gpu_device}"
    else:
        logger.info("No GPU detected - using CPU for topic modeling")
        device = "cpu"

    if not load_to_ram:
        logger.info("Model validation mode - not loading into RAM")
        return None

    try:
        # Check if a saved model exists
        if model_path.exists():
            logger.info(f"Loading existing topic model from {model_path}")
            topic_model = BERTopic.load(str(model_path))
            logger.info("Topic model loaded successfully from disk")
            return topic_model
        else:
            logger.info("No existing model found - will create new model on first fit")
            return None

    except Exception as e:
        logger.error(f"Error initializing topic model: {e}")
        raise


def create_topic_model(
    use_gpu: bool = True,
    min_topic_size: Optional[int] = None,
    max_topics: Optional[int] = None
) -> BERTopic:
    """
    Create a new BERTopic model instance with configured parameters.

    Args:
        use_gpu: Whether to use GPU acceleration
        min_topic_size: Minimum number of messages to form a topic (overrides settings)
        max_topics: Maximum number of topics to discover (overrides settings)

    Returns:
        Configured BERTopic model instance
    """
    device = f"cuda:{settings.gpu_device}" if use_gpu else "cpu"

    # Use provided values or fall back to settings
    min_size = min_topic_size if min_topic_size is not None else settings.topic_min_size
    nr_topics = max_topics if max_topics is not None else (
        settings.topic_n_topics if settings.topic_n_topics != "auto" else None
    )

    logger.info(f"Creating new topic model with device: {device}, "
                f"min_topic_size: {min_size}, nr_topics: {nr_topics}")

    # Initialize embedding model
    embedding_model = SentenceTransformer(
        settings.topic_embedding_model,
        device=device
    )

    # Configure topic model parameters
    topic_model_kwargs = {
        "embedding_model": embedding_model,
        "language": "multilingual",
        "min_topic_size": min_size,
        "nr_topics": nr_topics,
        "calculate_probabilities": True,
        "verbose": True,
    }

    # Use GPU-accelerated UMAP and HDBSCAN if available
    if use_gpu:
        try:
            from cuml.cluster import HDBSCAN
            from cuml.manifold import UMAP

            logger.info("Using GPU-accelerated UMAP and HDBSCAN")

            umap_model = UMAP(
                n_components=5,
                n_neighbors=15,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=min_size,
                min_samples=10,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )

            topic_model_kwargs["umap_model"] = umap_model
            topic_model_kwargs["hdbscan_model"] = hdbscan_model

        except ImportError:
            logger.warning("cuML not available - falling back to CPU-based dimensionality reduction")

    # Create model
    topic_model = BERTopic(**topic_model_kwargs)

    logger.info("Topic model created successfully")
    return topic_model


def save_topic_model(topic_model: BERTopic) -> None:
    """
    Save topic model to disk.

    Args:
        topic_model: BERTopic model to save
    """
    model_path = get_topic_model_path() / "bertopic_model"

    try:
        logger.info(f"Saving topic model to {model_path}")
        topic_model.save(
            str(model_path),
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model=True
        )
        logger.info("Topic model saved successfully")
    except Exception as e:
        logger.error(f"Error saving topic model: {e}")
        raise
