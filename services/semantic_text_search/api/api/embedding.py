import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model) -> None:
        self.model = model

    def embed_text(self, query: str) -> list[float]:
        """
        Generate an embedding for a given query.
        Args:
            query (str): The query for which to generate an embedding.
        Returns:
            list[float]: The embedding for the query as a list of floats.
        """
        try:
            logger.debug(f"Generating text embedding for: '{query}'")
            embedding = self.model.encode(query)

            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()

            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            else:
                logger.error(
                    f"Embedding is not a numpy array or torch tensor for '{query}'. Returning empty list."
                )
                return []

        except Exception as e:
            logger.error(f"Failed to generate text embedding for '{query}': {e}")
            raise RuntimeError(f"Text embedding generation failed: {str(e)}")
