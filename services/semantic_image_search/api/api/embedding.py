import logging

import torch
from transformers import CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPModel

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    CLIP Model Class for API-specific operations like text embedding generation
    and text-to-image search functionality.
    """

    def __init__(self, model: CLIPModel, processor: CLIPProcessor) -> None:
        self.model = model
        self.processor = processor

    def embed_text(self, text: str) -> list[float]:
        try:
            logger.debug(f"Generating text embedding for: '{text}'")

            # Preprocess text using CLIPProcessor
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # CLIP's maximum token length
            )

            inputs = {k: v for k, v in inputs.items()}

            # Generate text embedding using CLIP model
            with torch.no_grad():  # No gradients needed for inference
                text_features = self.model.get_text_features(**inputs)
                # Normalize embedding for cosine similarity comparison
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to list for search
            embedding = text_features[0].numpy().tolist()

            logger.debug(f"Generated text embedding with dimension: {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate text embedding for '{text}': {e}")
            raise RuntimeError(f"Text embedding generation failed: {str(e)}")
