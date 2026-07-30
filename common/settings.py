from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseSettings, validator

# Whitelists for validation
allowed_text_embedding_fields = [
    "caption",
    "text",
    "attachment.transcription",
]

allowed_image_attachment_types = ["photo"]

allowed_classification_fields = ["text", "caption"]


class AsrModelType(str, Enum):
    # Reference: https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py#L12
    tiny = "tiny"
    tiny_en = "tiny.en"
    base = "base"
    base_en = "base.en"
    small = "small"
    small_en = "small.en"
    medium = "medium"
    medium_en = "medium.en"
    large = "large"
    large_v1 = "large-v1"
    large_v2 = "large-v2"
    large_v3 = "large-v3"
    distil_large_v2 = "distil-large-v2"
    distil_medium_en = "distil-medium.en"
    distil_small_en = "distil-small.en"
    distil_large_v3 = "distil-large-v3"
    turbo = "turbo"
    large_v3_turbo = "large-v3-turbo"


class ImageEmbeddingModelType(str, Enum):
    clip_vit_base_patch16 = "openai/clip-vit-base-patch16"
    clip_vit_base_patch32 = "openai/clip-vit-base-patch32"  # smaller but faster model
    clip_vit_large_patch14 = "openai/clip-vit-large-patch14"  # larger and more accurate
    clip_vit_large_patch14_336 = (
        "openai/clip-vit-large-patch14-336"  # even larger, higher resolution
    )


class Settings(BaseSettings):
    # === REQUIRED SETTINGS (no defaults) ===
    # Storage credentials
    storage_access_key: str
    storage_secret_key: str

    # === OPTIONAL SETTINGS (have defaults) ===
    # Project
    compose_project_name: str = "teledash-processing"
    teledash_docker_network: str = "teledash_default"

    # Elasticsearch
    elastic_host: str = "elastic"
    elastic_port: int = 9200

    # Redis
    redis_port: int = 6379

    # GPU Configuration
    gpu_use: bool = False
    gpu_device: int = 0

    # Model Loading
    load_models_from_huggingface: bool = True

    # Classification
    classification_model: str = "digitaler-hass/TelConGBERT"
    classification_fields: List[str] = ["text", "caption"]
    classification_min_char_length: int = 50
    classification_fetch_batch_size: int = 265
    classification_languages: List[str] = ["de"]  # TelConGBERT only supports German
    classification_interval_minutes: int = 120

    # ASR (Automatic Speech Recognition)
    asr_model: AsrModelType = AsrModelType.medium
    asr_vad_filter: bool = True
    asr_attachment_types: List[str] = ["audio", "voice"]
    asr_concurrency: int = 1
    asr_interval_minutes: int = 60

    # Storage Provider
    storage_provider: Literal["aws", "s3-compatible"] = "s3-compatible"
    storage_region: Optional[str] = None
    storage_use_ssl: bool = False
    storage_verify_ssl: bool = True  # verify TLS cert; set False for self-signed endpoints (e.g. NAS quobjects)
    storage_endpoint: str = "minio:9000"
    storage_bucket_prefix: str = ""

    # Semantic Text Search
    text_embedding_model: str = (
        "Sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    text_embedding_dimension: int = 384
    text_embedding_fields: List[str] = ["text", "caption"]
    text_embedding_fetch_batch_size: int = 256
    text_embedding_model_batch_size: int = 32
    text_embeddings_interval_minutes: int = 30

    # Semantic Image Search
    image_embedding_model: ImageEmbeddingModelType = (
        ImageEmbeddingModelType.clip_vit_base_patch16
    )
    image_embedding_dimension: int = 512  # Auto-set based on model, default for base
    image_attachment_types: List[str] = ["photo"]
    image_embedding_model_batch_size: int = 4
    image_fetch_batch_size: int = 128
    image_embedding_interval_minutes: int = 30

    # Semantic Search APIs
    text_fast_api_port: int = 8001
    image_fast_api_port: int = 8002

    @validator("text_embedding_fields")
    def match_text_embedding_fields(cls, v):
        if v:
            invalid_types = [t for t in v if t not in allowed_text_embedding_fields]
            if invalid_types:
                raise ValueError(
                    f'"TEXT_EMBEDDING_FIELDS" contains invalid field name(s): {", ".join(invalid_types)}'
                )
        return v

    @validator("classification_fields")
    def match_classification_fields(cls, v):
        if v:
            invalid_types = [t for t in v if t not in allowed_classification_fields]
            if invalid_types:
                raise ValueError(
                    f'"CLASSIFICATION_FIELDS" contains invalid field name(s): {", ".join(invalid_types)}'
                )
        return v

    @validator("image_attachment_types")
    def match_image_attachment_types(cls, v):
        if v:
            invalid_types = [t for t in v if t not in allowed_image_attachment_types]
            if invalid_types:
                raise ValueError(
                    f'"IMAGE_ATTACHMENT_TYPES" contains invalid attachment type(s): {", ".join(invalid_types)}'
                )
        return v

    @validator("image_embedding_dimension", pre=True, always=True)
    def set_image_embedding_dimension(cls, v, values):
        """Auto-set embedding dimension based on model if not explicitly provided."""
        model = values.get(
            "image_embedding_model", ImageEmbeddingModelType.clip_vit_base_patch16
        )
        if model == ImageEmbeddingModelType.clip_vit_base_patch32:
            return 512
        elif model == ImageEmbeddingModelType.clip_vit_base_patch16:
            return 512
        elif model in [
            ImageEmbeddingModelType.clip_vit_large_patch14,
            ImageEmbeddingModelType.clip_vit_large_patch14_336,
        ]:
            return 768
        return v

    @validator("storage_region", always=True)
    def validate_storage_configuration(cls, v, values):
        """Validate storage configuration based on provider."""
        storage_provider = values.get("storage_provider", "s3-compatible")

        # AWS S3 requires region
        if storage_provider == "aws" and not v:
            raise ValueError("storage_region is required when using AWS provider")

        return v


settings = Settings()  # type: ignore
