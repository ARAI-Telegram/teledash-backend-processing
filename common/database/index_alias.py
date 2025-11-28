from enum import Enum


class IndexAlias(Enum):
    IMAGE_VECTOR_INDEX_ALIAS = "vectorized_images"
    TEXT_VECTOR_INDEX_ALIAS = "vectorized_messages"
    MESSAGE_INDEX_ALIAS = "messages"
