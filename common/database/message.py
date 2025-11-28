from datetime import datetime
from typing import Any, Optional, Tuple

from pydantic import BaseModel

from common.database.classification_result import ClassificationResult
from common.settings import settings
from common.utils import naive_utcnow


class Message(BaseModel):
    id: str
    chat_id: Optional[str] = None
    date: datetime

    def get_chat_id_from_id(self) -> Optional[str]:
        return self.chat_id

    @classmethod
    def get_defined_fields(cls) -> list[str]:
        """
        Dynamically retrieve and return the fields defined in the model.
        """
        return list(cls.__fields__.keys())

    def to_index_dict(self) -> dict[str, Any]:
        """
        Converts the object's attributes to a dictionary, excluding the "id" key.
        Returns:
            dict: The object's attributes without the "id" key.
        """
        return {key: value for key, value in self.dict().items() if key != "id"}

    class Config:
        extra = "allow"  # Allow extra fields for dynamic embeddings


class ASRMessage(Message):
    attachment: Optional[dict] = None
    text: Optional[str] = None


class ClassificationMessage(Message):
    language: str
    text: Optional[str] = None
    caption: Optional[str] = None
    attachment: Optional[dict] = None

    @classmethod
    def get_defined_fields(cls) -> list[str]:
        """
        Retrieve only the fields needed to perform the embedding task.
        """
        text_fields = settings.classification_fields
        basic_fields = ["id", "date", "language"]
        return text_fields + basic_fields

    def get_text_and_id(self) -> Optional[Tuple[str, str]]:
        """Extract text and ID from this message instance."""
        if not self.id:
            return None

        for field in settings.classification_fields:
            if hasattr(self, field):
                text_value = getattr(self, field)
                if text_value:
                    return text_value, self.id

        return None


class ClassificationResultMessage(BaseModel):
    message_id: str  # == id from source message
    classification_result: ClassificationResult


class TextSearchMessage(Message):
    language: str
    text: Optional[str] = None
    caption: Optional[str] = None
    attachment: Optional[dict] = None  # how only allow attachment.transcription?


class VectorizedTextMessage(Message):
    id: Optional[str] = None  # will later be set by ES
    message_id: str
    language: Optional[str] = None
    processed_at: datetime
    vector: list[float]  # The text vector embedding as a list of floats
    source_field: str

    def to_index_dict(self) -> dict[str, Any]:
        """
        Converts the object's attributes to a dictionary, excluding unwanted fields.
        """
        exclude_fields = {"id", "chat_id"}
        return {
            key: value
            for key, value in self.dict().items()
            if key not in exclude_fields
        }

    @classmethod
    def create_from_message(
        cls,
        message,
        embedding: tuple[str, list[float]],  # (source_field, embedding_vector)
    ) -> "VectorizedTextMessage":
        new_vector_message = cls(  # do we want to store the original text? Then must inherit from TextSearchMessage
            message_id=message.id,
            language=message.language,
            date=message.date,
            processed_at=naive_utcnow(),
            source_field=embedding[0],  # The field used for embedding
            vector=embedding[1],
        )

        return new_vector_message


class ImageSearchMessage(Message):
    caption: Optional[str] = None
    attachment: Optional[dict] = None


class VectorizedImageMessage(ImageSearchMessage):
    id: Optional[str] = None  # will later be set by ES
    processed_at: datetime
    message_id: str
    attachment_type: Optional[str]  # The type of the attachment (photo or video)
    vector: list[float]  # The image vector embedding as a list of floats

    def to_index_dict(self) -> dict[str, Any]:
        """
        Converts the object's attributes to a dictionary, excluding unwanted fields.
        """
        exclude_fields = {"id", "caption", "attachment"}
        return {
            key: value
            for key, value in self.dict().items()
            if key not in exclude_fields
        }

    @classmethod
    def create_from_message(
        cls,
        message: ImageSearchMessage,
        embedding: list[float],  # Single embedding for image (list of floats)
    ) -> "VectorizedImageMessage":
        # Extract attachment_type from the message (assuming 'attachment' is a dictionary)
        attachment_type = None
        if message.attachment and isinstance(message.attachment, dict):
            attachment_type = message.attachment.get("type", None)

        # Default to "photo" if attachment_type is not defined
        if not attachment_type:
            attachment_type = "photo"

        new_vector_message = cls(
            message_id=message.id,
            chat_id=message.chat_id,
            date=message.date,
            processed_at=naive_utcnow(),
            attachment_type=attachment_type,
            vector=embedding,
        )
        return new_vector_message
