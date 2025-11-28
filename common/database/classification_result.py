from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ErrorType(str, Enum):
    TEXT_TOO_SHORT = "Text too short"
    EMPTY_TEXT = "Text empty after preprocessing"


class ClassificationResult(BaseModel):
    classified: bool  # True if successful classification
    score_pos: Optional[float] = None  # Positive score, None if not classified
    error: Optional[ErrorType] = None  # Error reason if classification failed
    processed_at: Optional[datetime] = None
