from pydantic import BaseModel


class SemanticSearchResult(BaseModel):
    id: str
    score: float
