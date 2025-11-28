import logging
from typing import Optional

from elasticsearch.dsl.query import Query as ESQuery

from common.database.semantic_search_result import SemanticSearchResult

logger = logging.getLogger(__name__)


class SearchService:
    """
    Handles database-related operations for searching similar images.
    """

    def __init__(self, database) -> None:
        self.database = database

    async def search_similar_texts(
        self, text_embedding: list[float], top_k: int, filter: Optional[ESQuery]
    ) -> list[SemanticSearchResult]:
        """
        Perform text-to-text search by finding the most similar text embeddings.

        Args:
            text_embeding: The embedding vector of the input text.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries with the most similar text messages and their similarity scores.
        """
        try:
            logger.debug("Performing text-to-text search...")
            results: list[
                SemanticSearchResult
            ] = await self.database.search_similar_docs(
                size=top_k,
                query_vector=text_embedding,
                index_alias=self.database.service_index_alias,
                filter=filter,
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {str(e)}")
