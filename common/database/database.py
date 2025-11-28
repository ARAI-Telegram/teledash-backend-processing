from datetime import datetime
from typing import Optional

from celery.utils.log import get_task_logger
from elasticsearch import AsyncElasticsearch, ConnectionTimeout, Elasticsearch
from elasticsearch.dsl import AsyncSearch, Search
from elasticsearch.dsl.query import Query as ESQuery
from elasticsearch.helpers import bulk

from common.database.index_alias import IndexAlias
from common.database.semantic_search_result import SemanticSearchResult
from common.settings import settings

logger = get_task_logger(__name__)


class DatabaseSync:
    """
    Base class for synchronous Elasticsearch operations.
    Used by workers (e.g., Celery).
    """

    def __init__(self, connect: bool = True) -> None:
        self.es_client = None

        if connect:
            self.connect()

    def connect(self) -> None:
        """
        Establish a connection to the Elasticsearch cluster.
        """
        try:
            self.es_client = Elasticsearch(
                hosts=[f"http://{settings.elastic_host}:{settings.elastic_port}"],
                timeout=30,
                max_retries=10,
                retry_on_timeout=True,
            )
            # ping?
        except ConnectionError as e:
            print(f"Error connecting to Elasticsearch: {e}")
            raise
        except ConnectionTimeout as e:
            print(f"Elasticsearch connection timeout: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while connecting to Elasticsearch: {e}")
            raise

    @property
    def client(self) -> Elasticsearch:
        """
        Return the Elasticsearch client, ensuring it is initialized.
        """
        if not self.es_client:
            raise RuntimeError(
                "Elasticsearch client is not initialized. Call `connect()` first."
            )
        return self.es_client

    def get_all_indices_by_alias(self, alias: IndexAlias) -> list[str]:
        """
        Find all Elasticsearch indices that start with the given alias prefix.

        Args:
            alias: The IndexAlias to find indices for.

        Returns:
            List[str]: List of index names matching the alias pattern.
        """
        try:
            pattern = f"{alias.value}_*"
            indices = self.client.indices.get(index=pattern)
            return list(indices.keys())
        except Exception as e:
            logger.error(f"Error retrieving indices with prefix '{alias}_': {e}")
            return []

    def index_exists(self, index_name: str) -> bool:
        return bool(self.client.indices.exists(index=index_name))

    def index_empty(self, index_name: str) -> bool:
        """
        Check if the specified index is empty.
        Args:
            index_name (str): The name of the index to check.
        Returns:
            bool: True if the index is empty, False otherwise.
        """
        try:
            response = self.client.count(index=index_name)
            count = response.get("count", 0)
            return count == 0
        except Exception as e:
            logger.error(f"Error checking if index '{index_name}' is empty: {e}")
            return True

    def index_has_field(self, index_name: str, field: str) -> bool:
        """
        Check if the given index has a mapping for the specified field.
        """
        try:
            mapping = self.client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]["mappings"].get("properties", {})
            return field in properties
        except KeyError as e:
            logger.error(f"Mapping structure error for index {index_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to get mapping for index {index_name}: {e}")
            return False

    def index_has_attachment_type(self, index: str, attachment_type: str) -> bool:
        """
        Check if the given index contains at least one document with the specified attachment type.
        """
        try:
            query = {
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"exists": {"field": "attachment"}},
                            {"term": {"attachment.type": attachment_type}},
                        ]
                    }
                },
            }
            result = self.client.search(index=index, body=query)
            return result.get("hits", {}).get("total", {}).get("value", 0) > 0

        except KeyError as e:
            logger.error(
                f"Unexpected response structure when searching index {index}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to search index {index} for attachment type {attachment_type}: {e}"
            )
            return False

    def get_most_recent_doc_date(self, index_name: str) -> Optional[datetime]:
        """
        Get the most recent processed date from the specified index.
        Args:
            index_name (str): The name of the index to query.
        Returns:
            str | None: The most recent processed date as a string, or None if not found.
        """
        if not self.index_exists(index_name=index_name) or self.index_empty(index_name):
            return None

        try:
            search = (
                Search(using=self.client, index=index_name)
                .sort({"date": {"order": "desc"}})
                .source(["date"])
                .extra(size=1)
            )
            result = search.execute()

            if result.hits:
                return datetime.fromisoformat(str(result.hits[0].date))
            return None
        except (ValueError, AttributeError) as e:
            logger.error(f"Error parsing date from '{index_name}': {e}")
            return None
        except Exception as e:
            logger.error(
                f"Error retrieving last processed date from '{index_name}': {e}"
            )
            return None

    def bulk_write(
        self,
        actions: list[dict],
        max_retries: int = 3,
    ) -> None:
        if not actions:
            logger.warning("bulk_write called with empty actions list")
            return

        attempts = 0
        while attempts < max_retries:
            try:
                success, errors = bulk(self.client, actions)
                if errors:
                    raise RuntimeError(f"Bulk indexing encountered errors: {errors}")
                logger.info(f"Successfully indexed {success} documents.")
                break  # Exit loop on success
            except RuntimeError:
                raise  # Re-raise RuntimeError from bulk indexing errors
            except Exception as e:
                attempts += 1
                logger.error(
                    f"Attempt {attempts}/{max_retries}: Failed to bulk write: {e}"
                )

        if attempts == max_retries:
            raise RuntimeError("Max retries reached for bulk write.")


class DatabaseAsync:
    """
    Base class for asynchronous Elasticsearch operations.
    Used by FastAPI APIs.
    """

    def __init__(self, connect: bool = True) -> None:
        self.es_client = None
        self.message_index_alias = IndexAlias.MESSAGE_INDEX_ALIAS
        self.service_index_alias = None
        # self.index_alias: IndexAlias = None
        if connect:  # might not need this
            self.connect()

    def connect(self) -> None:
        """
        Establish a connection to the Elasticsearch cluster.
        """
        try:
            self.es_client = AsyncElasticsearch(
                hosts=[f"http://{settings.elastic_host}:{settings.elastic_port}"],
                timeout=30,
                max_retries=10,
                retry_on_timeout=True,
            )
            # ping?
        except ConnectionError as e:
            print(f"Error connecting to Elasticsearch: {e}")
            raise
        except ConnectionTimeout as e:
            print(f"Elasticsearch connection timeout: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while connecting to Elasticsearch: {e}")
            raise

    @property
    def client(self) -> AsyncElasticsearch:
        """
        Return the Elasticsearch client, ensuring it is initialized.
        """
        if not self.es_client:
            raise RuntimeError(
                "Elasticsearch client is not initialized. Call `connect()` first."
            )
        return self.es_client

    async def close(self) -> None:
        """
        Close the Elasticsearch connection.
        """
        await self.client.close()

    async def search_similar_docs(
        self,
        size: Optional[int],
        query_vector: list[float],
        index_alias: IndexAlias,
        filter: Optional[ESQuery] = None,
    ) -> list[SemanticSearchResult]:
        search = AsyncSearch(using=self.es_client, index=index_alias.value)
        search = search.source(includes=["message_id"])
        search = search[
            0:size
        ]  # global top x results; (from all k top results per shard combined)

        search = search.knn(
            field="vector",
            k=1000,
            num_candidates=10000,
            query_vector=query_vector,
            filter=filter if filter else None,
        )

        response = await search.execute()

        message_scores = {}
        for hit in response.hits:
            message_id = hit.message_id
            doc_score = hit.meta.score

            # Keep only the highest score for each message_id
            message_scores[message_id] = max(
                message_scores.get(message_id, 0), doc_score
            )
        return [
            SemanticSearchResult(id=message_id, score=doc_score)
            for message_id, doc_score in message_scores.items()
        ]
