from datetime import datetime
from typing import List

from celery.utils.log import get_task_logger
from elasticsearch.dsl import Search, UpdateByQuery

from common.database.database import DatabaseSync
from common.database.index_alias import IndexAlias
from common.database.message import ClassificationMessage as Message
from common.database.message import ClassificationResultMessage
from common.settings import settings
from common.utils import build_index_name

# from worker.database.utils import create_update_actions, get_chat_id_from_index_name

logger = get_task_logger(__name__)


class Database(DatabaseSync):
    def __init__(self, connect: bool = True) -> None:
        super().__init__(connect=connect)

    def _build_unclassified_query(
        self,
        chat_id: str | None = None,
        last_processed_date: datetime | None = None,
    ) -> tuple[Search, str]:
        """
        Build a search query for unclassified documents.

        Args:
            chat_id: Optional chat ID to filter by.
            last_processed_date: Optional timestamp filter to retrieve only documents
                                created after this date.

        Returns:
            Tuple of (Search object, index name)
        """
        if chat_id:
            index = build_index_name(IndexAlias.MESSAGE_INDEX_ALIAS, chat_id)
        else:
            index = IndexAlias.MESSAGE_INDEX_ALIAS.value

        should_clauses = [
            {"exists": {"field": field}} for field in settings.classification_fields
        ]
        must_clauses = [
            {
                "bool": {"must_not": [{"exists": {"field": "classification"}}]}
            },  # No classification object
            {"bool": {"should": should_clauses, "minimum_should_match": 1}},
            {"terms": {"language": settings.classification_languages}},
        ]
        must_not_clauses = [
            {
                "bool": {
                    "should": [
                        {"exists": {"field": field}}
                        for field in settings.classification_fields
                    ],
                    "minimum_should_match": 2,
                }
            }
        ]

        if last_processed_date:
            must_clauses.append(
                {"range": {"date": {"gt": last_processed_date.isoformat()}}}
            )

        search = Search(using=self.client, index=index)
        search = search.filter("bool", must=must_clauses, must_not=must_not_clauses)
        search = search.sort({"date": {"order": "asc"}})
        search = search.source(includes=Message.get_defined_fields())

        return search, index

    def get_unclassified_docs_count(
        self,
        chat_id: str | None = None,
        last_processed_date: datetime | None = None,
    ) -> int:
        """
        Get count of unclassified documents from Elasticsearch.

        Args:
            chat_id: Optional chat ID to filter by.
            last_processed_date: Optional timestamp filter to retrieve only documents
                                created after this date.

        Returns:
            Count of matching documents.
        """
        search, index = self._build_unclassified_query(chat_id, last_processed_date)

        if not self.index_exists(index) or self.index_empty(index):
            logger.warning(f"Index '{index}' empty or not found.")
            return 0

        return search.count()

    def get_unclassified_docs(
        self,
        size: int = 0,
        chat_id: str | None = None,
        last_processed_date: datetime | None = None,
    ) -> list[Message]:
        """
        Retrieve unclassified documents from Elasticsearch.

        Args:
            size: The number of documents to return. If 0, returns all matching documents.
            chat_id: Optional chat ID to filter by.
            last_processed_date: Optional timestamp filter to retrieve only documents
                                created after this date.

        Returns:
            List of Message instances.
        """
        search, index = self._build_unclassified_query(chat_id, last_processed_date)

        if not self.index_exists(index) or self.index_empty(index):
            logger.warning(f"Index '{index}' empty or not found.")
            return []

        if size > 0:
            search = search.extra(size=size)

        response = search.execute()
        hits = response["hits"]["hits"]
        return [Message(id=hit["_id"], **hit["_source"]) for hit in hits]

    def create_update_actions_per_chat(
        self, chat_id: str, messages: List[ClassificationResultMessage]
    ) -> List[dict]:
        """
        Create update actions for the given classification result messages.

        Args:
            messages (List[ClassificationResultMessage]): The classification results to create update actions for.

        Returns:
            List[dict]: A list of Elasticsearch update actions.
        """
        actions = []

        for message in messages:
            # Extract chat_id from message_id to build the correct index name
            index_name = build_index_name(
                index_alias=IndexAlias.MESSAGE_INDEX_ALIAS, chat_id=chat_id
            )
            action = {
                "_op_type": "update",
                "_index": index_name,
                "_id": message.message_id,
                "doc": {
                    "classification": message.classification_result.dict()
                },  # Wrap in "classification" field
                "doc_as_upsert": False,  # Don't create if document doesn't exist
            }
            actions.append(action)

        return actions

    def remove_classification_results(self) -> None:
        """
        Remove all classification objects from documents in the Elasticsearch index
        where the 'classification' field exists. Much simpler now with the new object structure.

        Raises:
            Exception: If the update process fails, an exception is raised and logged.
        """
        index = IndexAlias.MESSAGE_INDEX_ALIAS.value

        ubq = (
            UpdateByQuery(using=self.client, index=index)
            .query("exists", field="classification")
            .script(
                source="ctx._source.remove('classification');",
                lang="painless",
            )
        )
        ubq.execute()
        self.client.indices.refresh(index=index)
