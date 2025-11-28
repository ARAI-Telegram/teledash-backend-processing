from datetime import datetime
from typing import Any, Dict, Optional

from celery.utils.log import get_task_logger
from elasticsearch.dsl import Q, Search, UpdateByQuery

from common.database.database import DatabaseSync
from common.database.index_alias import IndexAlias
from common.database.message import ASRMessage
from common.settings import settings
from common.utils import build_index_name

logger = get_task_logger(__name__)


class Database(DatabaseSync):
    def __init__(self, connect: bool = True) -> None:
        super().__init__(connect=connect)

    def _build_unprocessed_audio_query(
        self,
        chat_id: str | None = None,
        last_processed_date: Optional[datetime] = None,
    ) -> tuple[Search, str]:
        """
        Build a search query for unprocessed audio attachments.

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

        must_clauses = [
            {
                "bool": {
                    "must_not": [
                        {"exists": {"field": "attachment.transcription_status"}}
                    ]
                }
            },
            {"terms": {"attachment.type": settings.asr_attachment_types}},
            # ToDo: documents can be audio/video files and should be processed
        ]

        if last_processed_date:
            must_clauses.append(
                {"range": {"date": {"gt": last_processed_date.isoformat()}}}
            )

        search = Search(using=self.client, index=index)
        search = search.filter("bool", must=must_clauses)
        search = search.sort({"date": {"order": "asc"}})
        search = search.source(includes=ASRMessage.get_defined_fields())

        return search, index

    def get_unprocessed_audio_count(
        self,
        chat_id: str | None = None,
        last_processed_date: Optional[datetime] = None,
    ) -> int:
        """
        Get count of messages with unprocessed audio attachments.

        Searches for documents with media attachments (types defined in settings.asr_attachment_types)
        that don't have a transcription status.

        Args:
            chat_id: Optional chat ID to filter by.
            last_processed_date: Optional timestamp filter to retrieve only documents
                                created after this date.

        Returns:
            Count of matching documents.

        Raises:
            Exception: If an error occurs during Elasticsearch query execution.
        """
        search, index = self._build_unprocessed_audio_query(
            chat_id, last_processed_date
        )

        if not self.index_exists(index_name=index):
            logger.warning(f"Index '{index}' does not exist.")
            return 0

        try:
            return search.count()
        except Exception as e:
            logger.error("Elasticsearch count error:", e)
            raise

    def get_unprocessed_audio_docs(
        self,
        size: int = 0,
        chat_id: str | None = None,
        last_processed_date: Optional[datetime] = None,
    ) -> list[ASRMessage]:
        """
        Retrieve messages with unprocessed audio attachments from Elasticsearch.

        Searches for documents with media attachments (types defined in settings.asr_attachment_types)
        that don't have a transcription status. Results are sorted by date in ascending order.

        Args:
            size: Maximum number of documents to return. If 0 (default), returns all matching documents.
            chat_id: Optional chat ID to filter by.
            last_processed_date: Optional timestamp filter to retrieve only documents
                                created after this date.

        Returns:
            List of ASRMessage objects.

        Raises:
            Exception: If an error occurs during Elasticsearch query execution.
        """
        search, index = self._build_unprocessed_audio_query(
            chat_id, last_processed_date
        )

        if not self.index_exists(index_name=index):
            logger.warning(f"Index '{index}' does not exist.")
            return []

        if size > 0:
            search = search.extra(size=size)

        try:
            response = search.execute()
            hits = response["hits"]["hits"]
            return [ASRMessage(id=hit["_id"], **hit["_source"]) for hit in hits]
        except Exception as e:
            logger.error("Elasticsearch query error:", e)
            raise

    def update_one_retry(
        self,
        index_name: str,
        doc_id: str,
        update_doc_body: Dict[str, Any],
        max_retries: int = 3,
    ) -> None:
        """
        Update a document in Elasticsearch with automatic retries on failure.

        Attempts to update a document by its ID in the specified index. If the update fails,
        it will retry up to the maximum number of attempts specified.

        Args:
            index_name: The name of the Elasticsearch index to update.
            doc_id: The ID of the document to update.
            update_doc_body: Dictionary containing the fields and values to update.
            max_retries: Maximum number of retry attempts (default: 3).

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        attempts = 0

        es_update_payload = {
            "doc": update_doc_body,
            "doc_as_upsert": False,
        }

        while attempts < max_retries:
            try:
                self.client.update(
                    index=index_name, id=doc_id, body=es_update_payload, refresh=True
                )
                logger.info(
                    f"Successfully updated document ID {doc_id} in index {index_name}."
                )
                return  # Exit on successful update
            except Exception as e:
                attempts += 1
                logger.error(
                    f"Attempt {attempts} of document update failed for doc ID {doc_id} "
                    f"in index {index_name}: {e}"
                )
                if attempts >= max_retries:
                    logger.error(
                        f"Document update failed for doc ID {doc_id} in index {index_name} "
                        f"after {max_retries} attempts."
                    )
                    raise RuntimeError(
                        f"Document update failed for doc ID {doc_id} in index {index_name} "
                        f"after {max_retries} attempts"
                    ) from e

    def remove_all_transcription_fields(self) -> None:
        """
        Removes all transcription-related fields from documents in the Elasticsearch index.

        This method targets documents where the field 'attachment.transcription_status' exists
        and removes the following fields:
        - attachment.transcription
        - attachment.transcription_status
        - attachment.transcription_language
        - attachment.transcription_language_probability

        The removal is performed using Elasticsearch's UpdateByQuery API.
        The index is refreshed as part of the operation to make changes immediately visible.

        Raises:
            ElasticsearchException: If the update process fails, the exception is caught,
                                   logged, and re-raised.
        """
        index = IndexAlias.MESSAGE_INDEX_ALIAS.value

        ubq = (
            UpdateByQuery(using=self.client, index=index)
            .query("exists", field="attachment.transcription_status")
            .script(
                source="""
                ctx._source.attachment.remove('transcription');
                ctx._source.attachment.remove('transcription_status');
                ctx._source.attachment.remove('transcription_language');
                ctx._source.attachment.remove('transcription_language_probability');
            """,
                lang="painless",
            )
        )
        try:
            response = ubq.execute()
            print(
                f"Removed transcription fields. "
                f"Updated: {response.updated}, "
                f"Took: {response.took}ms, "
                f"Total: {response.total}"
            )

            self.client.indices.refresh(index=index)

        except Exception as e:
            logger.error(f"Failed to remove all transcription fields: {e}")
            raise

    def clear_pending_states(self) -> None:
        """
        Clears pending transcription states from documents in the Elasticsearch index.
        """
        index = IndexAlias.MESSAGE_INDEX_ALIAS.value

        ubq = (
            UpdateByQuery(using=self.client, index=index)
            .query(Q("term", attachment__transcription_status="PENDING"))
            .script(
                source="""
                ctx._source.attachment.remove('transcription_status');
                """,
                lang="painless",
            )
        )
        try:
            response = ubq.execute()
            print(
                f"Cleared pending states. "
                f"Updated: {response.updated}, "
                f"Took: {response.took}ms, "
                f"Total: {response.total}"
            )

            self.client.indices.refresh(index=index)

        except Exception as e:
            logger.error(f"Failed to clear pending states: {e}")
            raise
