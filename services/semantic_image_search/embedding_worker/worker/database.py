from datetime import datetime
from typing import Any, List, Optional

from celery.utils.log import get_task_logger

from common.database.database import DatabaseSync
from common.database.index_alias import IndexAlias
from common.database.message import ImageSearchMessage as Message
from common.database.message import VectorizedImageMessage
from common.settings import settings
from common.utils import build_index_name, get_chat_id_from_index_name

logger = get_task_logger(__name__)


class Database(DatabaseSync):
    def __init__(self, connect: bool = True) -> None:
        super().__init__(connect=connect)
        self.service_index_alias = IndexAlias.IMAGE_VECTOR_INDEX_ALIAS.value

    def create_vectorized_image_index(
        self,
        index_name: str,
    ) -> None:
        """
        Creates an index for image vectors corresponding to messages of a given chat.

        Performs:
            - Checks if the index with the name 'vectorized_images_<chat_id>' already exists.
            - If not, creates a new index with mapping.

        Raises:
            Exception: If an error occurs during the index creation process, appropriate Elasticsearch
            exceptions will be raised.
        """

        if not self.index_exists(index_name):
            properties: dict[str, dict[str, Any]] = {
                "message_id": {"type": "keyword"},
                "chat_id": {"type": "keyword"},
                "date": {"type": "date"},
                "processed_at": {"type": "date"},
                "attachment_type": {"type": "keyword"},
            }
            properties["vector"] = {
                "type": "dense_vector",
                "element_type": "float",
                "dims": settings.image_embedding_dimension,
                "similarity": "cosine",  # default
                "index_options": {"type": "int8_hnsw"},  # default, quantized
                # "index_options": {"type": "hnsw"} # if quantized version above does not work
            }

            index_config = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                "mappings": {
                    "properties": properties,
                },
            }
            try:
                self.client.indices.create(
                    index=index_name,
                    body={
                        "settings": index_config["settings"],
                        "mappings": index_config["mappings"],
                        "aliases": {f"{self.service_index_alias}": {}},
                    },
                )
                logger.info(f"Created index: {index_name}")

            except Exception as e:
                if "resource_already_exists_exception" in str(e):
                    logger.error(
                        f"Index {index_name} already exists (caught during creation), refreshing."
                    )
                else:
                    logger.error(f"Failed to create index {index_name}: {e}")
                    raise
            finally:
                self.client.indices.refresh(index=index_name)
        else:
            logger.info(f"Index {index_name} already exists")

    def get_all_message_indices_with_new_images(  # TODO: check if can be used in common database with filter
        self, attachment_type: str = "photo"
    ) -> list[tuple[str, Optional[datetime]]]:
        """
        Returns a list of (message_index_name, last_vectorized_date) for indices that
        contain messages with the given attachment type and potentially new data.
        If no last_vectorized_date exists, returns None.

        Only indices that have:
        - The specified attachment_type (e.g., "photo")
        - The 'date' field in their mapping (required to sort)
        - New messages after last_vectorized_date (if available)
        are returned.
        """
        indices_with_images = []
        all_indices = self.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)
        if not all_indices:
            return []

        for index in all_indices:
            chat_id = get_chat_id_from_index_name(index)

            # Skip indices that have no 'date' field (can't sort)
            if not self.index_has_field(index, "date"):
                logger.info(
                    f"Skipping index {index} for chat {chat_id}: no 'date' field in mapping."
                )
                continue

            # Skip indices that have no messages with the specified attachment_type
            if not self.index_has_attachment_type(index, attachment_type):
                logger.info(
                    f"Skipping index {index} for chat {chat_id}: no messages with attachment type '{attachment_type}'."
                )
                continue

            # fetch last vectorized date; None if no vectorized data exists
            index_name = build_index_name(IndexAlias.IMAGE_VECTOR_INDEX_ALIAS, chat_id)
            last_vectorized_date = self.get_most_recent_doc_date(index_name)

            # If no last vectorized date, we want to process all messages
            if last_vectorized_date is None:
                indices_with_images.append((index, None))
                continue

            # Check if new messages exist after last_vectorized_date
            new_messages_count = self.get_images_to_vectorize_count(
                chat_id=chat_id,
                start_date=last_vectorized_date,
            )
            if new_messages_count > 0:
                indices_with_images.append((index, last_vectorized_date))

        return indices_with_images

    def get_images_to_vectorize_count(
        self,
        chat_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
    ) -> int:
        """
        Count image messages in an index.

        Args:
            chat_id (Optional[str]): Chat ID to filter by.
            start_date (Optional[datetime]): Only include documents after this date.

        Returns:
            int: Number of matching documents.
        """
        index = f"messages_{chat_id}" if chat_id else "messages"

        if not self.index_exists(index_name=index):
            logger.warning(f"Index '{index}' does not exist.")
            return 0

        count_body = {"query": self._build_image_query(start_date=start_date)}

        response = self.client.count(index=index, body=count_body)
        return response.get("count", 0)

    def _build_image_query(self, start_date: Optional[datetime] = None) -> dict:
        """
        Helper to build the ES 'must' clauses for images.

        Args:
            start_date (Optional[datetime]): Only include documents after this date.

        Returns:
            dict: ES bool query dict.
        """
        must_clauses = [
            {"exists": {"field": "attachment.storage_refs"}},
            {"terms": {"attachment.type": settings.image_attachment_types}},
        ]

        if start_date:
            must_clauses.append({"range": {"date": {"gt": start_date.isoformat()}}})

        return {"bool": {"must": must_clauses}}

    def get_images_to_vectorize_docs(  # TODO: check if can be used in common
        self,
        chat_id: Optional[str] = None,
        size: int = 0,
        start_date: Optional[datetime] = None,
    ) -> list[Message]:
        """
        Retrieve image messages as Message objects.

        Args:
            chat_id (Optional[str]): Chat ID to filter by.
            size (int): Number of documents to retrieve (0 = all).
            start_date (Optional[datetime]): Only include documents after this date.

        Returns:
            list[Message]: List of messages with images.
        """
        index = f"messages_{chat_id}" if chat_id else "messages"

        if not self.index_exists(index_name=index):
            logger.warning(f"Index '{index}' does not exist.")
            return []

        return_fields = Message.get_defined_fields()

        search_body = {
            "query": self._build_image_query(start_date=start_date),
            "sort": [{"date": {"order": "asc"}}],
            "_source": return_fields,
        }
        if size > 0:
            search_body["size"] = size

        response = self.client.search(index=index, body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return [Message(id=hit["_id"], **hit["_source"]) for hit in hits]

    def create_index_actions_per_chat(
        self, chat_id: str, messages: List[VectorizedImageMessage]
    ) -> List[dict]:
        """
        Create update actions for the given vectorized result messages.

        Args:
            messages (List[VectorizedImageMessage]): The vectorized results to create update actions for.

        Returns:
            List[dict]: A list of Elasticsearch update actions.
        """
        actions = []

        for message in messages:
            index_name = build_index_name(
                index_alias=IndexAlias.IMAGE_VECTOR_INDEX_ALIAS, chat_id=chat_id
            )
            action = {
                "_op_type": "index",
                "_index": index_name,
                "_source": message.to_index_dict(),
            }
            actions.append(action)

        return actions
