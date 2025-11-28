from datetime import datetime
from typing import Any, List, Optional

from celery.utils.log import get_task_logger

from common.database.database import DatabaseSync
from common.database.index_alias import IndexAlias
from common.database.message import TextSearchMessage, VectorizedTextMessage
from common.settings import settings
from common.utils import build_index_name

logger = get_task_logger(__name__)


class Database(DatabaseSync):
    def __init__(self, connect: bool = True) -> None:
        super().__init__(connect=connect)
        self.service_index_alias = IndexAlias.TEXT_VECTOR_INDEX_ALIAS.value

    def get_messages_to_embed(
        self,
        chat_id: str,
        size: int = 0,
        start_message_date: Optional[datetime] = None,
    ) -> list[TextSearchMessage]:
        """
        Retrieve documents from a specific chat that are not embedded so far and have one or
        more text fields of interest, sorted from oldest to newest based on the 'date' field.

        Args:
            size (int): The number of documents to retrieve in a batch. If 0, retrieve all matching documents.
            chat_id (str): The ID of the chat to retrieve messages for.
            start_message_date (Optional[datetime]): Filter to only include documents newer than this date.

        Returns:
            list[Message]: A list of Message objects.
        """
        index = build_index_name(IndexAlias.MESSAGE_INDEX_ALIAS, chat_id)

        if self.index_empty(index):
            logger.info(f"No messages found for index {index}.")
            return []

        return_fields = TextSearchMessage.get_defined_fields()
        must_clauses = [
            {
                "bool": {
                    "should": [
                        {"exists": {"field": field}}
                        for field in settings.text_embedding_fields
                    ],
                    "minimum_should_match": 1,
                }
            },
        ]

        if start_message_date:
            must_clauses.append(
                {"range": {"date": {"gt": start_message_date.isoformat()}}}
            )

        search_body = {
            "query": {"bool": {"must": must_clauses}},
            "sort": [{"date": {"order": "asc"}}],
            "_source": return_fields,
        }
        if size > 0:
            search_body["size"] = size

        response = self.client.search(index=index, body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return [TextSearchMessage(id=hit["_id"], **hit["_source"]) for hit in hits]

    def create_embedding_index(  # no special field analyzer is needed i think, we might downgrade it?
        self,
        chat_id: str,
    ) -> None:
        """
        Creates an index for vectorized messages of a given chat with specific field analyzers
        depending on the language of the messages.

        Performs:
            - Checks if the index with the name 'vectorized_messages_<chat_id>' already exists.
            - If not, creates a new index with mapping.
            - The mapping includes text fields for all possible embedding fields, and embedding subfields and additional fields like id and date.

        Raises:
            Exception: If an error occurs during the index creation process, appropriate Elasticsearch
            exceptions will be raised.
        """
        index_name = build_index_name(IndexAlias.TEXT_VECTOR_INDEX_ALIAS, chat_id)
        embedding_fields = settings.text_embedding_fields

        if not self.index_exists(index_name):
            properties: dict[str, dict[str, Any]] = {
                "id": {"type": "keyword"},
                "chat_id": {"type": "keyword"},
                "language": {"type": "keyword"},
                "date": {"type": "date"},
                "processed_at": {"type": "date"},
            }
            # Add all possible text and vector fields
            for field in embedding_fields:
                properties[field] = {"type": "text"}
                embedding_field = f"{field}_vector"
                properties[embedding_field] = {
                    "type": "dense_vector",
                    "element_type": "float",
                    "dims": settings.text_embedding_dimension,
                    "similarity": "cosine",  # default
                    "index_options": {"type": "int8_hnsw"},  # default, quantized
                }

            index_config = {
                "settings": {
                    "number_of_shards": 1,  # TODO
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
                        "aliases": {"vectorized_messages": {}},
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

    def create_index_actions_per_chat(
        self, chat_id: str, messages: List[VectorizedTextMessage]
    ) -> List[dict]:
        """
        Create update actions for the given vectorized result messages.

        Args:
            messages (List[VectorizedTextMessage]): The vectorized results to create update actions for.

        Returns:
            List[dict]: A list of Elasticsearch update actions.
        """
        actions = []

        for message in messages:
            index_name = build_index_name(
                index_alias=IndexAlias.TEXT_VECTOR_INDEX_ALIAS, chat_id=chat_id
            )
            action = {
                "_op_type": "index",
                "_index": index_name,
                "_source": message.to_index_dict(),
            }
            actions.append(action)

        return actions
