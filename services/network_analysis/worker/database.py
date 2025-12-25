"""
Database operations for Network Analysis service.
"""

from typing import Any, Dict, List, Optional

from celery.utils.log import get_task_logger
from elasticsearch import Elasticsearch

from common.settings import settings

logger = get_task_logger(__name__)


class Database:
    """Database interface for network analysis operations."""

    def __init__(self):
        """Initialize Elasticsearch client."""
        self.client = Elasticsearch(
            [
                {
                    "host": settings.elastic_host,
                    "port": settings.elastic_port,
                    "scheme": "http",
                }
            ],
            timeout=300,
            max_retries=10,
            retry_on_timeout=True,
        )

        # Ensure indices exist
        self._ensure_indices()

    def _ensure_indices(self):
        """Ensure required indices exist."""
        indices = {
            'network_metadata': {
                'mappings': {
                    'properties': {
                        'network_id': {'type': 'keyword'},
                        'created_at': {'type': 'date'},
                        'num_messages_analyzed': {'type': 'integer'},
                        'statistics': {'type': 'object'},
                        'num_communities': {'type': 'integer'},
                        'num_coordination_events': {'type': 'integer'}
                    }
                }
            },
            'network_nodes': {
                'mappings': {
                    'properties': {
                        'node_id': {'type': 'keyword'},
                        'title': {'type': 'text'},
                        'type': {'type': 'keyword'},
                        'message_count': {'type': 'integer'},
                        'betweenness': {'type': 'float'},
                        'eigenvector': {'type': 'float'},
                        'pagerank': {'type': 'float'},
                        'in_degree': {'type': 'float'},
                        'out_degree': {'type': 'float'},
                        'community_id': {'type': 'keyword'},
                        'community_size': {'type': 'integer'},
                        'updated_at': {'type': 'date'}
                    }
                }
            },
            'network_edges': {
                'mappings': {
                    'properties': {
                        'edge_id': {'type': 'keyword'},
                        'source': {'type': 'keyword'},
                        'target': {'type': 'keyword'},
                        'type': {'type': 'keyword'},
                        'weight': {'type': 'integer'},
                        'forward_count': {'type': 'integer'},
                        'mention_count': {'type': 'integer'},
                        'updated_at': {'type': 'date'}
                    }
                }
            },
            'network_communities': {
                'mappings': {
                    'properties': {
                        'community_id': {'type': 'keyword'},
                        'members': {'type': 'keyword'},
                        'size': {'type': 'integer'},
                        'created_at': {'type': 'date'}
                    }
                }
            },
            'network_coordination': {
                'mappings': {
                    'properties': {
                        'event_id': {'type': 'keyword'},
                        'type': {'type': 'keyword'},
                        'channels': {'type': 'keyword'},
                        'num_channels': {'type': 'integer'},
                        'num_posts': {'type': 'integer'},
                        'time_window_seconds': {'type': 'float'},
                        'first_post_time': {'type': 'date'},
                        'last_post_time': {'type': 'date'},
                        'sample_text': {'type': 'text'},
                        'message_ids': {'type': 'keyword'},
                        'created_at': {'type': 'date'}
                    }
                }
            }
        }

        for index_name, index_config in indices.items():
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=index_config)
                logger.info(f"Created index: {index_name}")

    def get_messages_for_network_analysis(
        self,
        chat_id: Optional[str] = None,
        limit: int = 50000
    ) -> List[Dict[str, Any]]:
        """
        Get messages for network analysis.

        Args:
            chat_id: Optional chat ID filter
            limit: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries
        """
        query = {"match_all": {}}

        if chat_id:
            query = {"term": {"chat_id": chat_id}}

        try:
            response = self.client.search(
                index="messages_*",
                query=query,
                size=limit,
                _source=[
                    "id",
                    "chat_id",
                    "chat_title",
                    "text",
                    "date",
                    "forward",
                    "entities"
                ],
                sort=[{"date": {"order": "desc"}}]
            )

            messages = [hit["_source"] for hit in response["hits"]["hits"]]
            logger.info(f"Retrieved {len(messages)} messages for network analysis")

            return messages

        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return []

    def get_network_nodes(
        self,
        community_id: Optional[str] = None,
        min_centrality: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get network nodes with optional filters.

        Args:
            community_id: Filter by community
            min_centrality: Minimum pagerank centrality
            limit: Maximum number of nodes to retrieve

        Returns:
            List of node dictionaries
        """
        query = {"match_all": {}}
        filters = []

        if community_id:
            filters.append({"term": {"community_id": community_id}})

        if min_centrality is not None:
            filters.append({"range": {"pagerank": {"gte": min_centrality}}})

        if filters:
            query = {"bool": {"filter": filters}}

        try:
            response = self.client.search(
                index="network_nodes",
                query=query,
                size=limit,
                sort=[{"pagerank": {"order": "desc"}}]
            )

            nodes = [hit["_source"] for hit in response["hits"]["hits"]]
            logger.info(f"Retrieved {len(nodes)} network nodes")

            return nodes

        except Exception as e:
            logger.error(f"Error retrieving nodes: {e}")
            return []

    def get_network_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        edge_type: Optional[str] = None,
        min_weight: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get network edges with optional filters.

        Args:
            source: Filter by source node
            target: Filter by target node
            edge_type: Filter by edge type ('forward' or 'mention')
            min_weight: Minimum edge weight
            limit: Maximum number of edges to retrieve

        Returns:
            List of edge dictionaries
        """
        filters = []

        if source:
            filters.append({"term": {"source": source}})

        if target:
            filters.append({"term": {"target": target}})

        if edge_type:
            filters.append({"term": {"type": edge_type}})

        if min_weight is not None:
            filters.append({"range": {"weight": {"gte": min_weight}}})

        query = {"bool": {"filter": filters}} if filters else {"match_all": {}}

        try:
            response = self.client.search(
                index="network_edges",
                query=query,
                size=limit,
                sort=[{"weight": {"order": "desc"}}]
            )

            edges = [hit["_source"] for hit in response["hits"]["hits"]]
            logger.info(f"Retrieved {len(edges)} network edges")

            return edges

        except Exception as e:
            logger.error(f"Error retrieving edges: {e}")
            return []

    def get_communities(self, min_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get detected communities.

        Args:
            min_size: Minimum community size

        Returns:
            List of community dictionaries
        """
        query = {"match_all": {}}

        if min_size is not None:
            query = {"range": {"size": {"gte": min_size}}}

        try:
            response = self.client.search(
                index="network_communities",
                query=query,
                size=100,
                sort=[{"size": {"order": "desc"}}]
            )

            communities = [hit["_source"] for hit in response["hits"]["hits"]]
            logger.info(f"Retrieved {len(communities)} communities")

            return communities

        except Exception as e:
            logger.error(f"Error retrieving communities: {e}")
            return []

    def get_coordination_events(
        self,
        min_channels: Optional[int] = None,
        max_time_window: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get coordination events.

        Args:
            min_channels: Minimum number of channels involved
            max_time_window: Maximum time window in seconds
            limit: Maximum number of events to retrieve

        Returns:
            List of coordination event dictionaries
        """
        filters = []

        if min_channels is not None:
            filters.append({"range": {"num_channels": {"gte": min_channels}}})

        if max_time_window is not None:
            filters.append({"range": {"time_window_seconds": {"lte": max_time_window}}})

        query = {"bool": {"filter": filters}} if filters else {"match_all": {}}

        try:
            response = self.client.search(
                index="network_coordination",
                query=query,
                size=limit,
                sort=[{"num_channels": {"order": "desc"}}]
            )

            events = [hit["_source"] for hit in response["hits"]["hits"]]
            logger.info(f"Retrieved {len(events)} coordination events")

            return events

        except Exception as e:
            logger.error(f"Error retrieving coordination events: {e}")
            return []
