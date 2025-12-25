"""
Celery tasks for Network Analysis service.
"""

from datetime import datetime
from typing import List, Optional

from celery import Task
from celery.utils.log import get_task_logger

from common.redis_queue_manager import QueueChecker
from common.settings import settings
from main import app
from worker.database import Database
from worker.network_analysis.network_builder import (
    TelegramNetworkAnalyzer,
    calculate_network_statistics,
)

logger = get_task_logger(__name__)


class NetworkAnalysisTask(Task):
    """
    Abstraction of Celery's Task class to support network analyzer instance.
    """

    abstract = True
    _analyzer = None  # Class-level variable shared across instances

    def __call__(self, *args, **kwargs):
        """
        Initialize analyzer on first call.
        """
        if NetworkAnalysisTask._analyzer is None:
            logger.info("Initializing network analyzer...")
            NetworkAnalysisTask._analyzer = TelegramNetworkAnalyzer()
            logger.info("Network analyzer initialized successfully.")
        return self.run(*args, **kwargs)

    @property
    def analyzer(self):
        """Get the shared analyzer instance."""
        return NetworkAnalysisTask._analyzer

    @analyzer.setter
    def analyzer(self, value):
        """Set the shared analyzer instance."""
        NetworkAnalysisTask._analyzer = value


@app.task(
    name="network_analysis.init_network_analysis",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 600},
)
def init_network_analysis() -> None:
    """
    Initialize network analysis process.
    """
    logger.info("Starting network analysis initialization")

    try:
        db = Database()
        if not db.client.ping():
            raise ConnectionError("Elasticsearch ping failed.")
        logger.info("Connected to database for network analysis")
    except Exception as e:
        logger.error(f"Error during database connection: {e}")
        raise

    queue_checker = QueueChecker()

    if not queue_checker.is_queue_empty("network_analysis"):
        pending_count = queue_checker.get_queue_length("network_analysis")
        logger.info(
            f"Skipping init_network_analysis - queue already has {pending_count} pending tasks"
        )
        return

    logger.info("Network analysis queue is empty, starting initialization...")

    try:
        # Get messages for network analysis
        messages = db.get_messages_for_network_analysis(limit=50000)

        if not messages:
            logger.info("No messages found for network analysis")
            return

        logger.info(f"Found {len(messages)} messages for network analysis")

        # Enqueue network analysis task
        build_and_store_network.apply_async(
            kwargs={"messages": messages},
            queue="network_analysis"
        )

        logger.info("Network analysis task enqueued successfully")

    except Exception as e:
        logger.error(f"Error during network analysis initialization: {e}")
        raise


@app.task(
    name="network_analysis.build_and_store_network",
    base=NetworkAnalysisTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 1800},
)
def build_and_store_network(messages: list) -> dict:
    """
    Build network graph from messages and store results.

    Args:
        messages: List of message dictionaries

    Returns:
        Dictionary with processing results
    """
    if not messages:
        logger.warning("No messages provided for network analysis")
        return {"status": "skipped", "reason": "no_messages"}

    logger.info(f"Building network from {len(messages)} messages")

    try:
        # Create new analyzer instance for this task
        analyzer = TelegramNetworkAnalyzer()

        # Build network graph
        graph = analyzer.build_channel_network(messages)

        # Calculate centrality metrics
        centrality_metrics = analyzer.calculate_centrality_metrics()

        # Detect communities
        communities = analyzer.detect_communities()

        # Find bridge channels
        bridges = analyzer.find_bridge_channels(threshold=0.01)

        # Detect coordinated behavior
        coordination_events = analyzer.detect_coordinated_behavior(
            messages,
            time_window=3600
        )

        # Calculate network statistics
        network_stats = calculate_network_statistics(graph)

        # Store results in database
        db = Database()

        # Store network metadata
        network_metadata = {
            'network_id': 'telegram_network_v1',
            'created_at': datetime.utcnow().isoformat(),
            'num_messages_analyzed': len(messages),
            'statistics': network_stats,
            'num_communities': len(communities),
            'num_coordination_events': len(coordination_events)
        }

        db.client.index(
            index='network_metadata',
            id='telegram_network_v1',
            document=network_metadata
        )

        # Store node centrality data
        for node_id in graph.nodes():
            node_data = {
                'node_id': node_id,
                'title': graph.nodes[node_id].get('title'),
                'type': graph.nodes[node_id].get('type'),
                'message_count': graph.nodes[node_id].get('message_count', 0),
                'betweenness': centrality_metrics.get('betweenness', {}).get(node_id, 0),
                'eigenvector': centrality_metrics.get('eigenvector', {}).get(node_id, 0),
                'pagerank': centrality_metrics.get('pagerank', {}).get(node_id, 0),
                'in_degree': centrality_metrics.get('in_degree', {}).get(node_id, 0),
                'out_degree': centrality_metrics.get('out_degree', {}).get(node_id, 0),
                'updated_at': datetime.utcnow().isoformat()
            }

            # Find which community this node belongs to
            for idx, community in enumerate(communities):
                if node_id in community:
                    node_data['community_id'] = f"community_{idx}"
                    node_data['community_size'] = len(community)
                    break

            db.client.index(
                index='network_nodes',
                id=node_id,
                document=node_data
            )

        # Store edges
        for source, target, edge_data in graph.edges(data=True):
            edge_id = f"{source}_{target}"
            edge_doc = {
                'edge_id': edge_id,
                'source': source,
                'target': target,
                'type': edge_data.get('type'),
                'weight': edge_data.get('weight', 0),
                'forward_count': edge_data.get('forward_count', 0),
                'mention_count': edge_data.get('mention_count', 0),
                'updated_at': datetime.utcnow().isoformat()
            }

            db.client.index(
                index='network_edges',
                id=edge_id,
                document=edge_doc
            )

        # Store communities
        for idx, community in enumerate(communities):
            community_doc = {
                'community_id': f"community_{idx}",
                'members': list(community),
                'size': len(community),
                'created_at': datetime.utcnow().isoformat()
            }

            db.client.index(
                index='network_communities',
                id=f"community_{idx}",
                document=community_doc
            )

        # Store coordination events
        for idx, event in enumerate(coordination_events):
            event['event_id'] = f"coordination_{idx}_{datetime.utcnow().timestamp()}"
            event['created_at'] = datetime.utcnow().isoformat()

            db.client.index(
                index='network_coordination',
                document=event
            )

        logger.info(
            f"Network analysis complete. "
            f"Nodes: {graph.number_of_nodes()}, "
            f"Edges: {graph.number_of_edges()}, "
            f"Communities: {len(communities)}, "
            f"Coordination events: {len(coordination_events)}"
        )

        return {
            "status": "success",
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_communities": len(communities),
            "num_coordination_events": len(coordination_events),
            "num_messages_processed": len(messages),
        }

    except Exception as e:
        logger.error(f"Error during network analysis: {e}")
        raise


@app.task(
    name="network_analysis.get_node_centrality",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 300},
)
def get_node_centrality(node_id: str) -> dict:
    """
    Get centrality metrics for a specific node.

    Args:
        node_id: Channel or user ID

    Returns:
        Dictionary with centrality metrics
    """
    logger.info(f"Getting centrality metrics for node {node_id}")

    try:
        db = Database()

        # Retrieve from database
        response = db.client.get(
            index='network_nodes',
            id=node_id
        )

        return {
            "status": "success",
            "node_data": response['_source']
        }

    except Exception as e:
        logger.error(f"Error getting node centrality: {e}")
        return {"status": "error", "reason": str(e)}


@app.task(
    name="network_analysis.find_shortest_path",
    base=NetworkAnalysisTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 300},
)
def find_shortest_path(source: str, target: str) -> dict:
    """
    Find shortest path between two nodes.

    Args:
        source: Source node ID
        target: Target node ID

    Returns:
        Dictionary with path information
    """
    logger.info(f"Finding shortest path from {source} to {target}")

    try:
        analyzer = NetworkAnalysisTask._analyzer

        if analyzer is None:
            return {"status": "error", "reason": "analyzer_not_initialized"}

        path = analyzer.get_shortest_path(source, target)

        if path is None:
            return {"status": "no_path", "path": []}

        return {
            "status": "success",
            "path": path,
            "length": len(path) - 1
        }

    except Exception as e:
        logger.error(f"Error finding shortest path: {e}")
        return {"status": "error", "reason": str(e)}


@app.task(
    name="network_analysis.export_network",
    base=NetworkAnalysisTask,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 600},
)
def export_network(output_format: str = "json") -> dict:
    """
    Export network for external visualization.

    Args:
        output_format: Format to export ('json', 'graphml', 'gexf')

    Returns:
        Dictionary with exported network data
    """
    logger.info(f"Exporting network in {output_format} format")

    try:
        analyzer = NetworkAnalysisTask._analyzer

        if analyzer is None:
            return {"status": "error", "reason": "analyzer_not_initialized"}

        network_data = analyzer.generate_network_export(output_format)

        if network_data is None:
            return {"status": "error", "reason": "export_failed"}

        return {
            "status": "success",
            "format": output_format,
            "data": network_data
        }

    except Exception as e:
        logger.error(f"Error exporting network: {e}")
        return {"status": "error", "reason": str(e)}
