"""
Network Analysis Implementation

This module builds and analyzes network graphs from Telegram channel relationships.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


class TelegramNetworkAnalyzer:
    """
    Analyze network structure of Telegram channels and users.
    """

    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for forwards/mentions
        logger.info("Initialized Telegram Network Analyzer")

    def build_channel_network(self, messages: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build network graph from message data.

        Nodes: Channels/Users
        Edges:
            - Forward relationships (A forwarded from B)
            - Mention relationships (A mentioned B)
            - Cross-posting (same content in multiple channels)

        Args:
            messages: List of message dictionaries

        Returns:
            NetworkX directed graph
        """
        logger.info(f"Building network from {len(messages)} messages")

        for msg in messages:
            # Add channel as node
            channel_id = msg.get('chat_id')
            if not channel_id:
                continue

            # Add or update channel node
            if channel_id not in self.graph:
                self.graph.add_node(
                    channel_id,
                    title=msg.get('chat_title', 'Unknown'),
                    type='channel',
                    message_count=0
                )

            # Increment message count
            self.graph.nodes[channel_id]['message_count'] += 1

            # Add forward edge if message is forwarded
            forward_info = msg.get('forward')
            if forward_info:
                forward_from_id = forward_info.get('chat_id')
                forward_from_title = forward_info.get('chat_title')

                if forward_from_id and forward_from_id != channel_id:
                    # Add source channel if not exists
                    if forward_from_id not in self.graph:
                        self.graph.add_node(
                            forward_from_id,
                            title=forward_from_title or 'Unknown',
                            type='channel',
                            message_count=0
                        )

                    # Add or update forward edge
                    if self.graph.has_edge(forward_from_id, channel_id):
                        self.graph[forward_from_id][channel_id]['weight'] += 1
                        self.graph[forward_from_id][channel_id]['forward_count'] += 1
                    else:
                        self.graph.add_edge(
                            forward_from_id,
                            channel_id,
                            type='forward',
                            weight=1,
                            forward_count=1,
                            mention_count=0
                        )

            # Add mention edges
            entities = msg.get('entities', [])
            for entity in entities:
                if entity.get('type') in ['mention', 'text_mention']:
                    mentioned_username = entity.get('username')
                    mentioned_user_id = entity.get('user_id')

                    # Use user_id if available, otherwise use username
                    mentioned_id = mentioned_user_id or mentioned_username

                    if mentioned_id and mentioned_id != channel_id:
                        # Add mentioned node if not exists
                        if mentioned_id not in self.graph:
                            self.graph.add_node(
                                mentioned_id,
                                title=mentioned_username or str(mentioned_user_id),
                                type='user' if mentioned_user_id else 'channel',
                                message_count=0
                            )

                        # Add or update mention edge
                        if self.graph.has_edge(channel_id, mentioned_id):
                            self.graph[channel_id][mentioned_id]['weight'] += 1
                            self.graph[channel_id][mentioned_id]['mention_count'] += 1
                        else:
                            self.graph.add_edge(
                                channel_id,
                                mentioned_id,
                                type='mention',
                                weight=1,
                                forward_count=0,
                                mention_count=1
                            )

        logger.info(
            f"Network built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        return self.graph

    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate influence metrics for nodes.

        Returns:
            Dictionary with centrality metrics for each node
        """
        logger.info("Calculating centrality metrics")

        try:
            metrics = {
                'betweenness': nx.betweenness_centrality(self.graph),
                'in_degree': dict(self.graph.in_degree(weight='weight')),
                'out_degree': dict(self.graph.out_degree(weight='weight')),
                'pagerank': nx.pagerank(self.graph, weight='weight')
            }

            # Try eigenvector centrality (may fail for some graphs)
            try:
                metrics['eigenvector'] = nx.eigenvector_centrality(
                    self.graph,
                    max_iter=1000,
                    weight='weight'
                )
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality did not converge, using PageRank instead")
                metrics['eigenvector'] = metrics['pagerank']

            logger.info("Centrality metrics calculated successfully")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating centrality metrics: {e}")
            return {}

    def detect_communities(self) -> List[Set[str]]:
        """
        Detect communities/clusters in the network.

        Returns:
            List of communities (sets of channel IDs)
        """
        logger.info("Detecting communities using Louvain algorithm")

        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()

            # Use Louvain algorithm
            from networkx.algorithms import community
            communities = community.louvain_communities(
                undirected,
                weight='weight',
                seed=42
            )

            logger.info(f"Detected {len(communities)} communities")
            return communities

        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return []

    def find_bridge_channels(self, threshold: float = 0.01) -> List[Tuple[str, float]]:
        """
        Find channels that bridge different communities.

        Args:
            threshold: Minimum betweenness centrality to consider

        Returns:
            List of (channel_id, betweenness_score) tuples
        """
        logger.info(f"Finding bridge channels with threshold {threshold}")

        try:
            betweenness = nx.betweenness_centrality(self.graph)
            bridges = [
                (node, score)
                for node, score in betweenness.items()
                if score > threshold
            ]

            # Sort by score descending
            bridges.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Found {len(bridges)} bridge channels")
            return bridges

        except Exception as e:
            logger.error(f"Error finding bridge channels: {e}")
            return []

    def detect_coordinated_behavior(
        self,
        messages: List[Dict[str, Any]],
        time_window: int = 3600
    ) -> List[Dict[str, Any]]:
        """
        Detect potential coordination between channels.

        Looks for:
        - Multiple channels posting identical/similar content within time window
        - Synchronized forwarding patterns

        Args:
            messages: List of message dictionaries with timestamps
            time_window: Seconds within which to consider messages coordinated

        Returns:
            List of coordination events
        """
        logger.info(f"Detecting coordinated behavior with time window {time_window}s")

        coordination_events = []

        # Group messages by content hash
        content_groups = {}
        for msg in messages:
            text = msg.get('text', '').strip()
            if not text or len(text) < 50:  # Skip very short messages
                continue

            # Simple content hash (could be improved with text similarity)
            content_hash = hash(text)
            chat_id = msg.get('chat_id')
            timestamp = msg.get('date')

            if content_hash not in content_groups:
                content_groups[content_hash] = []

            content_groups[content_hash].append({
                'chat_id': chat_id,
                'message_id': msg.get('id'),
                'timestamp': timestamp,
                'text': text[:200]  # Keep first 200 chars for reference
            })

        # Analyze groups for coordination
        for content_hash, posts in content_groups.items():
            if len(posts) < 2:
                continue

            # Sort by timestamp
            posts.sort(key=lambda x: x['timestamp'])

            # Check if posts are within time window
            first_time = datetime.fromisoformat(posts[0]['timestamp'].replace('Z', '+00:00'))
            last_time = datetime.fromisoformat(posts[-1]['timestamp'].replace('Z', '+00:00'))
            time_diff = (last_time - first_time).total_seconds()

            if time_diff <= time_window:
                # Get unique channels
                channels = list(set(p['chat_id'] for p in posts))

                coordination_events.append({
                    'type': 'identical_content',
                    'channels': channels,
                    'num_channels': len(channels),
                    'num_posts': len(posts),
                    'time_window_seconds': time_diff,
                    'first_post_time': posts[0]['timestamp'],
                    'last_post_time': posts[-1]['timestamp'],
                    'sample_text': posts[0]['text'],
                    'message_ids': [p['message_id'] for p in posts]
                })

        logger.info(f"Found {len(coordination_events)} coordination events")
        return coordination_events

    def generate_network_export(self, output_format: str = 'json') -> Any:
        """
        Export network for visualization or analysis.

        Args:
            output_format: Format to export ('json', 'graphml', 'gexf')

        Returns:
            Network data in specified format
        """
        logger.info(f"Exporting network in {output_format} format")

        try:
            if output_format == 'json':
                from networkx.readwrite import json_graph
                data = json_graph.node_link_data(self.graph)
                return data

            elif output_format == 'graphml':
                # Return as string
                import io
                buffer = io.StringIO()
                nx.write_graphml(self.graph, buffer)
                return buffer.getvalue()

            elif output_format == 'gexf':
                import io
                buffer = io.StringIO()
                nx.write_gexf(self.graph, buffer)
                return buffer.getvalue()

            else:
                raise ValueError(f"Unsupported format: {output_format}")

        except Exception as e:
            logger.error(f"Error exporting network: {e}")
            return None

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node."""
        if node_id not in self.graph:
            return None

        node_data = self.graph.nodes[node_id]
        in_edges = list(self.graph.in_edges(node_id, data=True))
        out_edges = list(self.graph.out_edges(node_id, data=True))

        return {
            'node_id': node_id,
            'title': node_data.get('title'),
            'type': node_data.get('type'),
            'message_count': node_data.get('message_count', 0),
            'in_degree': len(in_edges),
            'out_degree': len(out_edges),
            'total_in_weight': sum(edge[2].get('weight', 0) for edge in in_edges),
            'total_out_weight': sum(edge[2].get('weight', 0) for edge in out_edges),
            'forward_sources': [
                {'from': edge[0], 'weight': edge[2].get('forward_count', 0)}
                for edge in in_edges
                if edge[2].get('type') == 'forward'
            ],
            'mentions_to': [
                {'to': edge[1], 'weight': edge[2].get('mention_count', 0)}
                for edge in out_edges
                if edge[2].get('type') == 'mention'
            ]
        }

    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        try:
            if source not in self.graph or target not in self.graph:
                return None

            path = nx.shortest_path(self.graph, source=source, target=target)
            return path

        except nx.NetworkXNoPath:
            logger.info(f"No path exists between {source} and {target}")
            return None

        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            return None


def calculate_network_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate overall network statistics.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary with network statistics
    """
    logger.info("Calculating network statistics")

    try:
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph),
        }

        # Calculate additional metrics if graph is not empty
        if stats['num_nodes'] > 0:
            # Average degree
            degrees = [d for n, d in graph.degree(weight='weight')]
            stats['avg_degree'] = np.mean(degrees) if degrees else 0
            stats['max_degree'] = np.max(degrees) if degrees else 0

            # Try to calculate diameter (may be expensive for large graphs)
            try:
                if nx.is_weakly_connected(graph):
                    stats['diameter'] = nx.diameter(graph.to_undirected())
                else:
                    # Get largest connected component
                    largest_cc = max(
                        nx.weakly_connected_components(graph),
                        key=len
                    )
                    subgraph = graph.subgraph(largest_cc)
                    stats['diameter'] = nx.diameter(subgraph.to_undirected())
                    stats['largest_component_size'] = len(largest_cc)
            except:
                stats['diameter'] = None

        logger.info("Network statistics calculated")
        return stats

    except Exception as e:
        logger.error(f"Error calculating network statistics: {e}")
        return {}
