"""
Build entity co-occurrence network for visualization.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def build_entity_network(
    entities: List[Dict[str, Any]],
    entity_types: List[str] = None,
    min_cooccurrence: int = 2,
    max_nodes: int = 100
) -> Dict[str, Any]:
    """
    Build entity network graph for D3.js visualization.

    Args:
        entities: List of entity documents with co_occurring_entities
        entity_types: Optional filter by entity types
        min_cooccurrence: Minimum co-occurrence count for edges
        max_nodes: Maximum number of nodes to include

    Returns:
        Network graph with nodes and edges
    """
    try:
        # Filter entities by type if specified
        if entity_types:
            entities = [e for e in entities if e.get("type") in entity_types]

        # Sort by frequency and limit to max_nodes
        entities = sorted(entities, key=lambda e: e.get("frequency", 0), reverse=True)[:max_nodes]

        # Build nodes
        nodes = []
        entity_id_set = set()

        for entity in entities:
            entity_id = entity["entity_id"]
            entity_id_set.add(entity_id)

            node = {
                "id": entity_id,
                "name": entity["text"],
                "type": entity["type"],
                "count": entity.get("frequency", 0)
            }
            nodes.append(node)

        # Build edges from co-occurrence data
        edges = []
        seen_edges = set()

        for entity in entities:
            entity_id = entity["entity_id"]
            cooccurrences = entity.get("co_occurring_entities", [])

            for cooc in cooccurrences:
                target_id = cooc["entity_id"]

                # Only include if target is in our node set
                if target_id not in entity_id_set:
                    continue

                # Filter by min_cooccurrence
                if cooc.get("count", 0) < min_cooccurrence:
                    continue

                # Create edge (avoid duplicates)
                edge_key = tuple(sorted([entity_id, target_id]))
                if edge_key in seen_edges:
                    continue

                seen_edges.add(edge_key)

                edge = {
                    "source": entity_id,
                    "target": target_id,
                    "weight": cooc["count"]
                }
                edges.append(edge)

        logger.info(f"Built network with {len(nodes)} nodes and {len(edges)} edges")

        return {
            "nodes": nodes,
            "edges": edges
        }

    except Exception as e:
        logger.error(f"Error building entity network: {e}", exc_info=True)
        raise
