"""
Named Entity Recognition extraction logic using spaCy.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


# spaCy entity type mapping to our schema
ENTITY_TYPE_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",  # Geopolitical entity -> Location
    "EVENT": "EVENT",
    "MISC": "MISC",
    "PRODUCT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "MISC",
    "TIME": "MISC",
    "MONEY": "MISC",
    "QUANTITY": "MISC",
    "ORDINAL": "MISC",
    "CARDINAL": "MISC",
}


def extract_entities_batch(
    messages: List[Dict[str, Any]],
    nlp_model
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract named entities from a batch of messages using spaCy.

    Args:
        messages: List of message dictionaries with 'id', 'text', 'chat_id', 'date'
        nlp_model: Loaded spaCy NLP pipeline

    Returns:
        Tuple of (entity_mentions, unique_entities)
        - entity_mentions: List of individual entity mention documents
        - unique_entities: List of aggregated entity documents
    """
    entity_mentions = []
    entity_aggregator = {}  # entity_id -> entity data for deduplication

    try:
        # Batch process texts for efficiency
        texts = [msg.get("text", "") for msg in messages]
        docs = list(nlp_model.pipe(texts, batch_size=32))

        for msg, doc in zip(messages, docs):
            message_id = msg["id"]
            chat_id = msg["chat_id"]
            date = msg.get("date", datetime.utcnow())

            # Extract entities from this document
            for ent in doc.ents:
                # Map spaCy entity type to our schema
                entity_type = ENTITY_TYPE_MAP.get(ent.label_, "MISC")

                # Skip if not a relevant entity type
                if entity_type not in ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "MISC"]:
                    continue

                # Normalize entity text for deduplication
                normalized_text = ent.text.strip().lower()

                # Generate entity ID (hash of normalized text + type)
                entity_id = _generate_entity_id(normalized_text, entity_type)

                # Extract context (surrounding words)
                context_start = max(0, ent.start - 5)
                context_end = min(len(doc), ent.end + 5)
                context = doc[context_start:context_end].text

                # Create entity mention document
                mention = {
                    "message_id": message_id,
                    "chat_id": chat_id,
                    "date": date.isoformat() if isinstance(date, datetime) else date,
                    "entity_id": entity_id,
                    "entity_text": ent.text,
                    "entity_type": entity_type,
                    "start_offset": ent.start_char,
                    "end_offset": ent.end_char,
                    "context": context[:300],  # Limit context length
                    "confidence": 0.95,  # spaCy doesn't provide confidence, use default
                    "analyzed_at": datetime.utcnow().isoformat()
                }
                entity_mentions.append(mention)

                # Aggregate entity data
                if entity_id not in entity_aggregator:
                    entity_aggregator[entity_id] = {
                        "entity_id": entity_id,
                        "text": ent.text,  # Keep first occurrence
                        "normalized_text": normalized_text,
                        "type": entity_type,
                        "frequency": 0,
                        "message_ids": set(),
                        "chat_ids": set(),
                        "first_seen": date,
                        "last_seen": date,
                        "confidence_sum": 0.0
                    }

                # Update aggregated data
                entity_data = entity_aggregator[entity_id]
                entity_data["frequency"] += 1
                entity_data["message_ids"].add(message_id)
                entity_data["chat_ids"].add(chat_id)
                entity_data["confidence_sum"] += 0.95

                # Update date range
                if isinstance(date, datetime):
                    if date < entity_data["first_seen"]:
                        entity_data["first_seen"] = date
                    if date > entity_data["last_seen"]:
                        entity_data["last_seen"] = date

        # Convert aggregated entities to final format
        unique_entities = []
        for entity_id, data in entity_aggregator.items():
            entity = {
                "entity_id": entity_id,
                "text": data["text"],
                "normalized_text": data["normalized_text"],
                "type": data["type"],
                "frequency": data["frequency"],
                "confidence": data["confidence_sum"] / data["frequency"] if data["frequency"] > 0 else 0.0,
                "message_ids": list(data["message_ids"]),
                "chat_ids": list(data["chat_ids"]),
                "first_seen": data["first_seen"].isoformat() if isinstance(data["first_seen"], datetime) else data["first_seen"],
                "last_seen": data["last_seen"].isoformat() if isinstance(data["last_seen"], datetime) else data["last_seen"],
                "co_occurring_entities": []  # Will be computed separately
            }
            unique_entities.append(entity)

        logger.info(f"Extracted {len(entity_mentions)} entity mentions, {len(unique_entities)} unique entities")
        return entity_mentions, unique_entities

    except Exception as e:
        logger.error(f"Error extracting entities: {e}", exc_info=True)
        raise


def _generate_entity_id(normalized_text: str, entity_type: str) -> str:
    """
    Generate a unique entity ID based on normalized text and type.

    Args:
        normalized_text: Normalized entity text (lowercase, stripped)
        entity_type: Entity type (PERSON, ORG, LOC, etc.)

    Returns:
        Unique entity ID (hash string)
    """
    combined = f"{entity_type}:{normalized_text}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def compute_cooccurrences(
    entity_mentions: List[Dict[str, Any]],
    min_cooccurrence: int = 2
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute entity co-occurrence relationships.

    Two entities co-occur if they appear in the same message.

    Args:
        entity_mentions: List of entity mention documents
        min_cooccurrence: Minimum co-occurrence count to include

    Returns:
        Dictionary mapping entity_id -> list of co-occurring entities
    """
    # Group mentions by message
    message_entities = {}
    for mention in entity_mentions:
        message_id = mention["message_id"]
        entity_id = mention["entity_id"]

        if message_id not in message_entities:
            message_entities[message_id] = []
        message_entities[message_id].append({
            "entity_id": entity_id,
            "text": mention["entity_text"],
            "type": mention["entity_type"]
        })

    # Build co-occurrence matrix
    cooccurrence_matrix = {}

    for message_id, entities in message_entities.items():
        # For each pair of entities in the same message
        for i, entity1 in enumerate(entities):
            entity1_id = entity1["entity_id"]

            if entity1_id not in cooccurrence_matrix:
                cooccurrence_matrix[entity1_id] = {}

            for entity2 in entities[i+1:]:
                entity2_id = entity2["entity_id"]

                if entity1_id == entity2_id:
                    continue

                # Increment co-occurrence count
                if entity2_id not in cooccurrence_matrix[entity1_id]:
                    cooccurrence_matrix[entity1_id][entity2_id] = {
                        "entity_id": entity2_id,
                        "entity_text": entity2["text"],
                        "entity_type": entity2["type"],
                        "count": 0
                    }
                cooccurrence_matrix[entity1_id][entity2_id]["count"] += 1

                # Symmetric relationship
                if entity2_id not in cooccurrence_matrix:
                    cooccurrence_matrix[entity2_id] = {}
                if entity1_id not in cooccurrence_matrix[entity2_id]:
                    cooccurrence_matrix[entity2_id][entity1_id] = {
                        "entity_id": entity1_id,
                        "entity_text": entity1["text"],
                        "entity_type": entity1["type"],
                        "count": 0
                    }
                cooccurrence_matrix[entity2_id][entity1_id]["count"] += 1

    # Filter by minimum co-occurrence and convert to final format
    filtered_cooccurrences = {}
    for entity_id, cooccurrences in cooccurrence_matrix.items():
        filtered = [
            cooc for cooc in cooccurrences.values()
            if cooc["count"] >= min_cooccurrence
        ]
        # Sort by count descending
        filtered.sort(key=lambda x: x["count"], reverse=True)
        filtered_cooccurrences[entity_id] = filtered

    return filtered_cooccurrences
