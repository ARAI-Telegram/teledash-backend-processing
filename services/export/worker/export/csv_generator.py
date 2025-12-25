"""
CSV export generation logic.
"""

import logging
import csv
import io
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_csv_export(
    data: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    analysis_type: str = "sentiment"
) -> bytes:
    """
    Generate CSV export from analysis data.

    Args:
        data: List of result dictionaries
        fields: Optional specific fields to include
        analysis_type: Type of analysis (sentiment, entities, ngrams, topics)

    Returns:
        CSV data as bytes
    """
    logger.info(f"Generating CSV export for {analysis_type} with {len(data)} rows")

    if not data:
        logger.warning("No data provided for CSV export")
        return b""

    try:
        # Determine fields from first item if not specified
        if not fields:
            fields = _get_default_fields(analysis_type, data[0] if data else {})

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')

        # Write header
        writer.writeheader()

        # Write data rows
        for item in data:
            # Flatten nested structures for CSV
            flattened = _flatten_dict(item)
            writer.writerow(flattened)

        # Get CSV content
        csv_content = output.getvalue()
        output.close()

        logger.info(f"CSV generated successfully: {len(csv_content)} bytes")
        return csv_content.encode('utf-8')

    except Exception as e:
        logger.error(f"Error generating CSV: {e}", exc_info=True)
        raise


def _get_default_fields(analysis_type: str, sample_item: Dict[str, Any]) -> List[str]:
    """Get default fields for CSV export based on analysis type."""

    field_map = {
        "sentiment": [
            "message_id", "chat_id", "date", "sentiment", "confidence",
            "scores.positive", "scores.neutral", "scores.negative",
            "language", "text_snippet", "analyzed_at"
        ],
        "entities": [
            "entity_id", "text", "type", "frequency", "confidence",
            "first_seen", "last_seen", "chat_ids", "message_count"
        ],
        "ngrams": [
            "ngram_id", "text", "n", "frequency", "document_frequency",
            "tf_idf_score", "language", "generated_at"
        ],
        "topics": [
            "topic_id", "topic_number", "top_words", "size",
            "coherence_score", "first_seen", "last_seen",
            "metadata.extremism_category", "metadata.language"
        ]
    }

    default_fields = field_map.get(analysis_type, [])

    # If default fields not found, use all keys from sample item
    if not default_fields and sample_item:
        default_fields = list(sample_item.keys())

    return default_fields


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary for CSV export.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to comma-separated strings
            items.append((new_key, ', '.join(map(str, v)) if v else ''))
        elif isinstance(v, datetime):
            items.append((new_key, v.isoformat()))
        else:
            items.append((new_key, v))

    return dict(items)
