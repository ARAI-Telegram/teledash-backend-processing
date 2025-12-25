"""
Topic Modeling Implementation

This module contains the core topic modeling functionality using BERTopic.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from celery.utils.log import get_task_logger

from common.settings import settings

logger = get_task_logger(__name__)


def fit_topic_model_on_messages(
    messages: List[Dict[str, Any]],
    topic_model: BERTopic
) -> Tuple[BERTopic, List[int], np.ndarray]:
    """
    Fit topic model on message documents.

    Args:
        messages: List of message dictionaries with 'text' field
        topic_model: BERTopic model instance

    Returns:
        Tuple of (fitted_model, topics, probabilities)
    """
    if not messages:
        logger.warning("No messages provided for topic modeling")
        return topic_model, [], np.array([])

    logger.info(f"Fitting topic model on {len(messages)} messages")

    # Extract texts
    texts = [msg["text"] for msg in messages]

    # Fit and transform
    try:
        topics, probs = topic_model.fit_transform(texts)
        logger.info(f"Topic modeling complete. Found {len(set(topics))} topics")
        return topic_model, topics, probs

    except Exception as e:
        logger.error(f"Error during topic modeling: {e}")
        raise


def extract_topic_info(
    topic_model: BERTopic,
    messages: List[Dict[str, Any]],
    topics: List[int]
) -> List[Dict[str, Any]]:
    """
    Extract detailed information about discovered topics.

    Args:
        topic_model: Fitted BERTopic model
        messages: Original message documents
        topics: Topic assignments for each message

    Returns:
        List of topic information dictionaries
    """
    topic_info_list = []

    # Get topic info from BERTopic
    topic_info_df = topic_model.get_topic_info()

    for _, row in topic_info_df.iterrows():
        topic_num = row["Topic"]

        # Skip outlier topic (-1)
        if topic_num == -1:
            continue

        # Get top words for this topic
        topic_words = topic_model.get_topic(topic_num)
        if not topic_words:
            continue

        words, scores = zip(*topic_words[:10])  # Top 10 words

        # Get representative documents
        topic_indices = [i for i, t in enumerate(topics) if t == topic_num]
        representative_docs = []
        if topic_indices:
            # Get up to 5 representative documents
            sample_indices = topic_indices[:min(5, len(topic_indices))]
            representative_docs = [messages[i]["id"] for i in sample_indices if i < len(messages)]

        # Get message timestamps for this topic
        topic_dates = [
            datetime.fromisoformat(messages[i]["date"].replace('Z', '+00:00'))
            for i in topic_indices
            if i < len(messages) and messages[i].get("date")
        ]

        first_seen = min(topic_dates).isoformat() if topic_dates else None
        last_seen = max(topic_dates).isoformat() if topic_dates else None

        # Get primary channels for this topic
        topic_channels = [
            messages[i].get("chat_id")
            for i in topic_indices
            if i < len(messages) and messages[i].get("chat_id")
        ]
        channel_counts = pd.Series(topic_channels).value_counts()
        primary_channels = channel_counts.head(5).index.tolist()

        # Create topic dictionary
        topic_dict = {
            "topic_id": f"topic_{topic_num:04d}",
            "topic_number": topic_num,
            "top_words": list(words),
            "top_words_scores": list(scores),
            "representative_docs": representative_docs,
            "size": len(topic_indices),
            "coherence_score": None,  # TODO: Calculate coherence if needed
            "first_seen": first_seen,
            "last_seen": last_seen,
            "metadata": {
                "primary_channels": primary_channels,
                "language": "multilingual",
            }
        }

        # Try to detect extremism category from keywords
        topic_dict["metadata"]["extremism_category"] = detect_extremism_category(words)

        topic_info_list.append(topic_dict)

    logger.info(f"Extracted information for {len(topic_info_list)} topics")
    return topic_info_list


def detect_extremism_category(words: List[str]) -> Optional[str]:
    """
    Detect potential extremism category based on topic keywords.

    Args:
        words: List of top words in the topic

    Returns:
        Extremism category or None
    """
    # Convert words to lowercase for matching
    words_lower = [w.lower() for w in words]

    # Simple keyword matching (can be enhanced)
    right_wing_keywords = [
        'remigration', 'überfremdung', 'volkstod', 'umvolkung',
        'ausländer', 'migration', 'afd', 'patriot'
    ]
    left_wing_keywords = [
        'antifa', 'antikapitalismus', 'revolutionär', 'autonome'
    ]
    islamic_keywords = [
        'dschihad', 'scharia', 'kalifat', 'islam', 'muslim'
    ]
    conspiracy_keywords = [
        'verschwörung', 'lügenpresse', 'deep state', 'nwo',
        'corona', 'impfung', 'great reset'
    ]

    # Check for matches
    if any(kw in words_lower for kw in right_wing_keywords):
        return "right-wing"
    elif any(kw in words_lower for kw in left_wing_keywords):
        return "left-wing"
    elif any(kw in words_lower for kw in islamic_keywords):
        return "islamic"
    elif any(kw in words_lower for kw in conspiracy_keywords):
        return "conspiracy"

    return None


def assign_topics_to_messages(
    messages: List[Dict[str, Any]],
    topics: List[int],
    probs: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Create message-topic assignments with probabilities.

    Args:
        messages: Original message documents
        topics: Topic assignments
        probs: Topic probabilities

    Returns:
        List of dicts with message_id and topic assignments
    """
    message_topics = []

    for i, (msg, topic, prob) in enumerate(zip(messages, topics, probs)):
        # Skip outlier topics
        if topic == -1:
            continue

        # Get top topic probabilities if available
        if len(prob.shape) > 0 and prob.shape[0] > 1:
            # Multiple probabilities available
            top_indices = np.argsort(prob)[-3:][::-1]  # Top 3
            topic_assignments = [
                {
                    "topic_id": f"topic_{topics[idx]:04d}",
                    "probability": float(prob[idx])
                }
                for idx in top_indices
                if prob[idx] > 0.1  # Only include if probability > 10%
            ]
        else:
            # Single probability
            topic_assignments = [{
                "topic_id": f"topic_{topic:04d}",
                "probability": 1.0
            }]

        message_topics.append({
            "message_id": msg["id"],
            "chat_id": msg.get("chat_id"),
            "topics": topic_assignments
        })

    logger.info(f"Created topic assignments for {len(message_topics)} messages")
    return message_topics


def track_topics_over_time(
    topic_model: BERTopic,
    messages: List[Dict[str, Any]],
    topics: List[int],
    granularity: str = "week"
) -> pd.DataFrame:
    """
    Track how topics evolve over time.

    Args:
        topic_model: Fitted BERTopic model
        messages: Message documents with timestamps
        topics: Topic assignments
        granularity: Time granularity ('day', 'week', 'month')

    Returns:
        DataFrame with topic evolution over time
    """
    # Extract timestamps
    timestamps = [
        datetime.fromisoformat(msg["date"].replace('Z', '+00:00'))
        for msg in messages
        if msg.get("date")
    ]

    if not timestamps or len(timestamps) != len(topics):
        logger.warning("Insufficient timestamp data for temporal analysis")
        return pd.DataFrame()

    try:
        # Use BERTopic's built-in temporal analysis
        topics_over_time = topic_model.topics_over_time(
            docs=[msg["text"] for msg in messages],
            timestamps=timestamps,
            evolution_tuning=True,
            global_tuning=True
        )

        logger.info(f"Generated temporal topic analysis with {len(topics_over_time)} time points")
        return topics_over_time

    except Exception as e:
        logger.error(f"Error in temporal topic analysis: {e}")
        return pd.DataFrame()


def detect_topic_coordination(
    topic_distributions: Dict[str, List[float]],
    threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    Detect potential coordination between channels based on topic similarity.

    Args:
        topic_distributions: Dict mapping channel_id to topic distribution vector
        threshold: Minimum similarity to consider coordinated

    Returns:
        List of (channel1, channel2, similarity) tuples
    """
    from scipy.spatial.distance import cosine

    coordinated_pairs = []

    channel_ids = list(topic_distributions.keys())

    for i, channel1 in enumerate(channel_ids):
        for channel2 in channel_ids[i+1:]:
            dist1 = topic_distributions[channel1]
            dist2 = topic_distributions[channel2]

            # Calculate cosine similarity
            similarity = 1 - cosine(dist1, dist2)

            if similarity >= threshold:
                coordinated_pairs.append((channel1, channel2, similarity))

    logger.info(f"Found {len(coordinated_pairs)} potentially coordinated channel pairs")
    return coordinated_pairs
