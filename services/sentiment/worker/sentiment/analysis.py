"""
Sentiment analysis logic using Hugging Face transformers.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def analyze_sentiment_batch(texts: List[str], model) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of texts.

    Args:
        texts: List of text strings to analyze
        model: Hugging Face sentiment analysis pipeline

    Returns:
        List of sentiment results with labels and scores
    """
    try:
        # Run batch inference
        results = model(texts, batch_size=8, truncation=True)

        # Transform results to our format
        sentiment_results = []
        for result in results:
            label = result['label'].lower()
            score = result['score']

            # Map model labels to our schema (positive, neutral, negative)
            # oliverguhr/german-sentiment-bert outputs: positive, neutral, negative
            sentiment_results.append({
                'sentiment': label,
                'confidence': score,
                'scores': {
                    'positive': score if label == 'positive' else (1 - score) / 2,
                    'neutral': score if label == 'neutral' else (1 - score) / 2,
                    'negative': score if label == 'negative' else (1 - score) / 2,
                }
            })

        return sentiment_results

    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
        raise


def create_sentiment_result_document(
    message_id: str,
    chat_id: str,
    date: datetime,
    text_snippet: str,
    language: str,
    sentiment: str,
    confidence: float,
    scores: Dict[str, float],
    model_version: str = "oliverguhr/german-sentiment-bert"
) -> Dict[str, Any]:
    """
    Create a sentiment result document for Elasticsearch.

    Args:
        message_id: Message ID
        chat_id: Chat ID
        date: Message date
        text_snippet: Preview of the text (max 200 chars)
        language: Detected language
        sentiment: Sentiment label (positive, neutral, negative)
        confidence: Confidence score
        scores: Detailed scores for each sentiment
        model_version: Model version used

    Returns:
        Dictionary ready for Elasticsearch indexing
    """
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "date": date.isoformat() if isinstance(date, datetime) else date,
        "sentiment": sentiment,
        "confidence": confidence,
        "scores": scores,
        "language": language,
        "text_snippet": text_snippet[:200] if text_snippet else None,
        "analyzed_at": datetime.utcnow().isoformat(),
        "model_version": model_version
    }
