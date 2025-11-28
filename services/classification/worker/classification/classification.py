from typing import Any, Dict, List
from collections import Counter

from celery.utils.log import get_task_logger
from transformers import TextClassificationPipeline

from common.database.classification_result import ClassificationResult
from common.database.message import ClassificationResultMessage
from common.utils import naive_utcnow

logger = get_task_logger(__name__)


def create_failed_classification_results(
    documents: List[Dict[str, Any]],  # Use Any since pandas might convert enums
) -> List[ClassificationResultMessage]:
    """
    Create ClassificationResultMessage objects for unclassifiable documents.

    Args:
        documents: List of document dicts with 'id' and 'error' fields

    Returns:
        List of ClassificationResultMessage objects for failed classifications
    """
    logger.info(
        "Creating classification results for %d unclassifiable documents",
        len(documents),
    )

    results = []
    for doc in documents:
        classification_result = ClassificationResult(
            classified=False,
            # score_pos=None,
            error=doc.get("error"),
            processed_at=naive_utcnow(),
        )

        result_message = ClassificationResultMessage(
            message_id=doc["id"], classification_result=classification_result
        )
        results.append(result_message)

    return results


def classify_batch(
    documents: List[Dict[str, str]], classifier: TextClassificationPipeline
) -> List[ClassificationResultMessage]:
    """
    Predict labels and scores for a batch of documents.

    Args:
        documents: List of dicts with 'id' and 'text' keys
        classifier: The Hugging Face classification pipeline

    Returns:
        List of ClassificationResultMessage objects with classification results
    """
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict) or "id" not in doc or "text" not in doc:
            raise ValueError(f"Document at index {i} missing 'id' or 'text' keys")

    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    try:
        classifier_results = classifier(texts)
    except Exception as e:
        logger.error("Error occurred while classifying batch: %s", str(e))
        raise

    if not isinstance(classifier_results, list):
        raise ValueError("Classifier did not return a valid list of results.")

    # Ensure all items in the results are dictionaries with expected keys
    if not all(
        isinstance(res, dict) and "label" in res and "score" in res
        for res in classifier_results
    ):
        raise ValueError("Unexpected format in classifier results.")

    results = []
    for doc_id, classifier_result in zip(ids, classifier_results):
        # Get normalized positive score
        score_pos = (
            classifier_result["score"]
            if classifier_result["label"] == "LABEL_1"
            else 1.0 - classifier_result["score"]
        )

        classification_result = ClassificationResult(
            classified=True,
            score_pos=score_pos,
            #   source_field="",  # TODO: Can't determine source field. We'd need to pass it earlier
            # error=None,
            processed_at=naive_utcnow(),
        )

        result_message = ClassificationResultMessage(
            message_id=doc_id, classification_result=classification_result
        )

        results.append(result_message)

    label_counts = Counter(res.classification_result.score_pos > 0.5 for res in results)
    logger.info(
        "Classification results with threshold 0.5: %s positive, %s negative",
        label_counts.get(True, 0),
        label_counts.get(False, 0),
    )

    return results
