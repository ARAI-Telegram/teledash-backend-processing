"""Unit tests for classification module.

These tests validate classification logic including:
- Batch classification with mocked pipeline
- Failed classification result creation
- Score normalization

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/test_classification.py -v"
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from common.database.classification_result import ErrorType
from common.database.message import ClassificationResultMessage
from worker.classification.classification import (
    classify_batch,
    create_failed_classification_results,
)


class TestCreateFailedClassificationResults:
    """Test the create_failed_classification_results function."""

    def test_create_single_failed_result(self):
        """Test creating a single failed classification result."""
        documents = [
            {"id": "msg_1", "error": ErrorType.TEXT_TOO_SHORT}
        ]

        results = create_failed_classification_results(documents)

        assert len(results) == 1
        assert isinstance(results[0], ClassificationResultMessage)
        assert results[0].message_id == "msg_1"
        assert results[0].classification_result.classified is False
        assert results[0].classification_result.error == ErrorType.TEXT_TOO_SHORT
        assert results[0].classification_result.score_pos is None
        assert results[0].classification_result.processed_at is not None

    def test_create_multiple_failed_results(self):
        """Test creating multiple failed classification results."""
        documents = [
            {"id": "msg_1", "error": ErrorType.TEXT_TOO_SHORT},
            {"id": "msg_2", "error": ErrorType.EMPTY_TEXT},
            {"id": "msg_3", "error": ErrorType.TEXT_TOO_SHORT},
        ]

        results = create_failed_classification_results(documents)

        assert len(results) == 3
        assert all(isinstance(r, ClassificationResultMessage) for r in results)
        assert all(not r.classification_result.classified for r in results)
        assert results[0].classification_result.error == ErrorType.TEXT_TOO_SHORT
        assert results[1].classification_result.error == ErrorType.EMPTY_TEXT
        assert results[2].classification_result.error == ErrorType.TEXT_TOO_SHORT

    def test_failed_result_has_processed_timestamp(self):
        """Test that failed results have processed_at timestamp."""
        documents = [{"id": "msg_1", "error": ErrorType.EMPTY_TEXT}]

        results = create_failed_classification_results(documents)

        assert results[0].classification_result.processed_at is not None
        assert isinstance(results[0].classification_result.processed_at, datetime)

    def test_failed_result_structure(self):
        """Test the structure of failed classification result."""
        documents = [{"id": "msg_1", "error": ErrorType.TEXT_TOO_SHORT}]

        results = create_failed_classification_results(documents)

        result = results[0]
        assert hasattr(result, 'message_id')
        assert hasattr(result, 'classification_result')
        assert hasattr(result.classification_result, 'classified')
        assert hasattr(result.classification_result, 'score_pos')
        assert hasattr(result.classification_result, 'error')
        assert hasattr(result.classification_result, 'processed_at')

    def test_empty_documents_list(self):
        """Test handling of empty documents list."""
        documents = []

        results = create_failed_classification_results(documents)

        assert len(results) == 0
        assert isinstance(results, list)


class TestClassifyBatch:
    """Test the classify_batch function."""

    def test_classify_batch_positive_labels(self):
        """Test classification with LABEL_1 (positive) predictions."""
        documents = [
            {"id": "msg_1", "text": "This is test text one"},
            {"id": "msg_2", "text": "This is test text two"},
        ]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.85},
            {"label": "LABEL_1", "score": 0.92},
        ]

        results = classify_batch(documents, mock_pipeline)

        assert len(results) == 2
        assert all(isinstance(r, ClassificationResultMessage) for r in results)
        assert results[0].classification_result.classified is True
        assert results[0].classification_result.score_pos == 0.85
        assert results[1].classification_result.score_pos == 0.92

    def test_classify_batch_negative_labels(self):
        """Test classification with LABEL_0 (negative) predictions."""
        documents = [
            {"id": "msg_1", "text": "This is test text one"},
            {"id": "msg_2", "text": "This is test text two"},
        ]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_0", "score": 0.75},  # Should convert to 1 - 0.75 = 0.25
            {"label": "LABEL_0", "score": 0.90},  # Should convert to 1 - 0.90 = 0.10
        ]

        results = classify_batch(documents, mock_pipeline)

        assert len(results) == 2
        assert abs(results[0].classification_result.score_pos - 0.25) < 0.001
        assert abs(results[1].classification_result.score_pos - 0.10) < 0.001

    def test_classify_batch_mixed_labels(self):
        """Test classification with mixed LABEL_0 and LABEL_1."""
        documents = [
            {"id": "msg_1", "text": "Text one"},
            {"id": "msg_2", "text": "Text two"},
            {"id": "msg_3", "text": "Text three"},
        ]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.85},  # Positive
            {"label": "LABEL_0", "score": 0.70},  # Negative -> 0.30
            {"label": "LABEL_1", "score": 0.60},  # Positive
        ]

        results = classify_batch(documents, mock_pipeline)

        assert len(results) == 3
        assert abs(results[0].classification_result.score_pos - 0.85) < 0.001
        assert abs(results[1].classification_result.score_pos - 0.30) < 0.001
        assert abs(results[2].classification_result.score_pos - 0.60) < 0.001

    def test_classify_batch_calls_pipeline_correctly(self):
        """Test that the pipeline is called with correct texts."""
        documents = [
            {"id": "msg_1", "text": "First text"},
            {"id": "msg_2", "text": "Second text"},
        ]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.5},
            {"label": "LABEL_1", "score": 0.5},
        ]

        classify_batch(documents, mock_pipeline)

        # Verify pipeline was called with the texts
        mock_pipeline.assert_called_once_with(["First text", "Second text"])

    def test_classify_batch_message_ids_match(self):
        """Test that message IDs are correctly assigned to results."""
        documents = [
            {"id": "msg_alpha", "text": "Text one"},
            {"id": "msg_beta", "text": "Text two"},
            {"id": "msg_gamma", "text": "Text three"},
        ]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.8},
            {"label": "LABEL_1", "score": 0.7},
            {"label": "LABEL_1", "score": 0.9},
        ]

        results = classify_batch(documents, mock_pipeline)

        assert results[0].message_id == "msg_alpha"
        assert results[1].message_id == "msg_beta"
        assert results[2].message_id == "msg_gamma"

    def test_classify_batch_has_timestamp(self):
        """Test that classification results have processed_at timestamp."""
        documents = [{"id": "msg_1", "text": "Test text"}]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "LABEL_1", "score": 0.75}]

        results = classify_batch(documents, mock_pipeline)

        assert results[0].classification_result.processed_at is not None
        assert isinstance(results[0].classification_result.processed_at, datetime)

    def test_classify_batch_missing_id_raises_error(self):
        """Test that missing 'id' key raises ValueError."""
        documents = [
            {"text": "Text without ID"},
        ]

        mock_pipeline = Mock()

        with pytest.raises(ValueError, match="missing 'id' or 'text' keys"):
            classify_batch(documents, mock_pipeline)

    def test_classify_batch_missing_text_raises_error(self):
        """Test that missing 'text' key raises ValueError."""
        documents = [
            {"id": "msg_1"},
        ]

        mock_pipeline = Mock()

        with pytest.raises(ValueError, match="missing 'id' or 'text' keys"):
            classify_batch(documents, mock_pipeline)

    def test_classify_batch_invalid_document_format_raises_error(self):
        """Test that non-dict documents raise ValueError."""
        documents = [
            "not a dictionary",
        ]

        mock_pipeline = Mock()

        with pytest.raises(ValueError, match="missing 'id' or 'text' keys"):
            classify_batch(documents, mock_pipeline)

    def test_classify_batch_pipeline_error_propagates(self):
        """Test that pipeline errors are propagated."""
        documents = [{"id": "msg_1", "text": "Test text"}]

        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Pipeline error")

        with pytest.raises(Exception, match="Pipeline error"):
            classify_batch(documents, mock_pipeline)

    def test_classify_batch_invalid_pipeline_response_raises_error(self):
        """Test that invalid pipeline response raises ValueError."""
        documents = [{"id": "msg_1", "text": "Test text"}]

        mock_pipeline = Mock()
        mock_pipeline.return_value = "not a list"

        with pytest.raises(ValueError, match="did not return a valid list"):
            classify_batch(documents, mock_pipeline)

    def test_classify_batch_malformed_results_raises_error(self):
        """Test that malformed classifier results raise ValueError."""
        documents = [{"id": "msg_1", "text": "Test text"}]

        mock_pipeline = Mock()
        # Missing 'score' key
        mock_pipeline.return_value = [{"label": "LABEL_1"}]

        with pytest.raises(ValueError, match="Unexpected format in classifier results"):
            classify_batch(documents, mock_pipeline)

    def test_classify_batch_single_document(self):
        """Test classification with single document."""
        documents = [{"id": "msg_1", "text": "Single text"}]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "LABEL_1", "score": 0.88}]

        results = classify_batch(documents, mock_pipeline)

        assert len(results) == 1
        assert results[0].message_id == "msg_1"
        assert results[0].classification_result.score_pos == 0.88

    def test_classify_batch_empty_documents_list(self):
        """Test classification with empty documents list."""
        documents = []

        mock_pipeline = Mock()
        mock_pipeline.return_value = []

        results = classify_batch(documents, mock_pipeline)

        assert len(results) == 0
        mock_pipeline.assert_called_once_with([])

    def test_classify_batch_threshold_behavior(self):
        """Test score threshold behavior (0.5 cutoff)."""
        documents = [
            {"id": "msg_1", "text": "Text one"},
            {"id": "msg_2", "text": "Text two"},
            {"id": "msg_3", "text": "Text three"},
        ]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.51},  # Above threshold
            {"label": "LABEL_1", "score": 0.49},  # Below threshold
            {"label": "LABEL_0", "score": 0.51},  # Below threshold (becomes 0.49)
        ]

        with patch('worker.classification.classification.logger'):
            results = classify_batch(documents, mock_pipeline)

        # Check that scores are correctly assigned
        assert results[0].classification_result.score_pos > 0.5
        assert results[1].classification_result.score_pos < 0.5
        assert results[2].classification_result.score_pos < 0.5

    def test_classify_batch_all_classified_true(self):
        """Test that all successful results have classified=True."""
        documents = [
            {"id": "msg_1", "text": "Text one"},
            {"id": "msg_2", "text": "Text two"},
        ]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.8},
            {"label": "LABEL_0", "score": 0.3},
        ]

        results = classify_batch(documents, mock_pipeline)

        assert all(r.classification_result.classified for r in results)

    def test_classify_batch_no_errors_in_results(self):
        """Test that successful classifications have no error field."""
        documents = [{"id": "msg_1", "text": "Test text"}]

        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "LABEL_1", "score": 0.75}]

        results = classify_batch(documents, mock_pipeline)

        # Successful classification should have no error
        # Note: The error field might be None rather than absent
        assert results[0].classification_result.error is None or \
               not hasattr(results[0].classification_result, 'error') or \
               results[0].classification_result.error is None
