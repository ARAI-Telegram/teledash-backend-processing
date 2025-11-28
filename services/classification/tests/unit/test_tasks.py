"""Unit tests for Celery tasks.

These tests validate task logic including:
- Classification task model caching
- Task initialization and batch creation

Note: Full Celery task execution is better tested in integration tests.

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/test_tasks.py -v"
"""

import pytest
from unittest.mock import Mock, patch

import pandas as pd


class TestClassificationTask:
    """Test the ClassificationTask base class."""

    @patch('worker.classification.tasks.init_classification_model')
    def test_classification_task_loads_model_on_first_call(self, mock_init_model):
        """Test that model is loaded on first task call."""
        from worker.classification.tasks import ClassificationTask

        # Reset class-level model
        ClassificationTask._model = None

        mock_model = Mock()
        mock_init_model.return_value = mock_model

        # Create a task instance
        task = ClassificationTask()
        task.run = Mock(return_value=None)

        # Call the task
        task()

        # Verify model was loaded
        mock_init_model.assert_called_once()
        assert ClassificationTask._model == mock_model

    @patch('worker.classification.tasks.init_classification_model')
    def test_classification_task_reuses_loaded_model(self, mock_init_model):
        """Test that model is not reloaded on subsequent calls."""
        from worker.classification.tasks import ClassificationTask

        # Set up a pre-loaded model
        mock_model = Mock()
        ClassificationTask._model = mock_model

        # Create a task instance
        task = ClassificationTask()
        task.run = Mock(return_value=None)

        # Call the task
        task()

        # Verify model was NOT loaded again
        mock_init_model.assert_not_called()

    def test_classification_task_model_property(self):
        """Test the model property accessor."""
        from worker.classification.tasks import ClassificationTask

        mock_model = Mock()
        ClassificationTask._model = mock_model

        task = ClassificationTask()
        assert task.model == mock_model


class TestInitClassification:
    """Test the init_classification task logic."""

    @patch('worker.classification.tasks.Database')
    @patch('worker.classification.tasks.QueueChecker')
    def test_init_classification_skips_if_queue_not_empty(self, mock_queue_checker_class, mock_database_class):
        """Test that initialization is skipped if queue has pending tasks."""
        from worker.classification.tasks import init_classification

        mock_db = Mock()
        mock_database_class.return_value = mock_db

        mock_queue_checker = Mock()
        mock_queue_checker.is_queue_empty.return_value = False
        mock_queue_checker.get_queue_length.return_value = 5
        mock_queue_checker_class.return_value = mock_queue_checker

        init_classification()

        # Verify queue was checked
        mock_queue_checker.is_queue_empty.assert_called_once_with("classification")

        # Verify no documents were fetched (task exited early)
        mock_db.get_all_indices_by_alias.assert_not_called()

    @patch('worker.classification.tasks.Database')
    @patch('worker.classification.tasks.QueueChecker')
    def test_init_classification_exits_if_no_indices(self, mock_queue_checker_class, mock_database_class):
        """Test that initialization exits if no message indices found."""
        from worker.classification.tasks import init_classification

        mock_db = Mock()
        mock_db.get_all_indices_by_alias.return_value = []
        mock_database_class.return_value = mock_db

        mock_queue_checker = Mock()
        mock_queue_checker.is_queue_empty.return_value = True
        mock_queue_checker_class.return_value = mock_queue_checker

        init_classification()

        # Verify indices were checked
        mock_db.get_all_indices_by_alias.assert_called_once()

        # Verify no documents were fetched
        mock_db.get_unclassified_docs.assert_not_called()

    @patch('worker.classification.tasks.Database')
    @patch('worker.classification.tasks.QueueChecker')
    def test_init_classification_handles_database_error(self, mock_queue_checker_class, mock_database_class):
        """Test that database errors are propagated."""
        from worker.classification.tasks import init_classification

        mock_database_class.side_effect = Exception("Database connection error")

        mock_queue_checker = Mock()
        mock_queue_checker_class.return_value = mock_queue_checker

        with pytest.raises(Exception, match="Database connection error"):
            init_classification()
