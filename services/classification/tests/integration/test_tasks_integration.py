"""Integration tests for classification tasks.

These tests validate the full task execution with real components.

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/integration/test_tasks_integration.py -v"
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from common.settings import settings


@pytest.mark.skipif(
    settings.gpu_use,
    reason="GPU_USE=true, CPU tests only run when GPU_USE=false to avoid model conflicts"
)
class TestClassifyDocumentsIntegration:
    """Integration tests for classify_documents task with real model."""

    @pytest.fixture(scope="class")
    def cpu_model(self):
        """Load classification model on CPU for testing."""
        from worker.model_init import init_classification_model

        original_gpu = settings.gpu_use
        settings.gpu_use = False

        try:
            model = init_classification_model()
            yield model
        finally:
            settings.gpu_use = original_gpu

            if model is not None:
                del model

            import gc
            gc.collect()

    def test_classify_documents_with_real_model(self, cpu_model):
        """Test classify_documents task with real model and mocked database."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        from worker.classification.tasks import classify_documents, ClassificationTask

        # Set up the model
        ClassificationTask._model = cpu_model

        # Mock database
        with patch('worker.classification.tasks.Database') as mock_db_class:
            mock_db = Mock()
            mock_db.create_update_actions_per_chat.return_value = []
            mock_db.bulk_write.return_value = None
            mock_db_class.return_value = mock_db

            # Prepare document batch with valid German text
            document_batch = {
                "chat_id": "test_chat",
                "documents": [
                    {
                        "id": "msg_1",
                        "text": "Die Regierung plant geheime Maßnahmen gegen die Bevölkerung und versteckt die Wahrheit."
                    },
                    {
                        "id": "msg_2",
                        "text": "Heute ist ein schöner Tag. Die Sonne scheint und ich freue mich."
                    },
                    {
                        "id": "msg_3",
                        "text": "Test"  # Too short
                    }
                ]
            }

            # Execute the task directly
            classify_documents(document_batch)

            # Verify database interactions
            mock_db.create_update_actions_per_chat.assert_called_once()
            call_args = mock_db.create_update_actions_per_chat.call_args[0]

            # Should be called with chat_id
            assert call_args[0] == "test_chat"

            # Should have results for all 3 documents
            results = call_args[1]
            assert len(results) == 3

            # First two should be classified successfully
            classified_results = [r for r in results if r.classification_result.classified]
            assert len(classified_results) >= 2

            # At least one should be failed (the short text)
            failed_results = [r for r in results if not r.classification_result.classified]
            assert len(failed_results) >= 1

            # Verify bulk_write was called
            mock_db.bulk_write.assert_called_once()

    def test_classify_documents_handles_empty_after_preprocessing(self, cpu_model):
        """Test that documents empty after preprocessing are handled correctly."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        from worker.classification.tasks import classify_documents, ClassificationTask

        ClassificationTask._model = cpu_model

        with patch('worker.classification.tasks.Database') as mock_db_class:
            mock_db = Mock()
            mock_db.create_update_actions_per_chat.return_value = []
            mock_db.bulk_write.return_value = None
            mock_db_class.return_value = mock_db

            # Documents that will be empty after preprocessing
            document_batch = {
                "chat_id": "test_chat",
                "documents": [
                    {"id": "msg_1", "text": "@user1 @user2 @user3"},
                    {"id": "msg_2", "text": "https://example.com"},
                    {"id": "msg_3", "text": "   "},
                ]
            }

            classify_documents(document_batch)

            # All should result in failed classifications
            call_args = mock_db.create_update_actions_per_chat.call_args[0]
            results = call_args[1]

            assert len(results) == 3
            assert all(not r.classification_result.classified for r in results)

    def test_classify_documents_with_mixed_content(self, cpu_model):
        """Test classify_documents with mix of valid, short, and empty texts."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        from worker.classification.tasks import classify_documents, ClassificationTask

        ClassificationTask._model = cpu_model

        with patch('worker.classification.tasks.Database') as mock_db_class:
            mock_db = Mock()
            mock_db.create_update_actions_per_chat.return_value = []
            mock_db.bulk_write.return_value = None
            mock_db_class.return_value = mock_db

            document_batch = {
                "chat_id": "test_chat",
                "documents": [
                    # Valid long text
                    {
                        "id": "msg_1",
                        "text": "Dies ist ein ausreichend langer Text der klassifiziert werden kann und genug Inhalt hat."
                    },
                    # Short text
                    {"id": "msg_2", "text": "Hi"},
                    # Text with noise that becomes empty
                    {"id": "msg_3", "text": "@user https://example.com"},
                    # Another valid text
                    {
                        "id": "msg_4",
                        "text": "Wissenschaftler haben eine neue Entdeckung gemacht die unser Verständnis verändert."
                    },
                ]
            }

            classify_documents(document_batch)

            call_args = mock_db.create_update_actions_per_chat.call_args[0]
            results = call_args[1]

            # Should have 4 results
            assert len(results) == 4

            # Should have both successful and failed classifications
            classified = [r for r in results if r.classification_result.classified]
            failed = [r for r in results if not r.classification_result.classified]

            assert len(classified) > 0
            assert len(failed) > 0

    def test_classify_documents_preserves_message_ids(self, cpu_model):
        """Test that message IDs are correctly preserved through classification."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        from worker.classification.tasks import classify_documents, ClassificationTask

        ClassificationTask._model = cpu_model

        with patch('worker.classification.tasks.Database') as mock_db_class:
            mock_db = Mock()
            mock_db.create_update_actions_per_chat.return_value = []
            mock_db.bulk_write.return_value = None
            mock_db_class.return_value = mock_db

            document_batch = {
                "chat_id": "test_chat",
                "documents": [
                    {"id": "custom_id_alpha", "text": "Text eins für die Klassifizierung"},
                    {"id": "custom_id_beta", "text": "Text zwei für die Klassifizierung"},
                    {"id": "custom_id_gamma", "text": "Text drei für die Klassifizierung"},
                ]
            }

            classify_documents(document_batch)

            call_args = mock_db.create_update_actions_per_chat.call_args[0]
            results = call_args[1]

            result_ids = {r.message_id for r in results}
            expected_ids = {"custom_id_alpha", "custom_id_beta", "custom_id_gamma"}

            assert result_ids == expected_ids



@pytest.mark.skipif(
    not settings.gpu_use,
    reason="GPU_USE=false, GPU tests only run when GPU_USE=true"
)
class TestClassifyDocumentsGPU:
    """Integration tests for classify_documents task on GPU."""

    @pytest.fixture(scope="class")
    def gpu_model(self):
        """Load classification model on GPU for testing."""
        from worker.model_init import init_classification_model

        model = init_classification_model()

        if model is None:
            pytest.skip("GPU model failed to load - check CUDA availability")

        yield model

        if model is not None:
            del model

        import gc
        gc.collect()

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_classify_documents_gpu(self, gpu_model):
        """Test classify_documents with GPU model."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        from worker.classification.tasks import classify_documents, ClassificationTask

        ClassificationTask._model = gpu_model

        with patch('worker.classification.tasks.Database') as mock_db_class:
            mock_db = Mock()
            mock_db.create_update_actions_per_chat.return_value = []
            mock_db.bulk_write.return_value = None
            mock_db_class.return_value = mock_db

            document_batch = {
                "chat_id": "test_chat",
                "documents": [
                    {
                        "id": "msg_1",
                        "text": "Die Regierung verbirgt wichtige Informationen vor der Öffentlichkeit."
                    }
                ]
            }

            classify_documents(document_batch)

            # Verify it executed successfully
            mock_db.create_update_actions_per_chat.assert_called_once()
            mock_db.bulk_write.assert_called_once()
