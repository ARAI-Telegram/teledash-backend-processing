"""Unit tests for database module.

These tests validate database operations including:
- Creating update actions
- Removing classification results

Note: Complex Elasticsearch query tests are better suited for integration tests.

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/test_database.py -v"
"""

from unittest.mock import Mock, patch

from common.database.classification_result import ClassificationResult, ErrorType
from common.database.index_alias import IndexAlias
from common.database.message import ClassificationResultMessage
from common.settings import settings


class TestCreateUpdateActions:
    """Test the create_update_actions_per_chat method."""

    def test_create_update_actions_single_message(self):
        """Test creating update action for single message."""
        from worker.database import Database
        from common.utils import naive_utcnow

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            with patch('worker.database.build_index_name', return_value="messages_test_chat"):
                classification_result = ClassificationResult(
                    classified=True,
                    score_pos=0.85,
                    processed_at=naive_utcnow()
                )

                result_message = ClassificationResultMessage(
                    message_id="msg_1",
                    classification_result=classification_result
                )

                actions = db.create_update_actions_per_chat("test_chat", [result_message])

                assert len(actions) == 1
                assert actions[0]["_op_type"] == "update"
                assert actions[0]["_index"] == "messages_test_chat"
                assert actions[0]["_id"] == "msg_1"
                assert "classification" in actions[0]["doc"]
                assert actions[0]["doc_as_upsert"] is False

    def test_create_update_actions_multiple_messages(self):
        """Test creating update actions for multiple messages."""
        from worker.database import Database
        from common.utils import naive_utcnow

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            with patch('worker.database.build_index_name', return_value="messages_test_chat"):
                messages = []
                for i in range(3):
                    classification_result = ClassificationResult(
                        classified=True,
                        score_pos=0.8 + i * 0.05,
                        processed_at=naive_utcnow()
                    )
                    result_message = ClassificationResultMessage(
                        message_id=f"msg_{i}",
                        classification_result=classification_result
                    )
                    messages.append(result_message)

                actions = db.create_update_actions_per_chat("test_chat", messages)

                assert len(actions) == 3
                assert all(action["_op_type"] == "update" for action in actions)
                assert all(action["_index"] == "messages_test_chat" for action in actions)
                assert [action["_id"] for action in actions] == ["msg_0", "msg_1", "msg_2"]

    def test_create_update_actions_includes_classification_data(self):
        """Test that update actions include classification data."""
        from worker.database import Database
        from common.utils import naive_utcnow

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            with patch('worker.database.build_index_name', return_value="messages_test_chat"):
                classification_result = ClassificationResult(
                    classified=True,
                    score_pos=0.92,
                    processed_at=naive_utcnow()
                )

                result_message = ClassificationResultMessage(
                    message_id="msg_1",
                    classification_result=classification_result
                )

                actions = db.create_update_actions_per_chat("test_chat", [result_message])

                doc = actions[0]["doc"]["classification"]
                assert doc["classified"] is True
                assert doc["score_pos"] == 0.92
                assert doc["processed_at"] is not None

    def test_create_update_actions_failed_classification(self):
        """Test creating update action for failed classification."""
        from worker.database import Database
        from common.utils import naive_utcnow

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            with patch('worker.database.build_index_name', return_value="messages_test_chat"):
                classification_result = ClassificationResult(
                    classified=False,
                    error=ErrorType.TEXT_TOO_SHORT,
                    processed_at=naive_utcnow()
                )

                result_message = ClassificationResultMessage(
                    message_id="msg_1",
                    classification_result=classification_result
                )

                actions = db.create_update_actions_per_chat("test_chat", [result_message])

                doc = actions[0]["doc"]["classification"]
                assert doc["classified"] is False
                assert doc["error"] == "Text too short"
                assert doc["score_pos"] is None

    def test_create_update_actions_empty_list(self):
        """Test creating actions with empty message list."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            actions = db.create_update_actions_per_chat("test_chat", [])

            assert actions == []


class TestRemoveClassificationResults:
    """Test the remove_classification_results method."""

    def test_remove_classification_results(self):
        """Test removing all classification results."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            mock_client.indices.refresh = Mock()

            # Mock the client property
            with patch.object(type(db), 'client', new_callable=lambda: property(lambda self: mock_client)):
                # Mock UpdateByQuery
                with patch('worker.database.UpdateByQuery') as mock_ubq_class:
                    mock_ubq = Mock()
                    mock_ubq.query.return_value = mock_ubq
                    mock_ubq.script.return_value = mock_ubq
                    mock_ubq.execute.return_value = None
                    mock_ubq_class.return_value = mock_ubq

                    db.remove_classification_results()

                    # Verify UpdateByQuery was configured correctly
                    mock_ubq.query.assert_called_once_with("exists", field="classification")
                    mock_ubq.script.assert_called_once()
                    mock_ubq.execute.assert_called_once()

                    # Verify index refresh was called
                    mock_client.indices.refresh.assert_called_once_with(
                        index=IndexAlias.MESSAGE_INDEX_ALIAS.value
                    )

    def test_remove_classification_results_script_content(self):
        """Test that removal script is correct."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            mock_client.indices.refresh = Mock()

            # Mock the client property
            with patch.object(type(db), 'client', new_callable=lambda: property(lambda self: mock_client)):
                with patch('worker.database.UpdateByQuery') as mock_ubq_class:
                    mock_ubq = Mock()
                    mock_ubq.query.return_value = mock_ubq
                    mock_ubq.script.return_value = mock_ubq
                    mock_ubq.execute.return_value = None
                    mock_ubq_class.return_value = mock_ubq

                    db.remove_classification_results()

                    # Verify script removes classification field
                    script_call = mock_ubq.script.call_args
                    assert script_call[1]["source"] == "ctx._source.remove('classification');"
                    assert script_call[1]["lang"] == "painless"
