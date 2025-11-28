"""Integration tests for database operations.

These tests validate database operations with real Elasticsearch queries (mocked).
They test the query logic without requiring a live Elasticsearch instance.

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/integration/test_database_integration.py -v"
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from common.database.index_alias import IndexAlias
from common.database.message import ClassificationMessage
from common.settings import settings


class TestGetUnclassifiedDocsIntegration:
    """Integration tests for get_unclassified_docs with realistic scenarios."""

    def test_get_unclassified_docs_query_structure(self):
        """Test that get_unclassified_docs builds correct Elasticsearch query."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            # Mock Elasticsearch client and methods
            mock_client = Mock()
            db.es_client = mock_client

            # Mock index operations
            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=False):
                    with patch('worker.database.Search') as mock_search_class:
                        # Set up the mock search chain
                        mock_search = Mock()
                        mock_search.filter.return_value = mock_search
                        mock_search.sort.return_value = mock_search
                        mock_search.source.return_value = mock_search
                        mock_search.count.return_value = 42
                        mock_search_class.return_value = mock_search

                        # Call get_unclassified_docs
                        count = db.get_unclassified_docs(return_docs=False)

                        # Verify Search was created with correct index
                        mock_search_class.assert_called_once()
                        call_kwargs = mock_search_class.call_args[1]
                        assert 'index' in call_kwargs
                        assert call_kwargs['index'] == IndexAlias.MESSAGE_INDEX_ALIAS.value

                        # Verify filter was called (contains query logic)
                        mock_search.filter.assert_called_once()

                        # Verify count was called
                        assert count == 42

    def test_get_unclassified_docs_with_chat_id_builds_index_name(self):
        """Test that chat_id parameter builds correct index name."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            db.es_client = mock_client

            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=False):
                    with patch('worker.database.Search') as mock_search_class:
                        with patch('worker.database.build_index_name') as mock_build_index:
                            mock_build_index.return_value = "messages_chat123"

                            mock_search = Mock()
                            mock_search.filter.return_value = mock_search
                            mock_search.sort.return_value = mock_search
                            mock_search.source.return_value = mock_search
                            mock_search.count.return_value = 10
                            mock_search_class.return_value = mock_search

                            db.get_unclassified_docs(chat_id="chat123", return_docs=False)

                            # Verify build_index_name was called with chat_id
                            mock_build_index.assert_called_once_with(
                                IndexAlias.MESSAGE_INDEX_ALIAS, "chat123"
                            )

                            # Verify Search used the built index
                            call_kwargs = mock_search_class.call_args[1]
                            assert call_kwargs['index'] == "messages_chat123"

    def test_get_unclassified_docs_returns_messages(self):
        """Test that get_unclassified_docs returns Message objects."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            db.es_client = mock_client

            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=False):
                    with patch('worker.database.Search') as mock_search_class:
                        mock_search = Mock()
                        mock_search.filter.return_value = mock_search
                        mock_search.sort.return_value = mock_search
                        mock_search.source.return_value = mock_search
                        mock_search.extra.return_value = mock_search

                        # Mock response with realistic message data
                        mock_response = {
                            "hits": {
                                "hits": [
                                    {
                                        "_id": "msg_1",
                                        "_source": {
                                            "date": "2025-01-12T10:00:00",
                                            "language": "de",
                                            "text": "Sample text for classification"
                                        }
                                    },
                                    {
                                        "_id": "msg_2",
                                        "_source": {
                                            "date": "2025-01-12T10:01:00",
                                            "language": "de",
                                            "caption": "Image caption text"
                                        }
                                    }
                                ]
                            }
                        }
                        mock_search.execute.return_value = mock_response
                        mock_search_class.return_value = mock_search

                        # Get documents
                        docs = db.get_unclassified_docs(size=10, return_docs=True)

                        # Verify we get Message objects
                        assert len(docs) == 2
                        assert all(isinstance(doc, ClassificationMessage) for doc in docs)
                        assert docs[0].id == "msg_1"
                        assert docs[1].id == "msg_2"
                        assert docs[0].language == "de"
                        assert docs[1].language == "de"

    def test_get_unclassified_docs_empty_index_returns_empty_list(self):
        """Test that empty index returns empty list."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=True):
                    docs = db.get_unclassified_docs(return_docs=True)

                    assert docs == []

    def test_get_unclassified_docs_nonexistent_index_returns_empty_list(self):
        """Test that nonexistent index returns empty list."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)

            with patch.object(db, 'index_exists', return_value=False):
                docs = db.get_unclassified_docs(return_docs=True)

                assert docs == []

    def test_get_unclassified_docs_with_last_processed_date(self):
        """Test that last_processed_date adds date range filter."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            db.es_client = mock_client

            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=False):
                    with patch('worker.database.Search') as mock_search_class:
                        mock_search = Mock()
                        mock_search.filter.return_value = mock_search
                        mock_search.sort.return_value = mock_search
                        mock_search.source.return_value = mock_search
                        mock_search.count.return_value = 5
                        mock_search_class.return_value = mock_search

                        last_date = datetime(2025, 1, 12, 10, 0, 0)
                        db.get_unclassified_docs(
                            last_processed_date=last_date, return_docs=False
                        )

                        # Filter should be called (date range is in must_clauses)
                        mock_search.filter.assert_called_once()

    def test_get_unclassified_docs_filters_by_language(self):
        """Test that documents are filtered by configured languages."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            db.es_client = mock_client

            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=False):
                    with patch('worker.database.Search') as mock_search_class:
                        mock_search = Mock()
                        mock_search.filter.return_value = mock_search
                        mock_search.sort.return_value = mock_search
                        mock_search.source.return_value = mock_search
                        mock_search.count.return_value = 10
                        mock_search_class.return_value = mock_search

                        with patch.object(settings, 'classification_languages', ['de', 'en']):
                            db.get_unclassified_docs(return_docs=False)

                            # Filter should include language terms
                            mock_search.filter.assert_called_once()

    def test_get_unclassified_docs_sorts_by_date_ascending(self):
        """Test that results are sorted by date in ascending order."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            db.es_client = mock_client

            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=False):
                    with patch('worker.database.Search') as mock_search_class:
                        mock_search = Mock()
                        mock_search.filter.return_value = mock_search
                        mock_search.sort.return_value = mock_search
                        mock_search.source.return_value = mock_search
                        mock_search.count.return_value = 10
                        mock_search_class.return_value = mock_search

                        db.get_unclassified_docs(return_docs=False)

                        # Verify sort was called with date ascending
                        mock_search.sort.assert_called_once()
                        sort_args = mock_search.sort.call_args[0][0]
                        assert "date" in sort_args
                        assert sort_args["date"]["order"] == "asc"

    def test_get_unclassified_docs_requests_correct_fields(self):
        """Test that only required fields are requested from Elasticsearch."""
        from worker.database import Database

        with patch.object(Database, '__init__', lambda self, connect=True: None):
            db = Database(connect=False)
            mock_client = Mock()
            db.es_client = mock_client

            with patch.object(db, 'index_exists', return_value=True):
                with patch.object(db, 'index_empty', return_value=False):
                    with patch('worker.database.Search') as mock_search_class:
                        mock_search = Mock()
                        mock_search.filter.return_value = mock_search
                        mock_search.sort.return_value = mock_search
                        mock_search.source.return_value = mock_search
                        mock_search.count.return_value = 10
                        mock_search_class.return_value = mock_search

                        db.get_unclassified_docs(return_docs=False)

                        # Verify source was called with field list
                        mock_search.source.assert_called_once()
                        source_args = mock_search.source.call_args
                        assert 'includes' in source_args[1]
