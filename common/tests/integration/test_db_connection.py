"""Integration tests for Database classes with real Elasticsearch backend.

These tests require a properly configured .env file with valid Elasticsearch credentials.
They test actual connectivity and operations against a real Elasticsearch backend.

Run these tests with:
    python3 -m pytest common/tests/integration/test_db_connection.py -v
"""

import pytest

from common.database.database import DatabaseAsync, DatabaseSync
from common.database.index_alias import IndexAlias


class TestDatabaseSyncConnection:
    """Integration tests for DatabaseSync class with real Elasticsearch backend."""

    @pytest.fixture(scope="class")
    def db(self):
        """Fixture to provide a connected DatabaseSync instance."""
        db = DatabaseSync(connect=True)
        yield db

    def test_database_connection(self, db):
        """Test basic connectivity to the Elasticsearch backend."""
        assert db.es_client is not None
        assert db.client is not None

    def test_ping_elasticsearch(self, db):
        """Test that Elasticsearch is reachable."""
        # Ping the cluster
        assert db.client.ping() is True

    def test_cluster_health(self, db):
        """Test cluster health check."""
        health = db.client.cluster.health()
        assert health is not None
        assert "status" in health
        # Status can be green, yellow, or red
        assert health["status"] in ["green", "yellow", "red"]

    def test_get_all_indices_by_alias(self, db):
        """Test retrieving indices by alias."""
        # Try to get indices for MESSAGE_INDEX_ALIAS
        indices = db.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)
        # Should return a list (might be empty if no indices exist)
        assert isinstance(indices, list)

    def test_index_exists(self, db):
        """Test index existence check."""
        # Test with a non-existent index
        exists = db.index_exists("nonexistent-test-index-12345")
        assert exists is False

    def test_index_has_field(self, db):
        """Test checking if index has a specific field."""
        # Get actual indices first
        indices = db.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)

        if indices:
            # Test with first available index
            index_name = indices[0]
            # Most message indices should have a 'date' field
            has_date_field = db.index_has_field(index_name, "date")
            # This might be True or False depending on the index structure
            assert isinstance(has_date_field, bool)

            # Test with non-existent field
            has_fake_field = db.index_has_field(index_name, "nonexistent_field_12345")
            assert has_fake_field is False

    def test_index_empty(self, db):
        """Test checking if index is empty."""
        # Get actual indices first
        indices = db.get_all_indices_by_alias(IndexAlias.MESSAGE_INDEX_ALIAS)

        if indices:
            # Test with first available index
            index_name = indices[0]
            is_empty = db.index_empty(index_name)
            # Should return a boolean
            assert isinstance(is_empty, bool)


class TestDatabaseAsyncConnection:
    """Integration tests for DatabaseAsync class with real Elasticsearch backend."""

    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test basic connectivity to the Elasticsearch backend."""
        db = DatabaseAsync(connect=True)
        try:
            assert db.es_client is not None
            assert db.client is not None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_ping_elasticsearch(self):
        """Test that Elasticsearch is reachable."""
        db = DatabaseAsync(connect=True)
        try:
            # Ping the cluster
            result = await db.client.ping()
            assert result is True
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_cluster_health(self):
        """Test cluster health check."""
        db = DatabaseAsync(connect=True)
        try:
            health = await db.client.cluster.health()
            assert health is not None
            assert "status" in health
            # Status can be green, yellow, or red
            assert health["status"] in ["green", "yellow", "red"]
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test that async connection can be closed properly."""
        db = DatabaseAsync(connect=True)
        assert db.client is not None

        # Close the connection
        await db.close()
        # After closing, the client should still exist but connection is closed
        assert db.es_client is not None
