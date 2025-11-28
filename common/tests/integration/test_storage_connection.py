"""Integration tests for Storage class with real storage backend.

These tests require a properly configured .env file with valid storage credentials.
They test actual connectivity and operations against a real storage backend.

Run these tests with:
    python3 -m pytest common/tests/integration/ -v
"""

import os
import tempfile
import uuid

import pytest

from common.storage import Storage, StorageBucketNames


class TestStorageConnection:
    """Integration tests for Storage class with real storage backend."""

    @pytest.fixture(scope="class")
    def storage(self):
        """Fixture to provide a connected Storage instance."""
        storage = Storage(connect=True)
        yield storage

    def test_storage_connection(self, storage):
        """Test basic connectivity to the storage backend."""
        assert storage.client is not None
        storage.bucket_exists("nonexistent-bucket")  # Should not raise exception

    def test_file_operations(self, storage):
        """Test file upload, download, and delete operations on existing buckets."""

        # Use an existing bucket that should be available (photos is commonly used)
        bucket_name = StorageBucketNames.photos.value
        test_filename = f"integration-test-{uuid.uuid4().hex[:8]}.txt"
        test_content = b"This is an integration test file"

        # Create temporary local file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            # Upload file to existing bucket
            storage.fput_object(
                bucket_name, test_filename, temp_file_path, "text/plain"
            )

            # Download and verify content
            response = storage.get_object(bucket_name, test_filename)
            downloaded_content = response["Body"].read()
            assert downloaded_content == test_content, (
                "Downloaded content should match uploaded content"
            )
            assert response["ContentType"] == "text/plain", (
                "Content type should be preserved"
            )

            # Delete the file
            storage.remove_object(bucket_name, test_filename)

            # Verify file is gone
            try:
                storage.get_object(bucket_name, test_filename)
                assert False, "File should not exist after deletion"
            except Exception:
                # Expected - file should not exist
                pass

        finally:
            # Cleanup: Make sure test file is removed
            try:
                storage.remove_object(bucket_name, test_filename)
            except Exception:
                pass  # File might already be deleted

            # Clean up local temp file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
