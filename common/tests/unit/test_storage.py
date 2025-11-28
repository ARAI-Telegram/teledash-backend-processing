"""Unit tests for Storage class."""

import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError
from botocore.config import Config

from common.storage import Storage, StorageBucketNames


class TestStorageConfiguration:
    """Test Storage class configuration and initialization."""

    def test_aws_s3_configuration(self, mock_settings_aws, mock_boto3_client):
        """Test AWS S3 configuration uses default endpoints."""
        mock_client_func, mock_client_instance = mock_boto3_client

        storage = Storage(connect=True)

        # Verify boto3 client called with correct configuration
        mock_client_func.assert_called_once_with(
            "s3",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            region_name="us-east-1",
            # No endpoint_url for AWS - this is critical!
            # No config for AWS - path-style addressing is for S3-compatible only
        )

    def test_s3_compatible_configuration(
        self, mock_settings_s3_compatible, mock_boto3_client
    ):
        """Test S3-compatible configuration uses custom endpoint."""
        mock_client_func, mock_client_instance = mock_boto3_client

        storage = Storage(connect=True)

        # Verify boto3 client called with correct configuration
        call_args = mock_client_func.call_args
        assert call_args[0][0] == "s3"
        assert call_args[1]["aws_access_key_id"] == "minio_user"
        assert call_args[1]["aws_secret_access_key"] == "minio_password"
        assert call_args[1]["endpoint_url"] == "http://minio:9000"
        assert call_args[1]["use_ssl"] is False
        # Verify path-style addressing config
        assert "config" in call_args[1]
        assert isinstance(call_args[1]["config"], Config)

    def test_cloudflare_r2_configuration(
        self, mock_settings_cloudflare_r2, mock_boto3_client
    ):
        """Test Cloudflare R2 configuration."""
        mock_client_func, mock_client_instance = mock_boto3_client

        storage = Storage(connect=True)

        call_args = mock_client_func.call_args
        assert call_args[1]["endpoint_url"] == "https://account-id.r2.cloudflarestorage.com"
        # R2 uses SSL
        assert "use_ssl" not in call_args[1] or call_args[1].get("use_ssl") is not False


class TestHelperMethods:
    """Test Storage helper methods."""

    def test_get_endpoint_url_aws(self, mock_settings_aws):
        """Test endpoint URL generation for AWS."""
        storage = Storage(connect=False)
        assert storage._get_endpoint_url() is None

    def test_get_endpoint_url_s3_compatible_http(self, mock_settings_s3_compatible):
        """Test endpoint URL generation for S3-compatible (HTTP)."""
        storage = Storage(connect=False)
        assert storage._get_endpoint_url() == "http://minio:9000"

    def test_get_endpoint_url_s3_compatible_https(self, mock_settings_cloudflare_r2):
        """Test endpoint URL generation for S3-compatible (HTTPS)."""
        storage = Storage(connect=False)
        # Endpoint already has protocol
        assert storage._get_endpoint_url() == "https://account-id.r2.cloudflarestorage.com"

    def test_get_bucket_name_with_prefix(self, mock_settings_s3_compatible_with_prefix):
        """Test bucket name with prefix."""
        storage = Storage(connect=False)
        assert storage._get_bucket_name("photos") == "teledash-photos"
        assert storage._get_bucket_name("videos") == "teledash-videos"

    def test_get_bucket_name_without_prefix(self, mock_settings_s3_compatible):
        """Test bucket name without prefix."""
        storage = Storage(connect=False)
        assert storage._get_bucket_name("photos") == "photos"
        assert storage._get_bucket_name("videos") == "videos"


class TestBucketOperations:
    """Test bucket existence and creation operations."""

    def test_bucket_exists_true(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test bucket_exists returns True when bucket exists."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.head_bucket.return_value = True

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        assert storage.bucket_exists("test-bucket") is True
        mock_client_instance.head_bucket.assert_called_with(Bucket="test-bucket")

    def test_bucket_exists_false_404(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test bucket_exists returns False for 404 error."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        assert storage.bucket_exists("nonexistent") is False

    def test_bucket_exists_false_no_such_bucket(
        self, mock_settings_s3_compatible, mock_boto3_client
    ):
        """Test bucket_exists returns False for NoSuchBucket error."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket"}}, "HeadBucket"
        )

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        assert storage.bucket_exists("nonexistent") is False

    def test_bucket_exists_reraises_access_denied(
        self, mock_settings_s3_compatible, mock_boto3_client
    ):
        """Test bucket_exists re-raises non-404 errors."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "HeadBucket"
        )

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        with pytest.raises(ClientError):
            storage.bucket_exists("forbidden")

    def test_bucket_exists_with_prefix(
        self, mock_settings_s3_compatible_with_prefix, mock_boto3_client
    ):
        """Test bucket_exists applies prefix."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.head_bucket.return_value = True

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.bucket_exists("photos")
        mock_client_instance.head_bucket.assert_called_with(Bucket="teledash-photos")

    def test_make_bucket_success(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test make_bucket creates bucket successfully."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.create_bucket.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.make_bucket("test-bucket")
        mock_client_instance.create_bucket.assert_called_with(Bucket="test-bucket")

    def test_make_bucket_with_prefix(
        self, mock_settings_s3_compatible_with_prefix, mock_boto3_client
    ):
        """Test make_bucket applies prefix."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.create_bucket.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.make_bucket("photos")
        mock_client_instance.create_bucket.assert_called_with(Bucket="teledash-photos")

    def test_make_bucket_handles_already_owned(
        self, mock_settings_s3_compatible, mock_boto3_client
    ):
        """Test make_bucket handles BucketAlreadyOwnedByYou gracefully."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.create_bucket.side_effect = ClientError(
            {"Error": {"Code": "BucketAlreadyOwnedByYou"}}, "CreateBucket"
        )

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        # Should not raise
        storage.make_bucket("existing")

    def test_make_bucket_reraises_other_errors(
        self, mock_settings_s3_compatible, mock_boto3_client
    ):
        """Test make_bucket re-raises other errors."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.create_bucket.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "CreateBucket"
        )

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        with pytest.raises(ClientError):
            storage.make_bucket("forbidden")

    def test_make_bucket_aws_us_east_1_no_location_constraint(
        self, mock_settings_aws, mock_boto3_client
    ):
        """Test make_bucket for AWS us-east-1 doesn't add LocationConstraint."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_settings_aws.storage_region = "us-east-1"
        mock_client_instance.create_bucket.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.make_bucket("test-bucket")
        # Should be called without CreateBucketConfiguration
        mock_client_instance.create_bucket.assert_called_with(Bucket="test-bucket")

    def test_make_bucket_aws_other_region_with_location_constraint(
        self, mock_settings_aws_with_prefix, mock_boto3_client
    ):
        """Test make_bucket for AWS non-us-east-1 adds LocationConstraint."""
        mock_client_func, mock_client_instance = mock_boto3_client
        # mock_settings_aws_with_prefix uses eu-central-1
        mock_client_instance.create_bucket.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.make_bucket("test-bucket")
        # Should be called with CreateBucketConfiguration
        mock_client_instance.create_bucket.assert_called_with(
            Bucket="teledash-test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
        )


class TestFileOperations:
    """Test file operations (upload, download, delete)."""

    def test_fput_object_success(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test fput_object uploads file successfully."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.upload_file.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.fput_object("photos", "test.jpg", "/tmp/test.jpg", "image/jpeg")
        mock_client_instance.upload_file.assert_called_with(
            "/tmp/test.jpg",
            "photos",
            "test.jpg",
            ExtraArgs={"ContentType": "image/jpeg"},
        )

    def test_fput_object_without_content_type(
        self, mock_settings_s3_compatible, mock_boto3_client
    ):
        """Test fput_object uploads file without content type."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.upload_file.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.fput_object("photos", "test.jpg", "/tmp/test.jpg")
        mock_client_instance.upload_file.assert_called_with(
            "/tmp/test.jpg", "photos", "test.jpg"
        )

    def test_fput_object_with_prefix(
        self, mock_settings_s3_compatible_with_prefix, mock_boto3_client
    ):
        """Test fput_object applies bucket prefix."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.upload_file.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.fput_object("photos", "test.jpg", "/tmp/test.jpg", "image/jpeg")
        mock_client_instance.upload_file.assert_called_with(
            "/tmp/test.jpg",
            "teledash-photos",
            "test.jpg",
            ExtraArgs={"ContentType": "image/jpeg"},
        )

    def test_fget_object_success(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test fget_object downloads file successfully."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.download_file.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.fget_object("photos", "test.jpg", "/tmp/test.jpg")
        mock_client_instance.download_file.assert_called_with(
            "photos", "test.jpg", "/tmp/test.jpg"
        )

    def test_fget_object_with_prefix(
        self, mock_settings_s3_compatible_with_prefix, mock_boto3_client
    ):
        """Test fget_object applies bucket prefix."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.download_file.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.fget_object("photos", "test.jpg", "/tmp/test.jpg")
        mock_client_instance.download_file.assert_called_with(
            "teledash-photos", "test.jpg", "/tmp/test.jpg"
        )

    def test_get_object_success(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test get_object retrieves file object."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_response = {"Body": Mock(), "ContentType": "image/jpeg"}
        mock_client_instance.get_object.return_value = mock_response

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        result = storage.get_object("photos", "test.jpg")
        mock_client_instance.get_object.assert_called_with(
            Bucket="photos", Key="test.jpg"
        )
        assert result == mock_response

    def test_get_object_with_prefix(
        self, mock_settings_s3_compatible_with_prefix, mock_boto3_client
    ):
        """Test get_object applies bucket prefix."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_response = {"Body": Mock(), "ContentType": "image/jpeg"}
        mock_client_instance.get_object.return_value = mock_response

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.get_object("photos", "test.jpg")
        mock_client_instance.get_object.assert_called_with(
            Bucket="teledash-photos", Key="test.jpg"
        )

    def test_remove_object_success(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test remove_object deletes file successfully."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.delete_object.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.remove_object("photos", "test.jpg")
        mock_client_instance.delete_object.assert_called_with(
            Bucket="photos", Key="test.jpg"
        )

    def test_remove_object_with_prefix(
        self, mock_settings_s3_compatible_with_prefix, mock_boto3_client
    ):
        """Test remove_object applies bucket prefix."""
        mock_client_func, mock_client_instance = mock_boto3_client
        mock_client_instance.delete_object.return_value = None

        storage = Storage(connect=False)
        storage.client = mock_client_instance

        storage.remove_object("photos", "test.jpg")
        mock_client_instance.delete_object.assert_called_with(
            Bucket="teledash-photos", Key="test.jpg"
        )


class TestAutomaticBucketCreation:
    """Test automatic bucket creation on connect."""

    def test_creates_all_required_buckets(
        self, mock_settings_s3_compatible, mock_boto3_client
    ):
        """Test that all required buckets are created automatically."""
        mock_client_func, mock_client_instance = mock_boto3_client

        # Mock all buckets don't exist
        mock_client_instance.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        mock_client_instance.create_bucket.return_value = None

        storage = Storage(connect=True)

        # Get all bucket names from enum
        expected_buckets = [member.value for member in StorageBucketNames]

        # Verify all buckets were checked and created
        assert mock_client_instance.head_bucket.call_count == len(expected_buckets)
        assert mock_client_instance.create_bucket.call_count == len(expected_buckets)

    def test_skips_existing_buckets(self, mock_settings_s3_compatible, mock_boto3_client):
        """Test that existing buckets are not recreated."""
        mock_client_func, mock_client_instance = mock_boto3_client

        # Mock all buckets already exist
        mock_client_instance.head_bucket.return_value = True

        storage = Storage(connect=True)

        # Verify buckets were checked but not created
        expected_bucket_count = len([member.value for member in StorageBucketNames])
        assert mock_client_instance.head_bucket.call_count == expected_bucket_count
        assert mock_client_instance.create_bucket.call_count == 0
