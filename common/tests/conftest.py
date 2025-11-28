"""Shared pytest fixtures for common module tests."""

import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def mock_settings_aws():
    """Mock settings for AWS S3."""
    with patch("common.storage.settings") as mock_settings:
        mock_settings.storage_provider = "aws"
        mock_settings.storage_region = "us-east-1"
        mock_settings.storage_use_ssl = True
        mock_settings.storage_endpoint = "minio:9000"  # Should be ignored for AWS
        mock_settings.storage_access_key = "test_access_key"
        mock_settings.storage_secret_key = "test_secret_key"
        mock_settings.storage_bucket_prefix = ""
        yield mock_settings


@pytest.fixture
def mock_settings_aws_with_prefix():
    """Mock settings for AWS S3 with bucket prefix."""
    with patch("common.storage.settings") as mock_settings:
        mock_settings.storage_provider = "aws"
        mock_settings.storage_region = "eu-central-1"
        mock_settings.storage_use_ssl = True
        mock_settings.storage_endpoint = "minio:9000"
        mock_settings.storage_access_key = "test_access_key"
        mock_settings.storage_secret_key = "test_secret_key"
        mock_settings.storage_bucket_prefix = "teledash-"
        yield mock_settings


@pytest.fixture
def mock_settings_s3_compatible():
    """Mock settings for S3-compatible providers (MinIO, R2, etc.)."""
    with patch("common.storage.settings") as mock_settings:
        mock_settings.storage_provider = "s3-compatible"
        mock_settings.storage_region = None
        mock_settings.storage_use_ssl = False
        mock_settings.storage_endpoint = "minio:9000"
        mock_settings.storage_access_key = "minio_user"
        mock_settings.storage_secret_key = "minio_password"
        mock_settings.storage_bucket_prefix = ""
        yield mock_settings


@pytest.fixture
def mock_settings_s3_compatible_with_prefix():
    """Mock settings for S3-compatible with bucket prefix."""
    with patch("common.storage.settings") as mock_settings:
        mock_settings.storage_provider = "s3-compatible"
        mock_settings.storage_region = None
        mock_settings.storage_use_ssl = False
        mock_settings.storage_endpoint = "minio:9000"
        mock_settings.storage_access_key = "minio_user"
        mock_settings.storage_secret_key = "minio_password"
        mock_settings.storage_bucket_prefix = "teledash-"
        yield mock_settings


@pytest.fixture
def mock_settings_cloudflare_r2():
    """Mock settings for Cloudflare R2."""
    with patch("common.storage.settings") as mock_settings:
        mock_settings.storage_provider = "s3-compatible"
        mock_settings.storage_region = None
        mock_settings.storage_use_ssl = True
        mock_settings.storage_endpoint = "https://account-id.r2.cloudflarestorage.com"
        mock_settings.storage_access_key = "r2_access_key"
        mock_settings.storage_secret_key = "r2_secret_key"
        mock_settings.storage_bucket_prefix = ""
        yield mock_settings


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for unit tests."""
    with patch("common.storage.boto3.client") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        mock_instance.head_bucket.return_value = True
        yield mock_client, mock_instance
