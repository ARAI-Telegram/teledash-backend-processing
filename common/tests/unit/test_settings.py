"""Unit tests for Settings validation."""

import os

import pytest
from pydantic import ValidationError

from common.settings import Settings


class TestSettingsValidation:
    """Test cases for Settings validation logic."""

    def setup_method(self):
        """Clear environment variables before each test."""
        self.env_backup = {}
        storage_env_vars = [
            "STORAGE_PROVIDER",
            "STORAGE_REGION",
            "STORAGE_USE_SSL",
            "STORAGE_ENDPOINT",
            "STORAGE_ACCESS_KEY",
            "STORAGE_SECRET_KEY",
            "STORAGE_BUCKET_PREFIX",
            "ASR_ATTACHMENT_TYPES",
        ]

        for var in storage_env_vars:
            if var in os.environ:
                self.env_backup[var] = os.environ[var]
                del os.environ[var]

    def teardown_method(self):
        """Restore environment variables after each test."""
        for var, value in self.env_backup.items():
            os.environ[var] = value

    def test_aws_missing_region_fails(self):
        """Test that AWS configuration without region fails validation."""
        config = {
            "storage_provider": "aws",
            # Missing storage_region
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }
        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "storage_region is required when using AWS provider" in str(
            exc_info.value
        )

    def test_aws_empty_region_fails(self):
        """Test that AWS with empty string region fails validation."""
        config = {
            "storage_provider": "aws",
            "storage_region": "",  # Empty string should fail
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }
        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "storage_region is required when using AWS provider" in str(
            exc_info.value
        )

    def test_aws_with_region_succeeds(self):
        """Test that AWS with region succeeds."""
        config = {
            "storage_provider": "aws",
            "storage_region": "us-east-1",
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)
        assert settings.storage_provider == "aws"
        assert settings.storage_region == "us-east-1"

    def test_s3_compatible_without_region_succeeds(self):
        """Test that S3-compatible without region succeeds."""
        config = {
            "storage_provider": "s3-compatible",
            "storage_endpoint": "minio:9000",
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)
        assert settings.storage_provider == "s3-compatible"
        assert settings.storage_region is None

    def test_default_values(self):
        """Test default values are set correctly."""
        config = {
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)

        # Check defaults
        assert settings.storage_provider == "s3-compatible"
        assert settings.storage_region is None
        assert settings.storage_use_ssl is False
        assert settings.storage_endpoint == "minio:9000"
        assert settings.storage_bucket_prefix == "teledash-"

    def test_custom_bucket_prefix(self):
        """Test custom bucket prefix configuration."""
        config = {
            "storage_bucket_prefix": "dev-",
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)
        assert settings.storage_bucket_prefix == "dev-"

    def test_empty_bucket_prefix(self):
        """Test empty bucket prefix configuration."""
        config = {
            "storage_bucket_prefix": "",
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)
        assert settings.storage_bucket_prefix == ""

    def test_aws_s3_realistic_config(self):
        """Test realistic AWS S3 configuration."""
        config = {
            "storage_provider": "aws",
            "storage_region": "us-west-2",
            "storage_use_ssl": True,
            "storage_access_key": "AKIAIOSFODNN7EXAMPLE",
            "storage_secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)
        assert settings.storage_provider == "aws"
        assert settings.storage_region == "us-west-2"
        assert settings.storage_use_ssl is True

    def test_minio_realistic_config(self):
        """Test realistic MinIO configuration."""
        config = {
            "storage_provider": "s3-compatible",
            "storage_endpoint": "minio:9000",
            "storage_use_ssl": False,
            "storage_access_key": "minioadmin",
            "storage_secret_key": "minioadmin",
            "storage_bucket_prefix": "teledash-",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)
        assert settings.storage_provider == "s3-compatible"
        assert settings.storage_endpoint == "minio:9000"
        assert settings.storage_use_ssl is False
        assert settings.storage_bucket_prefix == "teledash-"

    def test_cloudflare_r2_realistic_config(self):
        """Test realistic Cloudflare R2 configuration."""
        config = {
            "storage_provider": "s3-compatible",
            "storage_endpoint": "https://account-id.r2.cloudflarestorage.com",
            "storage_use_ssl": True,
            "storage_access_key": "r2_access_key",
            "storage_secret_key": "r2_secret_key",
            "asr_attachment_types": ["audio", "voice"],
        }
        settings = Settings(**config)
        assert settings.storage_provider == "s3-compatible"
        assert (
            settings.storage_endpoint == "https://account-id.r2.cloudflarestorage.com"
        )
        assert settings.storage_region is None  # R2 doesn't use regions
        assert settings.storage_use_ssl is True

    def test_type_coercion_from_strings(self):
        """Test that environment variable strings are converted correctly."""
        config = {
            "storage_use_ssl": "true",  # String instead of bool
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }

        settings = Settings(**config)  # type: ignore
        assert settings.storage_use_ssl is True

    def test_ssl_false_string_coercion(self):
        """Test that 'false' string converts to False."""
        config = {
            "storage_use_ssl": "false",  # String 'false'
            "storage_access_key": "test_key",
            "storage_secret_key": "test_secret",
            "asr_attachment_types": ["audio", "voice"],
        }

        settings = Settings(**config)  # type: ignore
        assert settings.storage_use_ssl is False
