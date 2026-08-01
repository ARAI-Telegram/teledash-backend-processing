from enum import Enum
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from common.settings import settings


class StorageBucketNames(str, Enum):
    thumbnails = "thumbnails"  # One size of a photo or a file/sticker thumbnail.
    photos = "photos"  # photo files
    audios = "audios"  # audio files to be treated as music by the Telegram clients.
    documents = "documents"  # generic file, can be mp3, pdf etc.)
    videos = "videos"  # video files
    voices = "voices"  # voice note (audio)
    video_notes = "video-notes"  # video note files (video)


class Storage:
    def __init__(self, connect=True) -> None:
        if connect is True:
            self.connect()

    def _get_endpoint_url(self) -> Optional[str]:
        """Get the appropriate endpoint URL based on provider configuration."""
        if settings.storage_provider == "aws":
            # Use default AWS endpoints
            return None
        elif settings.storage_provider == "s3-compatible":
            # Use configured endpoint for non-AWS providers
            endpoint = settings.storage_endpoint
            if not endpoint.startswith(("http://", "https://")):
                protocol = "https" if settings.storage_use_ssl else "http"
                endpoint = f"{protocol}://{endpoint}"
            return endpoint
        return None

    def _get_bucket_name(self, bucket_name: str) -> str:
        """Get the full bucket name with optional prefix."""
        if settings.storage_bucket_prefix:
            return f"{settings.storage_bucket_prefix}{bucket_name}"
        return bucket_name

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if bucket exists."""
        try:
            self.client.head_bucket(Bucket=self._get_bucket_name(bucket_name))
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["404", "NoSuchBucket"]:
                return False
            else:
                raise e

    def make_bucket(self, bucket_name: str) -> None:
        """Create a bucket."""
        full_bucket_name = self._get_bucket_name(bucket_name)
        try:
            create_bucket_config = {}

            # AWS S3 requires CreateBucketConfiguration for non-us-east-1 regions
            if (
                settings.storage_provider == "aws"
                and settings.storage_region
                and settings.storage_region != "us-east-1"
            ):
                create_bucket_config["CreateBucketConfiguration"] = {
                    "LocationConstraint": settings.storage_region
                }

            if create_bucket_config:
                self.client.create_bucket(
                    Bucket=full_bucket_name, **create_bucket_config
                )
            else:
                self.client.create_bucket(Bucket=full_bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code != "BucketAlreadyOwnedByYou":
                raise e

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: Optional[str] = None,
    ) -> None:
        """Upload a file to the bucket."""
        full_bucket_name = self._get_bucket_name(bucket_name)
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self.client.upload_file(
            file_path, full_bucket_name, object_name, ExtraArgs=extra_args
        )

    def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """Download an object from the bucket to a local file."""
        full_bucket_name = self._get_bucket_name(bucket_name)
        self.client.download_file(full_bucket_name, object_name, file_path)

    def get_object(self, bucket_name: str, object_name: str):
        """Get an object from the bucket."""
        full_bucket_name = self._get_bucket_name(bucket_name)
        return self.client.get_object(Bucket=full_bucket_name, Key=object_name)

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        """Remove an object from the bucket."""
        full_bucket_name = self._get_bucket_name(bucket_name)
        self.client.delete_object(Bucket=full_bucket_name, Key=object_name)

    def connect(self) -> None:
        """Initialize boto3 S3 client with appropriate configuration."""
        # Determine endpoint URL based on provider
        endpoint_url = self._get_endpoint_url()

        # Configure boto3 client based on provider
        client_config = {
            "aws_access_key_id": settings.storage_access_key,
            "aws_secret_access_key": settings.storage_secret_key,
        }

        if endpoint_url:
            client_config["endpoint_url"] = endpoint_url

        if settings.storage_region:
            client_config["region_name"] = settings.storage_region

        # For S3-compatible providers, use path-style addressing and disable the default
        # request/response checksums, which some S3-compatible gateways (e.g. quobjects) reject.
        if settings.storage_provider == "s3-compatible":
            client_config["config"] = Config(
                s3={"addressing_style": "path"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            )

        # Set use_ssl based on settings
        if not settings.storage_use_ssl:
            client_config["use_ssl"] = False

        client_config["verify"] = settings.storage_verify_ssl

        self.client = boto3.client("s3", **client_config)

        # Create buckets if they don't exist
        buckets = [member.value for member in StorageBucketNames._member_map_.values()]
        for bucket in buckets:
            if not self.bucket_exists(bucket):
                self.make_bucket(bucket)


if __name__ == "__main__":
    storage = Storage()
