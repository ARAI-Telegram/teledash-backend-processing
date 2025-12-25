"""
Storage management for export files using MinIO/S3.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ExportStorage:
    """Manage export file storage in MinIO/S3."""

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str = "exports"):
        """
        Initialize storage client.

        Args:
            endpoint_url: MinIO/S3 endpoint URL
            access_key: Access key
            secret_key: Secret key
            bucket_name: Bucket name for exports
        """
        self.bucket_name = bucket_name

        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        # Ensure bucket exists
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Export bucket '{self.bucket_name}' exists")
        except ClientError:
            try:
                self.client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created export bucket '{self.bucket_name}'")
            except Exception as e:
                logger.error(f"Error creating bucket: {e}")
                raise

    def upload_export(
        self,
        file_data: bytes,
        task_id: str,
        file_format: str,
        analysis_type: str
    ) -> str:
        """
        Upload export file to storage.

        Args:
            file_data: File content as bytes
            task_id: Celery task ID
            file_format: File format (csv, pdf, html)
            analysis_type: Type of analysis

        Returns:
            Object key in storage
        """
        # Generate object key with folder structure
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        object_key = f"exports/{analysis_type}/{timestamp}_{task_id}.{file_format}"

        try:
            # Set content type based on format
            content_types = {
                'csv': 'text/csv',
                'pdf': 'application/pdf',
                'html': 'text/html'
            }
            content_type = content_types.get(file_format, 'application/octet-stream')

            # Upload to storage
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=file_data,
                ContentType=content_type,
                Metadata={
                    'task_id': task_id,
                    'analysis_type': analysis_type,
                    'generated_at': datetime.utcnow().isoformat()
                }
            )

            logger.info(f"Uploaded export to {object_key} ({len(file_data)} bytes)")
            return object_key

        except Exception as e:
            logger.error(f"Error uploading export: {e}", exc_info=True)
            raise

    def generate_presigned_url(
        self,
        object_key: str,
        expiration: int = 604800  # 7 days in seconds
    ) -> str:
        """
        Generate presigned URL for downloading export.

        Args:
            object_key: Object key in storage
            expiration: URL expiration time in seconds (default 7 days)

        Returns:
            Presigned download URL
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key
                },
                ExpiresIn=expiration
            )

            logger.info(f"Generated presigned URL for {object_key}")
            return url

        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}", exc_info=True)
            raise

    def delete_export(self, object_key: str):
        """
        Delete export file from storage.

        Args:
            object_key: Object key to delete
        """
        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            logger.info(f"Deleted export: {object_key}")

        except Exception as e:
            logger.error(f"Error deleting export: {e}", exc_info=True)
            raise

    def cleanup_old_exports(self, days_old: int = 7):
        """
        Delete exports older than specified days.

        Args:
            days_old: Delete exports older than this many days
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # List all objects
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='exports/'
            )

            if 'Contents' not in response:
                logger.info("No exports found for cleanup")
                return

            deleted_count = 0
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.delete_export(obj['Key'])
                    deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old exports")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            raise

    def get_file_size(self, object_key: str) -> Optional[int]:
        """Get file size in bytes."""
        try:
            response = self.client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return response['ContentLength']

        except Exception as e:
            logger.error(f"Error getting file size: {e}")
            return None
