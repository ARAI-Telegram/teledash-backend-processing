"""
Database connection module for N-grams service.
"""

from elasticsearch import Elasticsearch
from common.settings import settings


class Database:
    """Database client wrapper for Elasticsearch."""

    def __init__(self):
        """Initialize Elasticsearch client."""
        self.client = Elasticsearch(
            [{"host": settings.elastic_host, "port": settings.elastic_port, "scheme": "http"}],
            timeout=300,
            max_retries=10,
            retry_on_timeout=True
        )

        # Verify connection
        if not self.client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
