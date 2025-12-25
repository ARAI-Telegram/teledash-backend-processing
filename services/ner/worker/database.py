"""
Database operations for Named Entity Recognition service.
"""

from elasticsearch import Elasticsearch

from common.settings import settings


class Database:
    """Database client for NER operations."""

    def __init__(self):
        """Initialize Elasticsearch client."""
        self.client = Elasticsearch(
            [
                {
                    "host": settings.elastic_host,
                    "port": settings.elastic_port,
                    "scheme": "http",
                }
            ],
            timeout=300,
            max_retries=10,
            retry_on_timeout=True,
        )
