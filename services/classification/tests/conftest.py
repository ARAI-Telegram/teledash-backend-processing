"""Shared test fixtures for Classification service."""
import sys
from pathlib import Path
from datetime import datetime

import pytest

from common.settings import settings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_german_texts():
    """Fixture providing sample German text for classification testing."""
    return [
        {
            "id": "test_msg_1",
            "text": "Die Regierung versucht uns zu kontrollieren durch geheime Überwachung. Das ist eine Verschwörung gegen das Volk.",
            "language": "de"
        },
        {
            "id": "test_msg_2",
            "text": "Heute ist ein schöner Tag. Die Sonne scheint und ich freue mich auf den Spaziergang im Park.",
            "language": "de"
        },
        {
            "id": "test_msg_3",
            "text": "Wissenschaftler haben eine neue Entdeckung gemacht, die unser Verständnis des Universums verändert.",
            "language": "de"
        },
    ]


@pytest.fixture
def sample_short_texts():
    """Fixture providing texts that are too short for classification."""
    return [
        {"id": "short_1", "text": "Hallo", "language": "de"},
        {"id": "short_2", "text": "Test", "language": "de"},
        {"id": "short_3", "text": "Okay", "language": "de"},
    ]


@pytest.fixture
def sample_empty_texts():
    """Fixture providing texts that become empty after preprocessing."""
    return [
        {"id": "empty_1", "text": "@user1 @user2 @user3", "language": "de"},  # Only handles
        {"id": "empty_2", "text": "https://example.com http://test.com", "language": "de"},  # Only URLs
        {"id": "empty_3", "text": "   ", "language": "de"},  # Only whitespace
    ]


@pytest.fixture
def sample_texts_with_noise():
    """Fixture providing texts with URLs, IBANs, and social media handles."""
    return [
        {
            "id": "noise_1",
            "text": "Schau mal hier @username https://example.com das ist wichtig für die Analyse der Situation",
            "language": "de"
        },
        {
            "id": "noise_2",
            "text": "Bankverbindung: DE89370400440532013000 aber das ist eigentlich irrelevant für den Inhalt der Nachricht",
            "language": "de"
        },
    ]


@pytest.fixture
def mock_classification_pipeline():
    """Fixture providing a mocked Hugging Face classification pipeline."""
    from unittest.mock import Mock

    mock_pipeline = Mock()
    # Configure mock to return classification results
    mock_pipeline.return_value = [
        {"label": "LABEL_1", "score": 0.85},  # Positive
        {"label": "LABEL_0", "score": 0.92},  # Negative
    ]

    return mock_pipeline


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return settings.gpu_use


@pytest.fixture
def model_cache_path(tmp_path):
    """Provide temporary cache path for model downloads."""
    cache_dir = tmp_path / ".cache" / "transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


@pytest.fixture(scope="session")
def tiny_model_name():
    """Use the configured classification model name.

    Returns the model name from settings for use in tests.
    """
    return settings.classification_model


@pytest.fixture
def sample_classification_messages():
    """Fixture providing sample ClassificationMessage objects."""
    from common.database.message import ClassificationMessage

    return [
        ClassificationMessage(
            id="msg_1",
            chat_id="test_chat_1",
            date=datetime(2025, 1, 12, 10, 0, 0),
            language="de",
            text="Dies ist eine Testnachricht für die Klassifizierung.",
        ),
        ClassificationMessage(
            id="msg_2",
            chat_id="test_chat_1",
            date=datetime(2025, 1, 12, 10, 1, 0),
            language="de",
            caption="Eine Bildunterschrift mit wichtigem Text.",
        ),
        ClassificationMessage(
            id="msg_3",
            chat_id="test_chat_1",
            date=datetime(2025, 1, 12, 10, 2, 0),
            language="de",
            text="@user https://example.com",  # Will become empty after preprocessing
        ),
    ]


@pytest.fixture
def mock_database():
    """Fixture providing a mocked Database instance."""
    from unittest.mock import Mock

    mock_db = Mock()
    mock_db.client = Mock()
    mock_db.client.ping.return_value = True

    # Mock common database methods
    mock_db.get_all_indices_by_alias.return_value = ["messages_test_chat_1"]
    mock_db.get_unclassified_docs.return_value = []
    mock_db.index_exists.return_value = True
    mock_db.index_empty.return_value = False
    mock_db.bulk_write.return_value = None

    return mock_db


@pytest.fixture
def mock_elasticsearch_client():
    """Fixture providing a mocked Elasticsearch client."""
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.indices.exists.return_value = True
    mock_client.indices.get.return_value = {"messages_test_chat_1": {}}
    mock_client.count.return_value = {"count": 10}

    return mock_client


@pytest.fixture
def sample_classification_results():
    """Fixture providing sample classification results."""
    from common.database.classification_result import ClassificationResult
    from common.utils import naive_utcnow

    return [
        ClassificationResult(
            classified=True,
            score_pos=0.85,
            processed_at=naive_utcnow(),
        ),
        ClassificationResult(
            classified=True,
            score_pos=0.15,
            processed_at=naive_utcnow(),
        ),
        ClassificationResult(
            classified=False,
            error="Text too short",
            processed_at=naive_utcnow(),
        ),
    ]
