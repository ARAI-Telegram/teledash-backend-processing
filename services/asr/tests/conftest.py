"""Shared test fixtures for ASR service."""
import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_audio_file():
    """Fixture providing path to sample silent audio file for testing.

    Uses a static 3-second silent audio file committed to the repository.
    """
    audio_path = Path(__file__).parent / "fixtures" / "sample_audio.wav"

    if not audio_path.exists():
        pytest.fail(
            f"Sample audio file not found: {audio_path}\n"
            "This file should be committed to the repository."
        )

    return str(audio_path)


@pytest.fixture(scope="session")
def sample_speech_file():
    """Fixture providing path to sample speech audio file for testing.

    Uses a static ~4-second audio file with English speech committed to the repository.
    Contains the phrase: "Hello, this is a test for automatic speech recognition."
    """
    audio_path = Path(__file__).parent / "fixtures" / "sample_speech.wav"

    if not audio_path.exists():
        pytest.fail(
            f"Sample speech file not found: {audio_path}\n"
            "This file should be committed to the repository."
        )

    return str(audio_path)


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    from common.settings import settings
    return settings.gpu_use


@pytest.fixture
def model_cache_path(tmp_path):
    """Provide temporary cache path for model downloads."""
    cache_dir = tmp_path / ".cache" / "faster-whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


@pytest.fixture(scope="session")
def tiny_model_name():
    """Use tiny model for faster tests."""
    return "tiny"
