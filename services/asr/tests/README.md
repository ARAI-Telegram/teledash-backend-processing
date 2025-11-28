# ASR Service Tests

This directory contains tests for the Automatic Speech Recognition (ASR) service.

## Prerequisites

### Install Test Dependencies

Install pytest and test dependencies:

```bash
docker exec -it asr bash -c "pip install -r common/requirements/requirements.test.txt"
```

### Test Audio Files

Two static audio files are included in the `fixtures/` directory and committed to the repository:

- **`sample_audio.wav`** - 3-second silent audio (~94KB)
- **`sample_speech.wav`** - 4-second speech audio with English text (~130KB)

No setup is required - use `sample_audio_file` or `sample_speech_file` fixtures in your tests.

## Running Tests

All tests should be run inside the Docker container to ensure the correct environment and dependencies.

### Run All Tests

```bash
# From repository root
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/ -v"
```

### Run Only Unit Tests (Fast)

```bash
# Unit tests are fast and use mocked dependencies
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/unit/ -v"
```

### Run Only Integration Tests

```bash
# Integration tests require sample audio file and model downloads (slower)
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/integration/ -v"
```

### Run Only CPU Tests

```bash
# Set GPU_USE=false in .env first, then rebuild
docker compose --profile asr down
docker compose --profile asr build
docker compose --profile asr up -d
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/ -v"

# CPU tests will run, GPU tests will be skipped
```

### Run Only GPU Tests

```bash
# Set GPU_USE=true in .env first, uncomment GPU reservations in docker-compose.yml
# Requires NVIDIA Container Toolkit and GPU hardware
docker compose --profile asr down
docker compose --profile asr build
docker compose --profile asr up -d
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/ -v"

# GPU tests will run, CPU tests will be skipped to avoid model conflicts
```

### Run with Coverage

```bash
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/ -v \
  --cov=worker.tasks \
  --cov-report=html"
```

Coverage report will be generated in `htmlcov/index.html` inside the container.

## Docker Setup for Testing

### CPU Mode

```bash
# 1. Set GPU_USE=false in .env
# 2. Comment out GPU reservations in docker-compose.yml (default)
# 3. Build and start container

docker compose --profile asr build
docker compose --profile asr up -d

# 4. Install pytest
docker exec -it asr bash -c "pip install -r common/requirements/requirements.test.txt"

# 5. Run tests
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/ -v"
```

### GPU Mode

```bash
# 1. Set GPU_USE=true in .env
# 2. Uncomment GPU reservations in docker-compose.yml
# 3. Build and start container (requires NVIDIA Container Toolkit)

docker compose --profile asr build
docker compose --profile asr up -d

# 4. Install pytest
docker exec -it asr bash -c "pip install -r common/requirements/requirements.test.txt"

# 5. Run tests
docker exec -it asr bash -c "cd /app && python3 -m pytest tests/ -v"
```

## Test Configuration

Tests use environment variables from `.env`:

- `GPU_USE` - Enable/disable GPU tests
- `GPU_DEVICE` - Which GPU to use (default: 0)
- `ASR_MODEL` - Which model to test (tests override with "tiny" for speed)

## Writing New Tests

### Unit Tests (tests/unit/)

Place fast, mocked tests in `unit/`. Use mocking to avoid expensive operations:

```python
import pytest
from unittest.mock import Mock, patch
from common.settings import settings

class TestMyFeature:
    @patch('worker.tasks.WhisperModel')
    def test_cpu_feature(self, mock_whisper):
        """Test on CPU with mocked model (fast)."""
        mock_model = Mock()
        mock_whisper.return_value = mock_model

        with patch.object(settings, 'gpu_use', False):
            # Your test code
            pass

    @patch('worker.tasks.WhisperModel')
    def test_gpu_feature(self, mock_whisper):
        """Test GPU configuration with mocked model."""
        mock_model = Mock()
        mock_whisper.return_value = mock_model

        with patch.object(settings, 'gpu_use', True):
            # Your test code
            pass
```

### Integration Tests (tests/integration/)

Place tests requiring real resources in `integration/`. These are slower:

```python
import pytest
from common.settings import settings

class TestMyFeatureIntegration:
    def test_cpu_feature_real(self, tiny_model_name):
        """Test with real model on CPU."""
        # Your test code with real model
        pass

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_gpu_feature_real(self, tiny_model_name):
        """Test with real model on GPU."""
        # Your test code with real model
        pass
```
