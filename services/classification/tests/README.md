# Classification Service Tests

This directory contains comprehensive tests for the classification service.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── fixtures/                # Test data (README only, data in conftest.py)
├── unit/                    # Fast, isolated tests with mocked dependencies
│   ├── test_preprocessing.py       # Text cleaning and validation
│   ├── test_classification.py      # Classification logic
│   ├── test_model_init.py          # Model loading
│   ├── test_database.py            # Database operations
│   └── test_tasks.py               # Celery tasks
└── integration/             # Slower tests with real models/backends
    ├── test_model_loading.py       # Real model downloads and loading
    └── test_classification_pipeline.py  # End-to-end classification
```

## Prerequisites

### Install Test Dependencies

Install pytest and test dependencies inside the Docker container:

```bash
docker exec -it classification bash -c "pip install -r common/requirements/requirements.test.txt"
```

### Configuration

Tests use environment variables from `.env`:
- `GPU_USE` - Enable/disable GPU tests (default: false)
- `GPU_DEVICE` - Which GPU to use (default: 0)
- `CLASSIFICATION_MODEL` - Model to test (default: TelConGBERT)
- `CLASSIFICATION_MIN_CHAR_LENGTH` - Minimum text length (default: 50)
- `CLASSIFICATION_LANGUAGES` - Supported languages (default: ["de"])

## Running Tests

All tests should be run inside the Docker container to ensure the correct environment and dependencies.

### Run All Tests

```bash
# From repository root
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ -v"
```

### Run Only Unit Tests (Fast)

```bash
# Unit tests are fast and use mocked dependencies
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/ -v"
```

### Run Only Integration Tests

```bash
# Integration tests require real model downloads (slower, ~hundreds of MB)
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/integration/ -v"
```

### Run Specific Test File

```bash
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/test_preprocessing.py -v"
```

### Run Specific Test

```bash
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/test_preprocessing.py::TestPreprocessDf::test_basic_preprocessing -v"
```

### Run with Coverage

```bash
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ -v \
  --cov=worker \
  --cov-report=html"
```

Coverage report will be generated in `htmlcov/index.html` inside the container.

## Docker Setup for Testing

### CPU Mode (Default)

```bash
# 1. Set GPU_USE=false in .env (default)
# 2. Comment out GPU reservations in docker-compose.yml (default)
# 3. Build and start container

docker compose --profile classification build
docker compose --profile classification up -d

# 4. Install pytest
docker exec -it classification bash -c "pip install -r common/requirements/requirements.test.txt"

# 5. Run tests
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ -v"
```

### GPU Mode

```bash
# 1. Set GPU_USE=true in .env
# 2. Uncomment GPU reservations in docker-compose.yml
# 3. Build and start container (requires NVIDIA Container Toolkit)

docker compose --profile classification build
docker compose --profile classification up -d

# 4. Install pytest
docker exec -it classification bash -c "pip install -r common/requirements/requirements.test.txt"

# 5. Run tests
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ -v"
```

## Test Categories

### Unit Tests

Fast, isolated tests with mocked dependencies. No model downloads or external services required.

#### test_preprocessing.py
- URL, IBAN, and handle removal
- Whitespace normalization
- Text length calculation
- Error flagging (empty text, text too short)
- Edge cases and boundary conditions

#### test_classification.py
- Batch classification with mocked pipeline
- Failed classification result creation
- Score normalization (LABEL_0 vs LABEL_1)
- Error handling and validation

#### test_model_init.py
- Model loading from local path vs Hugging Face
- CPU vs GPU device selection
- Error handling (network errors, CUDA errors, missing models)
- Pipeline configuration

#### test_database.py
- Retrieving unclassified documents
- Creating update actions for Elasticsearch
- Removing classification results
- Index and language filtering

#### test_tasks.py
- init_classification task (queue checking, batch enqueueing)
- classify_documents task (preprocessing, classification, bulk write)
- ClassificationTask base class (model caching)
- Error handling and retries

### Integration Tests

Slower tests that use real models and backends. Require model downloads on first run.

#### test_model_loading.py
- Loading real TelConGBERT model
- CPU vs GPU model initialization
- Single text and batch classification
- Model consistency and determinism
- Truncation of long texts

#### test_classification_pipeline.py
- Full preprocessing → classification → result pipeline
- Mix of valid and invalid texts
- Noise removal (URLs, handles, IBANs)
- Message ID preservation
- Batch processing efficiency

## Writing New Tests

### Unit Tests (tests/unit/)

Place fast, mocked tests in `unit/`. Use mocking to avoid expensive operations:

```python
import pytest
from unittest.mock import Mock, patch
from common.settings import settings

class TestMyFeature:
    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    def test_cpu_feature(self, mock_tokenizer):
        """Test on CPU with mocked model (fast)."""
        mock_tokenizer.return_value = Mock()

        with patch.object(settings, 'gpu_use', False):
            # Your test code
            pass
```

### Integration Tests (tests/integration/)

Place tests requiring real resources in `integration/`. These are slower:

```python
import pytest
from common.settings import settings

class TestMyFeatureIntegration:
    def test_cpu_feature_real(self):
        """Test with real model on CPU."""
        # Your test code with real model
        pass

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_gpu_feature_real(self):
        """Test with real model on GPU."""
        # Your test code with real model
        pass
```

## Test Data

Test data is defined as pytest fixtures in `conftest.py`:

- `sample_german_texts` - Valid German texts for classification
- `sample_short_texts` - Texts below minimum length
- `sample_empty_texts` - Texts that become empty after preprocessing
- `sample_texts_with_noise` - Texts with URLs, handles, IBANs
- `mock_classification_pipeline` - Mocked Hugging Face pipeline
- `mock_database` - Mocked Database instance
- `sample_classification_messages` - Sample message objects

## GPU Testing

### CPU Tests
- Run when `GPU_USE=false` in `.env`
- Use CPU device for model inference
- Faster for testing logic without GPU hardware

### GPU Tests
- Run when `GPU_USE=true` in `.env`
- Require NVIDIA Container Toolkit installed on host
- Require GPU hardware
- Test CUDA device selection and GPU-specific features
- Automatically skipped when GPU_USE=false

## Common Test Patterns

### Testing with Mocked Settings

```python
with patch.object(settings, 'classification_min_char_length', 50):
    result = preprocess_df(df)
```

### Testing Celery Tasks

```python
from worker.classification.tasks import ClassificationTask

# Set up mock model
mock_model = Mock()
ClassificationTask._model = mock_model

# Create task instance
task = ClassificationTask()
task.run = classify_documents.__get__(task, ClassificationTask)

# Execute
task.run(document_batch)
```

### Testing with Real Models

```python
@pytest.fixture(scope="class")
def cpu_model(self):
    """Load real model once for entire test class."""
    from worker.model_init import init_classification_model

    model = init_classification_model()
    yield model

    # Cleanup
    del model
    import gc
    gc.collect()
```

## Debugging Tips

### View Detailed Test Output

```bash
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ -vv -s"
```

The `-s` flag shows print statements and logs.

### Run Failed Tests Only

```bash
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ --lf"
```

### Stop on First Failure

```bash
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ -x"
```

### Run Tests Matching Pattern

```bash
docker exec -it classification bash -c "cd /app && python3 -m pytest tests/ -k 'preprocessing'"
```

## Troubleshooting

### Model Download Fails
- Check internet connection
- Verify `LOAD_MODELS_FROM_HUGGINGFACE=true` in `.env`
- Check Hugging Face API is accessible
- Free up disk space for model downloads

### CUDA Errors
- Verify NVIDIA Container Toolkit is installed: `nvidia-smi`
- Check GPU reservations are uncommented in `docker-compose.yml`
- Verify `GPU_USE=true` in `.env`
- Check GPU is not in use by another process

### Import Errors
- Verify container is built: `docker compose --profile classification build`
- Check test dependencies are installed: `pip list | grep pytest`
- Verify Python path includes parent directory

### Elasticsearch Connection Errors
- These are expected in unit tests (using mocks)
- For integration tests requiring Elasticsearch, ensure teledash-backend is running

## Test Coverage Goals

Aim for:
- **Unit tests**: >90% coverage of business logic
- **Integration tests**: Cover critical user-facing paths
- **Edge cases**: Boundary conditions, empty inputs, very long inputs
- **Error handling**: Network errors, invalid inputs, CUDA errors

## Best Practices

1. **Fast unit tests**: Mock external dependencies (models, databases, network)
2. **Descriptive names**: Test names should describe what they test
3. **One assertion per concept**: Test one thing at a time
4. **Use fixtures**: Reuse common test data and setup
5. **Clean up resources**: Delete models, close connections in teardown
6. **Skip appropriately**: Use `pytest.skip()` when preconditions aren't met
7. **Document complex tests**: Add docstrings explaining what's being tested

## Continuous Integration

To run tests in CI/CD:

```yaml
test-classification:
  script:
    - docker compose --profile classification build
    - docker compose --profile classification up -d
    - docker exec classification pip install -r common/requirements/requirements.test.txt
    - docker exec classification python3 -m pytest tests/ -v --cov=worker
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [CLAUDE_TESTING.md](../../../CLAUDE_TESTING.md) - Testing patterns and principles
- [ASR tests](../../asr/tests/) - Similar test structure for reference
