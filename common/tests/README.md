# Storage and Settings Tests

This directory contains comprehensive tests for the boto3 storage migration and settings validation.

## Installation

Install test dependencies:

```bash
pip install -r common/requirements/requirements.test.txt
```

## Running Tests

**Note:** If pytest is not in your PATH, use `python3 -m pytest` instead of `pytest`.

### Run All Tests

```bash
python3 -m pytest common/tests/ -v
```

### Run Only Unit Tests (Fast)

```bash
python3 -m pytest common/tests/unit/ -v
```

### Run Only Integration Tests

Integration tests require a properly configured `.env` file with valid storage credentials.

```bash
python3 -m pytest common/tests/integration/ -v
```

### Run with Coverage

```bash
python3 -m pytest common/tests/ -v --cov=common.storage --cov=common.settings --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

## Writing New Tests

### Unit Tests

Unit tests should:

- Use mocked boto3 client (see `conftest.py` fixtures)
- Be fast and isolated
- Not require external dependencies
- Test individual methods and logic

Example:

```python
def test_my_feature(self, mock_settings_s3_compatible, mock_boto3_client):
    mock_client_func, mock_client_instance = mock_boto3_client
    # Test implementation
```

### Integration Tests

Integration tests should:

- Use real storage backend
- Have proper cleanup (try/finally)
- Use UUID for unique test resource names

Example:

```python
def test_real_storage_operation(self, storage):
    # Test with real backend
    pass
```
