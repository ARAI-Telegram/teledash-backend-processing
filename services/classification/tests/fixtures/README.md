# Classification Test Fixtures

This directory contains test data for the classification service tests.

## Files

All test data is defined in `conftest.py` as pytest fixtures for easy reuse across tests.

## Test Data Categories

### German Text Samples (`sample_german_texts`)
- Contains realistic German text samples for classification testing
- Includes both potentially conspiracy-related and neutral content
- Used for integration tests with real models

### Short Texts (`sample_short_texts`)
- Texts below the minimum character length threshold
- Used to test error handling for `TEXT_TOO_SHORT` errors

### Empty Texts (`sample_empty_texts`)
- Texts that become empty after preprocessing (only URLs, handles, etc.)
- Used to test error handling for `EMPTY_TEXT` errors

### Noisy Texts (`sample_texts_with_noise`)
- Texts containing URLs, IBANs, and social media handles
- Used to test preprocessing and cleaning functions

## Usage

Tests automatically have access to these fixtures via pytest:

```python
def test_classification(sample_german_texts):
    # sample_german_texts is automatically provided
    for sample in sample_german_texts:
        result = classify(sample["text"])
        assert result is not None
```

## Adding New Test Data

To add new test data:

1. Add a new fixture in `conftest.py`
2. Follow the existing naming convention (`sample_*`)
3. Include diverse examples covering edge cases
4. Document the purpose in this README
