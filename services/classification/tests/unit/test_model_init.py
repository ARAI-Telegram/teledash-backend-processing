"""Unit tests for model initialization.

These tests validate model loading logic for both CPU and GPU modes.
They test configuration, error handling, and device selection.

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/test_model_init.py -v"
"""

import pytest
from unittest.mock import Mock, patch
from requests import HTTPError, Timeout

from common.settings import settings


class TestModelLoadingCPU:
    """Unit tests for model loading in CPU mode."""

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained')
    @patch('worker.model_init.pipeline')
    @patch('worker.model_init.model_path')
    def test_load_model_cpu_device_selection(self, mock_model_path, mock_pipeline, mock_model_class, mock_tokenizer):
        """Test that CPU device is selected when GPU_USE=false."""
        from worker.model_init import init_classification_model

        # Mock model path doesn't exist
        mock_model_path.exists.return_value = False
        mock_model_path.is_dir.return_value = False

        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_class.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance

        # Override settings for this test
        with patch.object(settings, 'gpu_use', False):
            with patch.object(settings, 'load_models_from_huggingface', True):
                model = init_classification_model()

        # Verify pipeline was called with CPU device (-1)
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['device'] == -1
        assert model == mock_pipeline_instance

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained')
    @patch('worker.model_init.pipeline')
    @patch('worker.model_init.model_path')
    def test_load_model_from_local_path(self, mock_model_path, mock_pipeline, mock_model_class, mock_tokenizer):
        """Test loading model from local path when it exists."""
        from worker.model_init import init_classification_model

        # Mock model path exists
        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = True

        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_class.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance

        with patch.object(settings, 'gpu_use', False):
            model = init_classification_model()

        # Verify tokenizer and model loaded from local path
        mock_tokenizer.assert_called_once()
        assert mock_tokenizer.call_args[1]['local_files_only'] is True

        mock_model_class.assert_called_once_with(mock_model_path)

        assert model == mock_pipeline_instance

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.model_path')
    def test_load_model_local_error_raises_runtime_error(self, mock_model_path, mock_tokenizer):
        """Test that OSError during local loading raises RuntimeError."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = True

        # Simulate OSError during local loading
        mock_tokenizer.side_effect = OSError("Model files corrupted")

        with pytest.raises(RuntimeError, match="Error loading model from local path"):
            init_classification_model()

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained')
    @patch('worker.model_init.model_path')
    def test_load_model_download_from_huggingface(self, mock_model_path, mock_model_class, mock_tokenizer):
        """Test downloading model from Hugging Face when not available locally."""
        from worker.model_init import init_classification_model

        # Model not available locally
        mock_model_path.exists.return_value = False

        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()

        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_class.return_value = mock_model_instance

        with patch.object(settings, 'load_models_from_huggingface', True):
            with patch.object(settings, 'gpu_use', False):
                with patch('worker.model_init.pipeline') as mock_pipeline:
                    mock_pipeline.return_value = Mock()
                    init_classification_model()

        # Verify download from Hugging Face (no local_files_only flag)
        mock_tokenizer.assert_called_once()
        assert 'local_files_only' not in mock_tokenizer.call_args[1] or \
               mock_tokenizer.call_args[1].get('local_files_only') is not True

    @patch('worker.model_init.model_path')
    def test_load_model_download_disabled_raises_error(self, mock_model_path):
        """Test that RuntimeError is raised when model not found and download disabled."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = False

        with patch.object(settings, 'load_models_from_huggingface', False):
            with pytest.raises(RuntimeError, match="Model not found locally and downloading is disabled"):
                init_classification_model()

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.model_path')
    def test_load_model_network_error_raises_runtime_error(self, mock_model_path, mock_tokenizer):
        """Test that network errors during download raise RuntimeError."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = False

        # Simulate network error
        mock_tokenizer.side_effect = HTTPError("Network error")

        with patch.object(settings, 'load_models_from_huggingface', True):
            with pytest.raises(RuntimeError, match="Error downloading model"):
                init_classification_model()

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.model_path')
    def test_load_model_timeout_raises_runtime_error(self, mock_model_path, mock_tokenizer):
        """Test that timeout errors during download raise RuntimeError."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = False

        # Simulate timeout
        mock_tokenizer.side_effect = Timeout("Connection timeout")

        with patch.object(settings, 'load_models_from_huggingface', True):
            with pytest.raises(RuntimeError, match="Error downloading model"):
                init_classification_model()


class TestModelLoadingGPU:
    """Unit tests for model loading in GPU mode."""

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained')
    @patch('worker.model_init.pipeline')
    @patch('worker.model_init.model_path')
    def test_load_model_gpu_device_selection(self, mock_model_path, mock_pipeline, mock_model_class, mock_tokenizer):
        """Test that CUDA device is selected when GPU_USE=true."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = False

        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_class.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance

        with patch.object(settings, 'gpu_use', True):
            with patch.object(settings, 'gpu_device', 0):
                with patch.object(settings, 'load_models_from_huggingface', True):
                    model = init_classification_model()

        # Verify pipeline was called with GPU device (0)
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['device'] == 0
        assert model == mock_pipeline_instance

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained')
    @patch('worker.model_init.pipeline')
    @patch('worker.model_init.model_path')
    def test_load_model_gpu_device_index(self, mock_model_path, mock_pipeline, mock_model_class, mock_tokenizer):
        """Test that GPU device index is correctly passed."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = False

        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_class.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance

        with patch.object(settings, 'gpu_use', True):
            with patch.object(settings, 'gpu_device', 2):
                with patch.object(settings, 'load_models_from_huggingface', True):
                    init_classification_model()

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['device'] == 2


class TestModelConfiguration:
    """Unit tests for model configuration and options."""

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained')
    @patch('worker.model_init.pipeline')
    @patch('worker.model_init.model_path')
    def test_pipeline_configuration(self, mock_model_path, mock_pipeline, mock_model_class, mock_tokenizer):
        """Test that pipeline is configured with correct parameters."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = False

        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_class.return_value = mock_model_instance
        mock_pipeline.return_value = mock_pipeline_instance

        with patch.object(settings, 'gpu_use', False):
            with patch.object(settings, 'load_models_from_huggingface', True):
                init_classification_model()

        # Verify pipeline configuration
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['task'] == 'text-classification'
        assert call_kwargs['max_length'] == 512
        assert call_kwargs['truncation'] is True
        assert 'model' in call_kwargs
        assert 'tokenizer' in call_kwargs

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.model_path')
    def test_tokenizer_clean_up_tokenization_spaces(self, mock_model_path, mock_tokenizer):
        """Test that tokenizer is loaded with clean_up_tokenization_spaces=True."""
        from worker.model_init import init_classification_model

        mock_model_path.exists.return_value = True
        mock_model_path.is_dir.return_value = True

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        with patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained'):
            with patch('worker.model_init.pipeline'):
                try:
                    init_classification_model()
                except Exception:
                    pass  # We only care about tokenizer call

        # Verify tokenizer was called with clean_up_tokenization_spaces
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs['clean_up_tokenization_spaces'] is True

    @patch('worker.model_init.AutoTokenizer.from_pretrained')
    @patch('worker.model_init.AutoModelForSequenceClassification.from_pretrained')
    @patch('worker.model_init.pipeline')
    @patch('worker.model_init.model_path')
    def test_model_name_from_settings(self, mock_model_path, mock_pipeline, mock_model_class, mock_tokenizer):
        """Test that model name is taken from settings."""
        from worker.model_init import init_classification_model, model_name

        mock_model_path.exists.return_value = False

        mock_tokenizer.return_value = Mock()
        mock_model_class.return_value = Mock()
        mock_pipeline.return_value = Mock()

        with patch.object(settings, 'classification_model', 'CustomModel'):
            with patch.object(settings, 'load_models_from_huggingface', True):
                with patch.object(settings, 'gpu_use', False):
                    # Need to reload module to pick up new model_name
                    with patch('worker.model_init.model_name', 'CustomModel'):
                        init_classification_model()

        # Verify correct model name was used
        mock_tokenizer.assert_called_once_with('CustomModel', clean_up_tokenization_spaces=True)
        mock_model_class.assert_called_once_with('CustomModel')

    def test_model_path_construction(self):
        """Test that model path is correctly constructed."""
        from worker.model_init import model_path, model_name

        # The model_path should be /app/tmp/model/{model_name}
        assert str(model_path) == f"/app/tmp/model/{model_name}"
