"""Unit tests for ASR model loading.

These tests validate model loading logic for both CPU and GPU modes.
They test configuration, error handling, and device selection.

Run these tests inside Docker container:
    docker exec -it asr bash -c "cd /app && python3 -m pytest tests/unit/ -v"
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from common.settings import settings


class TestModelLoadingCPU:
    """Unit tests for model loading in CPU mode."""

    @patch('worker.tasks.WhisperModel')
    def test_load_model_cpu_device_selection(self, mock_whisper):
        """Test that CPU device is selected when GPU_USE=false."""
        from worker.tasks import load_model

        mock_model = Mock()
        mock_whisper.return_value = mock_model

        # Override settings for this test
        with patch.object(settings, 'gpu_use', False):
            with patch.object(settings, 'gpu_device', 0):
                model, duration = load_model("tiny")

        # Verify WhisperModel was called with CPU device
        mock_whisper.assert_called_once()
        call_kwargs = mock_whisper.call_args[1]
        assert call_kwargs['device'] == 'cpu'
        assert call_kwargs['model_size_or_path'] == 'tiny'
        assert model == mock_model

    @patch('worker.tasks.WhisperModel')
    def test_load_model_cpu_handles_invalid_model(self, mock_whisper):
        """Test that invalid model names are handled gracefully."""
        from worker.tasks import load_model

        # Simulate model loading failure
        mock_whisper.side_effect = Exception("Model 'invalid' not found")

        with patch.object(settings, 'gpu_use', False):
            model, duration = load_model("invalid")

        assert model is None
        assert duration is None

    @patch('worker.tasks.WhisperModel')
    def test_load_model_cpu_returns_model_and_duration(self, mock_whisper):
        """Test that load_model returns both model and duration."""
        from worker.tasks import load_model

        mock_model = Mock()
        mock_whisper.return_value = mock_model

        with patch.object(settings, 'gpu_use', False):
            model, duration = load_model("tiny")

        assert model is not None
        assert isinstance(duration, (int, float))


class TestModelLoadingGPU:
    """Unit tests for model loading in GPU mode."""

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    @patch('worker.tasks.WhisperModel')
    def test_load_model_gpu_device_selection(self, mock_whisper):
        """Test that CUDA device is selected when GPU_USE=true."""
        from worker.tasks import load_model

        mock_model = Mock()
        mock_whisper.return_value = mock_model

        with patch.object(settings, 'gpu_use', True):
            with patch.object(settings, 'gpu_device', 0):
                model, duration = load_model("tiny")

        # Verify WhisperModel was called with CUDA device
        mock_whisper.assert_called_once()
        call_kwargs = mock_whisper.call_args[1]
        assert call_kwargs['device'] == 'cuda'
        assert call_kwargs['device_index'] == 0

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    @patch('worker.tasks.WhisperModel')
    def test_load_model_gpu_device_index(self, mock_whisper):
        """Test that GPU device index is correctly passed."""
        from worker.tasks import load_model

        mock_model = Mock()
        mock_whisper.return_value = mock_model

        with patch.object(settings, 'gpu_use', True):
            with patch.object(settings, 'gpu_device', 2):
                model, duration = load_model("tiny")

        call_kwargs = mock_whisper.call_args[1]
        assert call_kwargs['device_index'] == 2

    @patch('worker.tasks.WhisperModel')
    @patch('worker.tasks.logger')
    def test_load_model_gpu_cuda_error_handling(self, mock_logger, mock_whisper):
        """Test that CUDA errors are handled with helpful messages."""
        from worker.tasks import load_model

        # Simulate CUDA error
        mock_whisper.side_effect = Exception("CUDA driver version is insufficient")

        with patch.object(settings, 'gpu_use', True):
            with patch.object(settings, 'gpu_device', 0):
                model, duration = load_model("tiny")

        # Should return None on CUDA error
        assert model is None
        assert duration is None

        # Should log helpful error message
        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "GPU" in error_message
        assert "NVIDIA Container Toolkit" in error_message

    @patch('worker.tasks.WhisperModel')
    def test_load_model_gpu_non_cuda_error(self, mock_whisper):
        """Test that non-CUDA errors are handled differently."""
        from worker.tasks import load_model

        # Simulate non-CUDA error
        mock_whisper.side_effect = Exception("Network timeout")

        with patch.object(settings, 'gpu_use', True):
            model, duration = load_model("tiny")

        assert model is None
        assert duration is None


class TestModelConfiguration:
    """Unit tests for model configuration and options."""

    @patch('worker.tasks.WhisperModel')
    def test_model_cache_path_configuration(self, mock_whisper):
        """Test that model cache path is correctly configured."""
        from worker.tasks import load_model, MODEL_CACHE_PATH

        mock_model = Mock()
        mock_whisper.return_value = mock_model

        with patch.object(settings, 'gpu_use', False):
            load_model("tiny")

        call_kwargs = mock_whisper.call_args[1]
        assert call_kwargs['download_root'] == MODEL_CACHE_PATH

    @patch('worker.tasks.WhisperModel')
    def test_model_size_parameter(self, mock_whisper):
        """Test that model size is correctly passed to WhisperModel."""
        from worker.tasks import load_model

        mock_model = Mock()
        mock_whisper.return_value = mock_model

        with patch.object(settings, 'gpu_use', False):
            load_model("base")

        call_kwargs = mock_whisper.call_args[1]
        assert call_kwargs['model_size_or_path'] == 'base'
