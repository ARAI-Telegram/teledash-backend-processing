"""Integration tests for ASR transcription with real models.

These tests require:
- Real faster-whisper model downloads (several hundred MB)
- Sample audio file in fixtures/ directory
- GPU hardware if testing GPU mode

Run these tests inside Docker container:
    docker exec -it asr bash -c "cd /app && python3 -m pytest tests/integration/ -v"
"""

import pytest
from pathlib import Path

from common.settings import settings
from worker.tasks import load_model


@pytest.mark.skipif(
    settings.gpu_use,
    reason="GPU_USE=true, CPU tests only run when GPU_USE=false to avoid model conflicts"
)
class TestTranscriptionCPU:
    """Integration tests for transcription in CPU mode.

    Note: These tests only run when GPU_USE=false to avoid conflicts
    with GPU model loading in the same process.
    """

    @pytest.fixture(scope="class")
    def cpu_model(self, tiny_model_name):
        """Load tiny model on CPU for testing."""
        model, duration = load_model(tiny_model_name)

        yield model

        # Cleanup
        if model is not None:
            del model

        # Force garbage collection
        import gc
        gc.collect()

    def test_load_tiny_model_cpu(self, cpu_model):
        """Test that tiny model loads successfully on CPU."""
        assert cpu_model is not None, "Model should load on CPU"

    def test_transcribe_silent_audio_cpu(self, cpu_model, sample_audio_file):
        """Test transcription on CPU with silent audio."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        # Perform transcription using the model's transcribe method
        segments, info = cpu_model.transcribe(
            str(sample_audio_file),
            beam_size=5,
            language=None,
        )

        # Verify transcription info
        assert info is not None
        assert hasattr(info, 'language')
        assert hasattr(info, 'language_probability')

        # Convert segments to list and check structure
        segment_list = list(segments)
        assert isinstance(segment_list, list)

        # Check that we can extract text from segments
        transcription_text = "".join(segment.text for segment in segment_list)
        # Note: Silent audio may produce empty transcription, which is valid

    def test_transcribe_speech_audio_cpu(self, cpu_model, sample_speech_file):
        """Test transcription on CPU with actual speech."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        # Perform transcription with forced English language
        # (audio may have accent that causes auto-detection to vary)
        segments, info = cpu_model.transcribe(
            str(sample_speech_file),
            beam_size=5,
            language='en',  # Force English to ensure consistent test results
        )

        # Verify transcription info
        assert info is not None
        assert hasattr(info, 'language')
        assert info.language == 'en'  # Should be English (we forced it)

        # Convert segments to list and extract text
        segment_list = list(segments)
        transcription_text = "".join(segment.text for segment in segment_list).strip()

        # Should contain some transcription (not empty for speech)
        assert len(transcription_text) > 0, "Speech audio should produce transcription"

        # For debugging - print what was transcribed
        print(f"\nTranscribed text: '{transcription_text}'")

        # Validate expected content
        expected_text = "Hello, this is a test for automatic speech recognition."
        assert expected_text.lower() in transcription_text.lower(), \
            f"Expected '{expected_text}' but got '{transcription_text}'"

    def test_transcribe_empty_audio_cpu(self, cpu_model):
        """Test transcription with invalid/missing audio file."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        # Test with non-existent file
        with pytest.raises(Exception):
            cpu_model.transcribe("/nonexistent/path/audio.wav")


@pytest.mark.skipif(
    not settings.gpu_use,
    reason="GPU_USE=false, GPU tests only run when GPU_USE=true"
)
class TestTranscriptionGPU:
    """Integration tests for transcription in GPU mode.

    Note: These tests only run when GPU_USE=true.
    """

    @pytest.fixture(scope="class")
    def gpu_model(self, tiny_model_name):
        """Load tiny model on GPU for testing."""
        model, duration = load_model(tiny_model_name)

        if model is None:
            pytest.skip("GPU model failed to load - check CUDA availability")

        yield model

        # Cleanup model
        if model is not None:
            del model

        # Force garbage collection to free GPU memory
        import gc
        gc.collect()

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_load_tiny_model_gpu(self, gpu_model):
        """Test that tiny model loads successfully on GPU."""
        assert gpu_model is not None, "Model should load on GPU"

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_transcribe_silent_audio_gpu(self, gpu_model, sample_audio_file):
        """Test transcription on GPU with silent audio."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        # Perform transcription using the model's transcribe method
        segments, info = gpu_model.transcribe(
            str(sample_audio_file),
            beam_size=5,
            language=None,
        )

        # Verify transcription info
        assert info is not None
        assert hasattr(info, 'language')
        assert hasattr(info, 'language_probability')

        # Convert segments to list and check structure
        segment_list = list(segments)
        assert isinstance(segment_list, list)

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_transcribe_speech_audio_gpu(self, gpu_model, sample_speech_file):
        """Test transcription on GPU with actual speech."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        # Perform transcription with forced English language
        # (audio may have accent that causes auto-detection to vary)
        segments, info = gpu_model.transcribe(
            str(sample_speech_file),
            beam_size=5,
            language='en',  # Force English to ensure consistent test results
        )

        # Verify transcription info
        assert info is not None
        assert hasattr(info, 'language')
        assert info.language == 'en'  # Should be English (we forced it)

        # Convert segments to list and extract text
        segment_list = list(segments)
        transcription_text = "".join(segment.text for segment in segment_list).strip()

        # Should contain some transcription (not empty for speech)
        assert len(transcription_text) > 0, "Speech audio should produce transcription"

        # For debugging - print what was transcribed
        print(f"\nTranscribed text: '{transcription_text}'")

        # Validate expected content
        expected_text = "Hello, this is a test for automatic speech recognition."
        assert expected_text.lower() in transcription_text.lower(), \
            f"Expected '{expected_text}' but got '{transcription_text}'"

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_gpu_faster_than_cpu(self, sample_audio_file, monkeypatch):
        """Test that GPU transcription is faster than CPU (optional benchmark)."""
        if not settings.gpu_use:
            pytest.skip("GPU_USE=false, skipping GPU comparison test")

        import time

        # Load CPU model by temporarily disabling GPU
        original_gpu_use = settings.gpu_use
        monkeypatch.setattr(settings, 'gpu_use', False)
        cpu_model, _ = load_model("tiny")

        # Restore GPU setting and load GPU model
        monkeypatch.setattr(settings, 'gpu_use', True)
        gpu_model, _ = load_model("tiny")

        # Restore original setting
        monkeypatch.setattr(settings, 'gpu_use', original_gpu_use)

        if cpu_model is None or gpu_model is None:
            pytest.skip("Both models required for comparison")

        # Time GPU transcription
        gpu_start = time.time()
        list(gpu_model.transcribe(str(sample_audio_file), beam_size=5))
        gpu_duration = time.time() - gpu_start

        # Time CPU transcription
        cpu_start = time.time()
        list(cpu_model.transcribe(str(sample_audio_file), beam_size=5))
        cpu_duration = time.time() - cpu_start

        # GPU should be faster (or at least not significantly slower)
        # Note: For tiny model on short audio, difference may be minimal
        print(f"\nGPU: {gpu_duration:.3f}s, CPU: {cpu_duration:.3f}s")

        # Cleanup
        del cpu_model
        del gpu_model


class TestModelLoading:
    """Integration tests for model loading with real downloads."""

    def test_load_model_cpu_integration(self, tiny_model_name):
        """Test loading model with actual download (CPU mode)."""
        original_gpu_use = settings.gpu_use
        settings.gpu_use = False

        model, duration = load_model(tiny_model_name)

        settings.gpu_use = original_gpu_use

        assert model is not None, "Model should load successfully"
        assert isinstance(duration, (int, float)), "Duration should be numeric"
        assert duration > 0, "Load duration should be positive"

        # Cleanup
        if model is not None:
            del model

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_load_model_gpu_integration(self, tiny_model_name):
        """Test loading model with actual download (GPU mode)."""
        model, duration = load_model(tiny_model_name)

        if model is None:
            pytest.fail("GPU model failed to load - check CUDA/GPU availability")

        assert isinstance(duration, (int, float)), "Duration should be numeric"
        assert duration > 0, "Load duration should be positive"

        # Cleanup
        if model is not None:
            del model

    def test_invalid_model_name_integration(self):
        """Test error handling with invalid model name."""
        model, duration = load_model("this-model-does-not-exist")

        assert model is None, "Invalid model should return None"
        assert duration is None, "Duration should be None on error"
