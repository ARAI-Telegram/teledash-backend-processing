"""Integration tests for classification model loading with real models.

These tests require:
- Real model downloads from Hugging Face (several hundred MB)
- GPU hardware if testing GPU mode
- Internet connection for initial model download

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/integration/test_model_loading.py -v"
"""

import pytest

from common.settings import settings


@pytest.mark.skipif(
    settings.gpu_use,
    reason="GPU_USE=true, CPU tests only run when GPU_USE=false to avoid model conflicts"
)
class TestModelLoadingCPU:
    """Integration tests for model loading in CPU mode.

    Note: These tests only run when GPU_USE=false to avoid conflicts
    with GPU model loading in the same process.
    """

    @pytest.fixture(scope="class")
    def cpu_model(self, tiny_model_name):
        """Load classification model on CPU for testing."""
        from worker.model_init import init_classification_model

        # Force CPU mode
        original_gpu = settings.gpu_use
        settings.gpu_use = False

        try:
            model = init_classification_model()
            yield model
        finally:
            settings.gpu_use = original_gpu

            # Cleanup
            if model is not None:
                del model

            # Force garbage collection
            import gc
            gc.collect()

    def test_load_model_cpu(self, cpu_model):
        """Test that model loads successfully on CPU."""
        assert cpu_model is not None, "Model should load on CPU"

        # Verify it's a pipeline
        assert hasattr(cpu_model, '__call__')
        assert hasattr(cpu_model, 'model')
        assert hasattr(cpu_model, 'tokenizer')

    def test_model_device_is_cpu(self, cpu_model):
        """Test that model is on CPU device."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        # Check device - should be -1 for CPU in transformers
        assert cpu_model.device.type == 'cpu', "Model should be on CPU device"

    def test_classify_single_text_cpu(self, cpu_model, sample_german_texts):
        """Test classification of single German text on CPU."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        text = sample_german_texts[0]["text"]

        result = cpu_model([text])

        assert len(result) == 1
        assert "label" in result[0]
        assert "score" in result[0]
        assert result[0]["label"] in ["LABEL_0", "LABEL_1"]
        assert 0.0 <= result[0]["score"] <= 1.0

    def test_classify_batch_cpu(self, cpu_model, sample_german_texts):
        """Test batch classification on CPU."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        texts = [sample["text"] for sample in sample_german_texts]

        results = cpu_model(texts)

        assert len(results) == len(texts)
        for result in results:
            assert "label" in result
            assert "score" in result
            assert result["label"] in ["LABEL_0", "LABEL_1"]
            assert 0.0 <= result["score"] <= 1.0

    def test_model_truncates_long_text(self, cpu_model):
        """Test that model handles texts longer than max_length."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        # Create very long text (>512 tokens)
        long_text = "Das ist ein Test. " * 200  # ~600 tokens

        result = cpu_model([long_text])

        # Should not raise, should truncate
        assert len(result) == 1
        assert "label" in result[0]

    def test_model_handles_empty_batch(self, cpu_model):
        """Test that model handles empty batch gracefully."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        result = cpu_model([])

        assert result == []

    def test_model_consistency(self, cpu_model):
        """Test that same input gives consistent results."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        text = "Die Regierung plant geheime Maßnahmen gegen die Bevölkerung."

        result1 = cpu_model([text])
        result2 = cpu_model([text])

        # Results should be identical (model is deterministic)
        assert result1[0]["label"] == result2[0]["label"]
        assert abs(result1[0]["score"] - result2[0]["score"]) < 0.001


@pytest.mark.skipif(
    not settings.gpu_use,
    reason="GPU_USE=false, GPU tests only run when GPU_USE=true"
)
class TestModelLoadingGPU:
    """Integration tests for model loading in GPU mode.

    Note: These tests only run when GPU_USE=true.
    """

    @pytest.fixture(scope="class")
    def gpu_model(self, tiny_model_name):
        """Load classification model on GPU for testing."""
        from worker.model_init import init_classification_model

        model = init_classification_model()

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
    def test_load_model_gpu(self, gpu_model):
        """Test that model loads successfully on GPU."""
        assert gpu_model is not None, "Model should load on GPU"

        # Verify it's a pipeline
        assert hasattr(gpu_model, '__call__')
        assert hasattr(gpu_model, 'model')
        assert hasattr(gpu_model, 'tokenizer')

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_model_device_is_cuda(self, gpu_model):
        """Test that model is on CUDA device."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        # Check device - should be cuda
        assert gpu_model.device.type == 'cuda', "Model should be on CUDA device"

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_classify_single_text_gpu(self, gpu_model, sample_german_texts):
        """Test classification of single German text on GPU."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        text = sample_german_texts[0]["text"]

        result = gpu_model([text])

        assert len(result) == 1
        assert "label" in result[0]
        assert "score" in result[0]
        assert result[0]["label"] in ["LABEL_0", "LABEL_1"]
        assert 0.0 <= result[0]["score"] <= 1.0

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_classify_batch_gpu(self, gpu_model, sample_german_texts):
        """Test batch classification on GPU."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        texts = [sample["text"] for sample in sample_german_texts]

        results = gpu_model(texts)

        assert len(results) == len(texts)
        for result in results:
            assert "label" in result
            assert "score" in result


class TestModelDownload:
    """Integration tests for model downloading."""

    def test_model_download_from_huggingface(self):
        """Test downloading model from Hugging Face when not available locally."""
        from worker.model_init import init_classification_model

        # This test assumes model might not exist locally
        # If download is disabled, it should raise an error

        if not settings.load_models_from_huggingface:
            pytest.skip("Model downloading is disabled")

        original_gpu = settings.gpu_use
        settings.gpu_use = False

        try:
            model = init_classification_model()
            assert model is not None
        finally:
            settings.gpu_use = original_gpu
            if 'model' in locals() and model is not None:
                del model

    def test_model_local_loading(self):
        """Test loading model from local path when available."""
        from worker.model_init import model_path, init_classification_model

        if not model_path.exists():
            pytest.skip("Local model not available")

        original_gpu = settings.gpu_use
        settings.gpu_use = False

        try:
            model = init_classification_model()
            assert model is not None
        finally:
            settings.gpu_use = original_gpu
            if 'model' in locals() and model is not None:
                del model
