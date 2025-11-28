"""Integration tests for full classification pipeline with real models.

These tests validate the complete classification workflow:
- Preprocessing → Classification → Result creation
- End-to-end with real TelConGBERT model

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/integration/test_classification_pipeline.py -v"
"""

import pytest
from unittest.mock import patch

from common.database.classification_result import ErrorType
from common.settings import settings
from worker.classification.preprocessing import preprocess_df
from worker.classification.classification import classify_batch, create_failed_classification_results

import pandas as pd


@pytest.mark.skipif(
    settings.gpu_use,
    reason="GPU_USE=true, CPU tests only run when GPU_USE=false to avoid model conflicts"
)
class TestClassificationPipelineCPU:
    """Integration tests for full classification pipeline on CPU.

    Note: These tests only run when GPU_USE=false.
    """

    @pytest.fixture(scope="class")
    def cpu_model(self):
        """Load classification model on CPU for testing."""
        from worker.model_init import init_classification_model

        original_gpu = settings.gpu_use
        settings.gpu_use = False

        try:
            model = init_classification_model()
            yield model
        finally:
            settings.gpu_use = original_gpu

            if model is not None:
                del model

            import gc
            gc.collect()

    def test_full_pipeline_valid_texts(self, cpu_model, sample_german_texts):
        """Test full pipeline with valid German texts."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        # Create DataFrame from samples
        df = pd.DataFrame(sample_german_texts)
        df["error"] = None

        # Step 1: Preprocess
        df_preprocessed = preprocess_df(df)

        # All should be valid (no errors)
        valid_docs = df_preprocessed[df_preprocessed.error.isna()]
        assert len(valid_docs) > 0

        # Step 2: Classify
        documents = [
            {"id": str(row["id"]), "text": str(row["text"])}
            for _, row in valid_docs[["id", "text"]].iterrows()
        ]

        results = classify_batch(documents, cpu_model)

        # Verify results
        assert len(results) == len(valid_docs)
        for result in results:
            assert result.classification_result.classified is True
            assert result.classification_result.score_pos is not None
            assert 0.0 <= result.classification_result.score_pos <= 1.0
            assert result.classification_result.processed_at is not None

    def test_full_pipeline_short_texts(self, cpu_model, sample_short_texts):
        """Test full pipeline with texts that are too short."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        df = pd.DataFrame(sample_short_texts)
        df["error"] = None

        # Step 1: Preprocess
        with patch.object(settings, 'classification_min_char_length', 50):
            df_preprocessed = preprocess_df(df)

        # All should have errors
        invalid_docs = df_preprocessed[df_preprocessed.error.notna()]
        assert len(invalid_docs) == len(sample_short_texts)

        # Step 2: Create failed results
        unclassifiable_docs = [
            {"id": str(row["id"]), "error": row["error"]}
            for _, row in invalid_docs[["id", "error"]].iterrows()
        ]

        results = create_failed_classification_results(unclassifiable_docs)

        # Verify failed results
        assert len(results) == len(invalid_docs)
        for result in results:
            assert result.classification_result.classified is False
            assert result.classification_result.score_pos is None
            assert result.classification_result.error is not None

    def test_full_pipeline_empty_texts(self, cpu_model, sample_empty_texts):
        """Test full pipeline with texts that become empty after preprocessing."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        df = pd.DataFrame(sample_empty_texts)
        df["error"] = None

        # Step 1: Preprocess
        df_preprocessed = preprocess_df(df)

        # All should have an error (EMPTY_TEXT or TEXT_TOO_SHORT)
        invalid_docs = df_preprocessed[df_preprocessed.error.notna()]
        assert len(invalid_docs) == len(sample_empty_texts)
        # The logic checks length < min first, so empty texts may be flagged as TOO_SHORT
        assert all(invalid_docs["error"].isin([ErrorType.EMPTY_TEXT, ErrorType.TEXT_TOO_SHORT]))

        # Step 2: Create failed results
        unclassifiable_docs = [
            {"id": str(row["id"]), "error": row["error"]}
            for _, row in invalid_docs[["id", "error"]].iterrows()
        ]

        results = create_failed_classification_results(unclassifiable_docs)

        assert len(results) == len(invalid_docs)
        for result in results:
            # Either error type is acceptable for empty/very short texts
            assert result.classification_result.error in [ErrorType.EMPTY_TEXT, ErrorType.TEXT_TOO_SHORT]

    def test_full_pipeline_mixed_validity(self, cpu_model, sample_german_texts, sample_short_texts):
        """Test full pipeline with mix of valid and invalid texts."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        # Combine valid and invalid samples
        all_samples = sample_german_texts + sample_short_texts
        df = pd.DataFrame(all_samples)
        df["error"] = None

        # Step 1: Preprocess
        with patch.object(settings, 'classification_min_char_length', 50):
            df_preprocessed = preprocess_df(df)

        # Split into valid and invalid
        valid_docs = df_preprocessed[df_preprocessed.error.isna()]
        invalid_docs = df_preprocessed[df_preprocessed.error.notna()]

        assert len(valid_docs) > 0
        assert len(invalid_docs) > 0

        # Step 2a: Classify valid docs
        classified_results = []
        if not valid_docs.empty:
            documents = [
                {"id": str(row["id"]), "text": str(row["text"])}
                for _, row in valid_docs[["id", "text"]].iterrows()
            ]
            classified_results = classify_batch(documents, cpu_model)

        # Step 2b: Create failed results
        unclassified_results = []
        if not invalid_docs.empty:
            unclassifiable_docs = [
                {"id": str(row["id"]), "error": row["error"]}
                for _, row in invalid_docs[["id", "error"]].iterrows()
            ]
            unclassified_results = create_failed_classification_results(unclassifiable_docs)

        # Combine results
        all_results = classified_results + unclassified_results

        assert len(all_results) == len(df)

        # Verify classified results
        for result in classified_results:
            assert result.classification_result.classified is True

        # Verify failed results
        for result in unclassified_results:
            assert result.classification_result.classified is False

    def test_full_pipeline_noise_removal(self, cpu_model, sample_texts_with_noise):
        """Test that URLs and handles are removed before classification."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        df = pd.DataFrame(sample_texts_with_noise)
        df["error"] = None

        # Step 1: Preprocess
        df_preprocessed = preprocess_df(df)

        # Verify URLs and handles removed
        for text in df_preprocessed["text"]:
            assert "@" not in text
            assert "http" not in text
            assert "https" not in text

        # Step 2: Classify cleaned texts
        valid_docs = df_preprocessed[df_preprocessed.error.isna()]
        documents = [
            {"id": str(row["id"]), "text": str(row["text"])}
            for _, row in valid_docs[["id", "text"]].iterrows()
        ]

        results = classify_batch(documents, cpu_model)

        # Should still produce valid results
        assert len(results) > 0
        for result in results:
            assert result.classification_result.classified is True

    def test_pipeline_preserves_message_ids(self, cpu_model, sample_german_texts):
        """Test that message IDs are preserved through the pipeline."""
        if cpu_model is None:
            pytest.skip("CPU model not loaded")

        df = pd.DataFrame(sample_german_texts)
        df["error"] = None

        # Get original IDs
        original_ids = set(df["id"])

        # Preprocess
        df_preprocessed = preprocess_df(df)

        # IDs should still be present
        assert set(df_preprocessed["id"]) == original_ids

        # Classify
        documents = [
            {"id": str(row["id"]), "text": str(row["text"])}
            for _, row in df_preprocessed[["id", "text"]].iterrows()
        ]

        results = classify_batch(documents, cpu_model)

        # IDs should match
        result_ids = {result.message_id for result in results}
        assert result_ids == original_ids


@pytest.mark.skipif(
    not settings.gpu_use,
    reason="GPU_USE=false, GPU tests only run when GPU_USE=true"
)
class TestClassificationPipelineGPU:
    """Integration tests for full classification pipeline on GPU.

    Note: These tests only run when GPU_USE=true.
    """

    @pytest.fixture(scope="class")
    def gpu_model(self):
        """Load classification model on GPU for testing."""
        from worker.model_init import init_classification_model

        model = init_classification_model()

        if model is None:
            pytest.skip("GPU model failed to load - check CUDA availability")

        yield model

        if model is not None:
            del model

        import gc
        gc.collect()

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_full_pipeline_valid_texts_gpu(self, gpu_model, sample_german_texts):
        """Test full pipeline with valid German texts on GPU."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        df = pd.DataFrame(sample_german_texts)
        df["error"] = None

        # Preprocess
        df_preprocessed = preprocess_df(df)

        # Classify
        valid_docs = df_preprocessed[df_preprocessed.error.isna()]
        documents = [
            {"id": str(row["id"]), "text": str(row["text"])}
            for _, row in valid_docs[["id", "text"]].iterrows()
        ]

        results = classify_batch(documents, gpu_model)

        # Verify results
        assert len(results) > 0
        for result in results:
            assert result.classification_result.classified is True
            assert 0.0 <= result.classification_result.score_pos <= 1.0

    @pytest.mark.skipif(
        not settings.gpu_use,
        reason="GPU_USE=false, skipping GPU test"
    )
    def test_pipeline_batch_processing_gpu(self, gpu_model, sample_german_texts):
        """Test that GPU can handle batch processing efficiently."""
        if gpu_model is None:
            pytest.skip("GPU model not loaded")

        # Create larger batch by duplicating samples
        large_batch = sample_german_texts * 10  # 30 texts total
        df = pd.DataFrame(large_batch)
        df["error"] = None

        # Preprocess
        df_preprocessed = preprocess_df(df)

        # Classify
        documents = [
            {"id": str(row["id"]), "text": str(row["text"])}
            for _, row in df_preprocessed[["id", "text"]].iterrows()
        ]

        results = classify_batch(documents, gpu_model)

        # Should handle all at once
        assert len(results) == len(documents)
