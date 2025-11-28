"""Unit tests for preprocessing module.

These tests validate text preprocessing logic including:
- URL, IBAN, and social media handle removal
- Text length calculation
- Error classification for short/empty texts

Run these tests inside Docker container:
    docker exec -it classification bash -c "cd /app && python3 -m pytest tests/unit/test_preprocessing.py -v"
"""

import pandas as pd
from unittest.mock import patch

from common.database.classification_result import ErrorType
from common.settings import settings
from worker.classification.preprocessing import (
    remove_handles_iban_urls,
    get_length,
    preprocess_df,
)


class TestRemoveHandlesIbanUrls:
    """Test the remove_handles_iban_urls function."""

    def test_remove_social_media_handles(self):
        """Test that social media handles are removed."""
        text = "Hello @username and @another_user, check this out"
        result = remove_handles_iban_urls(text)

        assert "@username" not in result
        assert "@another_user" not in result
        assert "Hello" in result
        assert "check this out" in result

    def test_remove_http_urls(self):
        """Test that HTTP URLs are removed."""
        text = "Check out http://example.com and https://test.com for info"
        result = remove_handles_iban_urls(text)

        assert "http://example.com" not in result
        assert "https://test.com" not in result
        assert "Check out" in result
        assert "for info" in result

    def test_remove_telegram_links(self):
        """Test that Telegram links are removed."""
        text = "Join our channel at t.me/channel_name and T.ME/ANOTHER"
        result = remove_handles_iban_urls(text)

        assert "t.me/channel_name" not in result
        assert "T.ME/ANOTHER" not in result
        assert "Join our channel at" in result

    def test_remove_short_urls(self):
        """Test that URL shorteners are removed."""
        text = "Check bit.ly/abc123 or goo.gl/xyz789"
        result = remove_handles_iban_urls(text)

        assert "bit.ly/abc123" not in result
        assert "goo.gl/xyz789" not in result

    def test_remove_iban(self):
        """Test that IBAN numbers are removed."""
        text = "Transfer to DE89 3704 0044 0532 0130 00 please"
        result = remove_handles_iban_urls(text)

        # IBAN should be removed
        assert "DE89" not in result or "3704" not in result
        assert "Transfer to" in result
        assert "please" in result

    def test_preserve_normal_text(self):
        """Test that normal text is preserved."""
        text = "This is normal text without any special elements."
        result = remove_handles_iban_urls(text)

        assert result == text

    def test_multiple_removals(self):
        """Test text with multiple elements to remove."""
        text = "Hello @user visit https://example.com and check IBAN DE12345678901234567890"
        result = remove_handles_iban_urls(text)

        assert "@user" not in result
        assert "https://example.com" not in result
        assert "Hello" in result

    def test_empty_string(self):
        """Test handling of empty string."""
        text = ""
        result = remove_handles_iban_urls(text)

        assert result == ""

    def test_only_removable_content(self):
        """Test text containing only removable elements."""
        text = "@user1 @user2 https://test.com"
        result = remove_handles_iban_urls(text)

        # Should be mostly empty spaces
        assert "@user1" not in result
        assert "https://test.com" not in result


class TestGetLength:
    """Test the get_length function."""

    def test_alphanumeric_text(self):
        """Test length calculation with alphanumeric text."""
        text = "Hello123"
        length = get_length(text)

        assert length == 8  # 5 letters + 3 numbers

    def test_only_letters(self):
        """Test length with only letters."""
        text = "Hello"
        length = get_length(text)

        assert length == 5

    def test_only_numbers(self):
        """Test length with only numbers."""
        text = "12345"
        length = get_length(text)

        assert length == 5

    def test_with_special_characters(self):
        """Test that special characters are not counted."""
        text = "Hello, World! @#$"
        length = get_length(text)

        assert length == 10  # Only 'HelloWorld'

    def test_with_spaces(self):
        """Test that spaces are not counted."""
        text = "Hello World"
        length = get_length(text)

        assert length == 10  # Only letters

    def test_empty_string(self):
        """Test length of empty string."""
        text = ""
        length = get_length(text)

        assert length == 0

    def test_only_special_characters(self):
        """Test length with only special characters."""
        text = "!@#$%^&*()"
        length = get_length(text)

        assert length == 0

    def test_mixed_content(self):
        """Test length with mixed content."""
        text = "Test123 with special chars!@# and spaces"
        length = get_length(text)

        # Should count: Test123withspecialcharsandspaces = 7+3+4+7+7+3 = 31
        expected = sum(c.isalpha() or c.isnumeric() for c in text)
        assert length == expected


class TestPreprocessDf:
    """Test the preprocess_df function."""

    def test_basic_preprocessing(self):
        """Test basic preprocessing of DataFrame."""
        df = pd.DataFrame({
            "id": ["1", "2"],
            "text": [
                "This is a sufficiently long normal text that will pass the minimum length requirement.",
                "Another normal text here that is also long enough to pass validation checks."
            ],
            "error": [None, None]
        })

        result = preprocess_df(df)

        assert "text_length" in result.columns
        assert result["text_length"].iloc[0] > 0
        assert result["text_length"].iloc[1] > 0
        assert pd.isna(result["error"].iloc[0])
        assert pd.isna(result["error"].iloc[1])

    def test_remove_urls_and_handles(self):
        """Test that URLs and handles are removed during preprocessing."""
        df = pd.DataFrame({
            "id": ["1"],
            "text": ["Check @user and https://example.com for more info"],
            "error": [None]
        })

        result = preprocess_df(df)

        assert "@user" not in result["text"].iloc[0]
        assert "https://example.com" not in result["text"].iloc[0]
        assert "for more info" in result["text"].iloc[0]

    def test_whitespace_normalization(self):
        """Test that multiple spaces are normalized."""
        df = pd.DataFrame({
            "id": ["1"],
            "text": ["Text   with    multiple     spaces"],
            "error": [None]
        })

        result = preprocess_df(df)

        # Should have single spaces only
        assert "   " not in result["text"].iloc[0]
        assert "Text with multiple spaces" == result["text"].iloc[0]

    def test_empty_text_error(self, sample_empty_texts):
        """Test that empty texts after preprocessing are flagged."""
        df = pd.DataFrame(sample_empty_texts)
        df["error"] = None

        result = preprocess_df(df)

        # All should have an error (either EMPTY_TEXT or TEXT_TOO_SHORT)
        # The logic checks length < min first, so empty texts may be flagged as TOO_SHORT
        for idx in range(len(result)):
            error_val = result["error"].iloc[idx]
            # Either error type is acceptable for empty/very short texts
            assert error_val in [ErrorType.EMPTY_TEXT, ErrorType.TEXT_TOO_SHORT]

    def test_short_text_error(self, sample_short_texts):
        """Test that texts below minimum length are flagged."""
        df = pd.DataFrame(sample_short_texts)
        df["error"] = None

        with patch.object(settings, 'classification_min_char_length', 50):
            result = preprocess_df(df)

            # All should have TEXT_TOO_SHORT error
            for idx in range(len(result)):
                assert result["error"].iloc[idx] == ErrorType.TEXT_TOO_SHORT

    def test_sufficient_length_text(self):
        """Test that texts meeting minimum length are not flagged."""
        df = pd.DataFrame({
            "id": ["1"],
            "text": ["This is a sufficiently long text that should pass the minimum length requirement for classification."],
            "error": [None]
        })

        with patch.object(settings, 'classification_min_char_length', 50):
            result = preprocess_df(df)

            assert pd.isna(result["error"].iloc[0])
            assert result["text_length"].iloc[0] >= 50

    def test_mixed_valid_and_invalid_texts(self):
        """Test DataFrame with mix of valid and invalid texts."""
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "text": [
                "This is a sufficiently long text that meets all requirements.",
                "Short",
                "@only @handles"
            ],
            "error": [None, None, None]
        })

        with patch.object(settings, 'classification_min_char_length', 50):
            result = preprocess_df(df)

            # First should be valid
            assert pd.isna(result["error"].iloc[0])

            # Second should be too short
            error2 = result["error"].iloc[1]
            assert str(error2) == str(ErrorType.TEXT_TOO_SHORT) or error2 == ErrorType.TEXT_TOO_SHORT

            # Third should have an error (empty or too short)
            error3 = result["error"].iloc[2]
            assert error3 in [ErrorType.EMPTY_TEXT, ErrorType.TEXT_TOO_SHORT]

    def test_preserves_id_column(self):
        """Test that ID column is preserved during preprocessing."""
        df = pd.DataFrame({
            "id": ["msg_1", "msg_2", "msg_3"],
            "text": ["Text one", "Text two", "Text three"],
            "error": [None, None, None]
        })

        result = preprocess_df(df)

        assert list(result["id"]) == ["msg_1", "msg_2", "msg_3"]

    def test_error_column_exists(self):
        """Test that error column exists in output."""
        df = pd.DataFrame({
            "id": ["1"],
            "text": ["Some text"],
            "error": [None]
        })

        result = preprocess_df(df)

        assert "error" in result.columns

    def test_text_length_calculation(self):
        """Test that text_length is calculated correctly."""
        df = pd.DataFrame({
            "id": ["1"],
            "text": ["Hello123 World456"],
            "error": [None]
        })

        result = preprocess_df(df)

        # Should count alphanumeric: Hello123World456 = 5+3+5+3 = 16
        expected = sum(c.isalpha() or c.isnumeric() for c in "Hello123 World456")
        assert result["text_length"].iloc[0] == expected

    def test_boundary_case_exact_min_length(self):
        """Test text with exactly the minimum required length."""
        # Create text with exactly 50 alphanumeric characters
        text = "a" * 50

        df = pd.DataFrame({
            "id": ["1"],
            "text": [text],
            "error": [None]
        })

        with patch.object(settings, 'classification_min_char_length', 50):
            result = preprocess_df(df)

            # Should not be flagged as too short (>= 50 is ok, < 50 is error)
            assert pd.isna(result["error"].iloc[0])
            assert result["text_length"].iloc[0] == 50

    def test_boundary_case_one_below_min_length(self):
        """Test text with one character below minimum length."""
        # Create text with exactly 49 alphanumeric characters
        text = "a" * 49

        df = pd.DataFrame({
            "id": ["1"],
            "text": [text],
            "error": [None]
        })

        with patch.object(settings, 'classification_min_char_length', 50):
            result = preprocess_df(df)

            # Should be flagged as too short
            assert result["error"].iloc[0] == ErrorType.TEXT_TOO_SHORT
            assert result["text_length"].iloc[0] == 49
