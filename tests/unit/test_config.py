"""Unit tests for AppConfig and QueueConfig.

Tests cover:
- AppConfig.from_env: defaults, env var overrides, dev_mode parsing
- AppConfig._parse_positive_int: valid, non-integer, zero, negative
- AppConfig.to_env: round-trip consistency
- QueueConfig.from_env: defaults, env var overrides
- QueueConfig._parse_positive_int: valid, non-integer, zero, negative
- QueueConfig.to_env: round-trip consistency
"""

import sys
from unittest.mock import patch

import pytest

from src.services.api.src.config import AppConfig
from src.services.api.src.queue.config import QueueConfig


# ============================================================================
# TestAppConfigParsePositiveInt
# ============================================================================


class TestAppConfigParsePositiveInt:
    """Tests for AppConfig._parse_positive_int()."""

    def test_valid_integer(self):
        """Verify valid positive integer string is parsed correctly."""
        assert AppConfig._parse_positive_int("42", "TEST_VAR") == 42

    def test_non_integer_raises(self):
        """Verify non-integer string raises ValueError with variable name."""
        with pytest.raises(ValueError, match="TEST_VAR.*not a valid integer"):
            AppConfig._parse_positive_int("abc", "TEST_VAR")

    def test_zero_raises(self):
        """Verify zero raises ValueError requiring positive integer."""
        with pytest.raises(ValueError, match="TEST_VAR.*positive integer"):
            AppConfig._parse_positive_int("0", "TEST_VAR")

    def test_negative_raises(self):
        """Verify negative integer raises ValueError."""
        with pytest.raises(ValueError, match="TEST_VAR.*positive integer"):
            AppConfig._parse_positive_int("-5", "TEST_VAR")

    def test_float_string_raises(self):
        """Verify float string raises ValueError."""
        with pytest.raises(ValueError, match="TEST_VAR.*not a valid integer"):
            AppConfig._parse_positive_int("3.14", "TEST_VAR")

    def test_empty_string_raises(self):
        """Verify empty string raises ValueError."""
        with pytest.raises(ValueError, match="TEST_VAR.*not a valid integer"):
            AppConfig._parse_positive_int("", "TEST_VAR")


# ============================================================================
# TestAppConfigFromEnv
# ============================================================================


class TestAppConfigFromEnv:
    """Tests for AppConfig.from_env()."""

    def test_defaults(self):
        """Verify defaults are applied when no env vars are set."""
        with patch.dict("os.environ", {}, clear=True):
            AppConfig.from_env()
        assert AppConfig.broker_url == "redis://localhost:6379"
        assert AppConfig.socketio_max_http_buffer_size == 100_000_000
        assert AppConfig.socketio_ping_timeout == 60
        assert AppConfig.status_request_timeout_s == 60
        assert AppConfig.dev_mode is False

    def test_broker_url_override(self):
        """Verify NDIF_BROKER_URL env var overrides default."""
        with patch.dict("os.environ", {"NDIF_BROKER_URL": "redis://custom:1234"}):
            AppConfig.from_env()
        assert AppConfig.broker_url == "redis://custom:1234"

    def test_socketio_max_http_buffer_size_override(self):
        """Verify SOCKETIO_MAX_HTTP_BUFFER_SIZE env var overrides default."""
        with patch.dict("os.environ", {"SOCKETIO_MAX_HTTP_BUFFER_SIZE": "500"}):
            AppConfig.from_env()
        assert AppConfig.socketio_max_http_buffer_size == 500

    def test_socketio_ping_timeout_override(self):
        """Verify SOCKETIO_PING_TIMEOUT env var overrides default."""
        with patch.dict("os.environ", {"SOCKETIO_PING_TIMEOUT": "120"}):
            AppConfig.from_env()
        assert AppConfig.socketio_ping_timeout == 120

    def test_status_request_timeout_override(self):
        """Verify STATUS_REQUEST_TIMEOUT_S env var overrides default."""
        with patch.dict("os.environ", {"STATUS_REQUEST_TIMEOUT_S": "30"}):
            AppConfig.from_env()
        assert AppConfig.status_request_timeout_s == 30

    def test_dev_mode_true(self):
        """Verify NDIF_DEV_MODE=true sets dev_mode to True."""
        with patch.dict("os.environ", {"NDIF_DEV_MODE": "true"}):
            AppConfig.from_env()
        assert AppConfig.dev_mode is True

    def test_dev_mode_true_case_insensitive(self):
        """Verify NDIF_DEV_MODE is case-insensitive."""
        with patch.dict("os.environ", {"NDIF_DEV_MODE": "True"}):
            AppConfig.from_env()
        assert AppConfig.dev_mode is True

    def test_dev_mode_false_for_other_values(self):
        """Verify NDIF_DEV_MODE defaults to False for non-true values."""
        with patch.dict("os.environ", {"NDIF_DEV_MODE": "yes"}):
            AppConfig.from_env()
        assert AppConfig.dev_mode is False

    def test_min_nnsight_version_override(self):
        """Verify MIN_NNSIGHT_VERSION env var overrides default."""
        with patch.dict("os.environ", {"MIN_NNSIGHT_VERSION": "0.5.0"}):
            AppConfig.from_env()
        assert AppConfig.min_nnsight_version == "0.5.0"

    def test_min_python_version_override(self):
        """Verify MIN_PYTHON_VERSION env var overrides default."""
        with patch.dict("os.environ", {"MIN_PYTHON_VERSION": "3.13"}):
            AppConfig.from_env()
        assert AppConfig.min_python_version == "3.13"

    def test_min_python_version_default_matches_sys(self):
        """Verify default min_python_version matches current Python major.minor."""
        with patch.dict("os.environ", {}, clear=True):
            AppConfig.from_env()
        expected = ".".join(sys.version.split(".")[0:2])
        assert AppConfig.min_python_version == expected

    def test_invalid_buffer_size_raises(self):
        """Verify invalid SOCKETIO_MAX_HTTP_BUFFER_SIZE raises ValueError."""
        with patch.dict("os.environ", {"SOCKETIO_MAX_HTTP_BUFFER_SIZE": "not_int"}):
            with pytest.raises(ValueError, match="SOCKETIO_MAX_HTTP_BUFFER_SIZE"):
                AppConfig.from_env()

    def test_invalid_ping_timeout_raises(self):
        """Verify invalid SOCKETIO_PING_TIMEOUT raises ValueError."""
        with patch.dict("os.environ", {"SOCKETIO_PING_TIMEOUT": "-1"}):
            with pytest.raises(ValueError, match="SOCKETIO_PING_TIMEOUT"):
                AppConfig.from_env()


# ============================================================================
# TestAppConfigToEnv
# ============================================================================


class TestAppConfigToEnv:
    """Tests for AppConfig.to_env()."""

    def test_returns_all_keys(self):
        """Verify to_env() returns all expected environment variable keys."""
        with patch.dict("os.environ", {}, clear=True):
            AppConfig.from_env()
        result = AppConfig.to_env()
        assert "NDIF_BROKER_URL" in result
        assert "SOCKETIO_MAX_HTTP_BUFFER_SIZE" in result
        assert "SOCKETIO_PING_TIMEOUT" in result
        assert "STATUS_REQUEST_TIMEOUT_S" in result
        assert "MIN_NNSIGHT_VERSION" in result
        assert "MIN_PYTHON_VERSION" in result
        assert "NDIF_DEV_MODE" in result

    def test_values_match_class_attrs(self):
        """Verify to_env() values match the class attributes."""
        with patch.dict("os.environ", {"NDIF_BROKER_URL": "redis://test:9999"}):
            AppConfig.from_env()
        result = AppConfig.to_env()
        assert result["NDIF_BROKER_URL"] == AppConfig.broker_url
        assert result["SOCKETIO_MAX_HTTP_BUFFER_SIZE"] == AppConfig.socketio_max_http_buffer_size
        assert result["NDIF_DEV_MODE"] == AppConfig.dev_mode


# ============================================================================
# TestQueueConfigParsePositiveInt
# ============================================================================


class TestQueueConfigParsePositiveInt:
    """Tests for QueueConfig._parse_positive_int()."""

    def test_valid_integer(self):
        """Verify valid positive integer string is parsed correctly."""
        assert QueueConfig._parse_positive_int("10", "TEST_VAR") == 10

    def test_non_integer_raises(self):
        """Verify non-integer string raises ValueError with variable name."""
        with pytest.raises(ValueError, match="TEST_VAR.*not a valid integer"):
            QueueConfig._parse_positive_int("xyz", "TEST_VAR")

    def test_zero_raises(self):
        """Verify zero raises ValueError."""
        with pytest.raises(ValueError, match="TEST_VAR.*positive integer"):
            QueueConfig._parse_positive_int("0", "TEST_VAR")

    def test_negative_raises(self):
        """Verify negative integer raises ValueError."""
        with pytest.raises(ValueError, match="TEST_VAR.*positive integer"):
            QueueConfig._parse_positive_int("-10", "TEST_VAR")


# ============================================================================
# TestQueueConfigFromEnv
# ============================================================================


class TestQueueConfigFromEnv:
    """Tests for QueueConfig.from_env()."""

    def test_defaults(self):
        """Verify defaults are applied when no env vars are set."""
        with patch.dict("os.environ", {}, clear=True):
            QueueConfig.from_env()
        assert QueueConfig.broker_url == "redis://localhost:6379"
        assert QueueConfig.status_cache_freq_s == 120
        assert QueueConfig.processor_reply_freq_s == 3

    def test_broker_url_override(self):
        """Verify NDIF_BROKER_URL env var overrides default."""
        with patch.dict("os.environ", {"NDIF_BROKER_URL": "redis://other:5555"}):
            QueueConfig.from_env()
        assert QueueConfig.broker_url == "redis://other:5555"

    def test_status_cache_freq_override(self):
        """Verify COORDINATOR_STATUS_CACHE_FREQ_S env var overrides default."""
        with patch.dict("os.environ", {"COORDINATOR_STATUS_CACHE_FREQ_S": "60"}):
            QueueConfig.from_env()
        assert QueueConfig.status_cache_freq_s == 60

    def test_processor_reply_freq_override(self):
        """Verify COORDINATOR_PROCESSOR_REPLY_FREQ_S env var overrides default."""
        with patch.dict("os.environ", {"COORDINATOR_PROCESSOR_REPLY_FREQ_S": "10"}):
            QueueConfig.from_env()
        assert QueueConfig.processor_reply_freq_s == 10

    def test_invalid_status_cache_freq_raises(self):
        """Verify invalid COORDINATOR_STATUS_CACHE_FREQ_S raises ValueError."""
        with patch.dict("os.environ", {"COORDINATOR_STATUS_CACHE_FREQ_S": "bad"}):
            with pytest.raises(ValueError, match="COORDINATOR_STATUS_CACHE_FREQ_S"):
                QueueConfig.from_env()

    def test_invalid_processor_reply_freq_raises(self):
        """Verify invalid COORDINATOR_PROCESSOR_REPLY_FREQ_S raises ValueError."""
        with patch.dict("os.environ", {"COORDINATOR_PROCESSOR_REPLY_FREQ_S": "0"}):
            with pytest.raises(ValueError, match="COORDINATOR_PROCESSOR_REPLY_FREQ_S"):
                QueueConfig.from_env()


# ============================================================================
# TestQueueConfigToEnv
# ============================================================================


class TestQueueConfigToEnv:
    """Tests for QueueConfig.to_env()."""

    def test_returns_all_keys(self):
        """Verify to_env() returns all expected environment variable keys."""
        with patch.dict("os.environ", {}, clear=True):
            QueueConfig.from_env()
        result = QueueConfig.to_env()
        assert "NDIF_BROKER_URL" in result
        assert "COORDINATOR_STATUS_CACHE_FREQ_S" in result
        assert "COORDINATOR_PROCESSOR_REPLY_FREQ_S" in result

    def test_values_match_class_attrs(self):
        """Verify to_env() values match the class attributes."""
        with patch.dict("os.environ", {"COORDINATOR_STATUS_CACHE_FREQ_S": "240"}):
            QueueConfig.from_env()
        result = QueueConfig.to_env()
        assert result["COORDINATOR_STATUS_CACHE_FREQ_S"] == 240
        assert result["NDIF_BROKER_URL"] == QueueConfig.broker_url
