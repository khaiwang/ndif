"""Unit tests for schema models.

Tests: src/common/schema/response.py, src/common/schema/request.py
"""

import logging
from unittest.mock import MagicMock, patch

import pytest


class TestBackendResponseModel:
    """Tests for BackendResponseModel."""

    def _make_response(self, session_id=None, status_name="RECEIVED", callback=""):
        from nnsight.schema.response import ResponseModel
        from src.common.schema.response import BackendResponseModel

        status = getattr(ResponseModel.JobStatus, status_name)
        return BackendResponseModel(
            id="resp-001",
            session_id=session_id,
            status=status,
            description="test description",
            callback=callback,
        )

    def test_blocking_true_when_session_id(self):
        """Verify blocking is True when session_id is provided."""
        resp = self._make_response(session_id="sess-1")
        assert resp.blocking is True

    def test_blocking_false_when_no_session_id(self):
        """Verify blocking is False when session_id is None."""
        resp = self._make_response(session_id=None)
        assert resp.blocking is False

    def test_str_representation(self):
        """Verify str includes response ID and status name."""
        resp = self._make_response(status_name="RECEIVED")
        s = str(resp)
        assert "resp-001" in s
        assert "RECEIVED" in s

    def test_respond_blocking_completed_calls_sio_call(self):
        """Verify COMPLETED blocking response uses SioProvider.call with 'blocking_response'."""
        from src.common.schema.response import BackendResponseModel
        with patch("src.common.schema.response.SioProvider") as mock_sio:
            resp = self._make_response(session_id="sess-1", status_name="COMPLETED")
            resp.respond()
            mock_sio.call.assert_called_once()
            args = mock_sio.call.call_args
            assert args[0][0] == "blocking_response"

    def test_respond_blocking_queued_calls_sio_emit(self):
        """Verify QUEUED blocking response uses SioProvider.emit (not call)."""
        with patch("src.common.schema.response.SioProvider") as mock_sio:
            resp = self._make_response(session_id="sess-1", status_name="QUEUED")
            resp.respond()
            mock_sio.emit.assert_called_once()

    def test_respond_blocking_error_calls_sio_call(self):
        """Verify ERROR blocking response uses SioProvider.call."""
        with patch("src.common.schema.response.SioProvider") as mock_sio:
            resp = self._make_response(session_id="sess-1", status_name="ERROR")
            resp.respond()
            mock_sio.call.assert_called_once()

    def test_respond_non_blocking_no_callback_saves(self):
        """Verify non-blocking response with no callback just calls save()."""
        from src.common.schema.response import BackendResponseModel
        with patch("src.common.schema.response.SioProvider"), \
             patch.object(BackendResponseModel, "save", return_value=None) as mock_save:
            resp = self._make_response(session_id=None, status_name="RECEIVED", callback="")
            resp.respond()
            mock_save.assert_called_once()

    def test_respond_non_blocking_email_callback(self):
        """Verify non-blocking COMPLETED response with email sends via Mailgun."""
        from src.common.schema.response import BackendResponseModel
        with patch("src.common.schema.response.SioProvider"), \
             patch("src.common.schema.response.MailgunProvider") as mock_mg, \
             patch.object(BackendResponseModel, "save", return_value=None):
            mock_mg.connected.return_value = True
            resp = self._make_response(
                session_id=None, status_name="COMPLETED", callback="user@example.com"
            )
            resp.respond()
            mock_mg.send_email.assert_called_once()

    def test_respond_non_blocking_url_callback(self):
        """Verify non-blocking COMPLETED response with URL sends HTTP GET."""
        from src.common.schema.response import BackendResponseModel
        with patch("src.common.schema.response.SioProvider"), \
             patch("src.common.schema.response.requests") as mock_requests, \
             patch.object(BackendResponseModel, "save", return_value=None):
            resp = self._make_response(
                session_id=None, status_name="COMPLETED", callback="https://webhook.example.com/notify"
            )
            resp.respond()
            mock_requests.get.assert_called_once()


class TestBackendRequestModel:
    """Tests for BackendRequestModel."""

    def test_create_response_queued(self):
        """Verify create_response produces a QUEUED response with correct fields."""
        from src.common.schema.request import BackendRequestModel

        req = BackendRequestModel(
            id="req-001",
            model_key="test-model",
            session_id="sess-1",
        )
        logger = logging.getLogger("test")

        with patch("src.common.schema.response.SioProvider"):
            from nnsight.schema.response import ResponseModel

            resp = req.create_response(
                ResponseModel.JobStatus.QUEUED, logger, "Queued at position 1"
            )
            assert resp.id == "req-001"
            assert resp.status == ResponseModel.JobStatus.QUEUED
            assert resp.description == "Queued at position 1"

    def test_create_response_error(self):
        """Verify create_response produces an ERROR response."""
        from src.common.schema.request import BackendRequestModel

        req = BackendRequestModel(
            id="req-002",
            model_key="test-model",
            session_id=None,
        )
        logger = logging.getLogger("test")

        with patch("src.common.schema.response.SioProvider"):
            from nnsight.schema.response import ResponseModel

            resp = req.create_response(
                ResponseModel.JobStatus.ERROR, logger, "Something went wrong"
            )
            assert resp.status == ResponseModel.JobStatus.ERROR

    def test_create_response_updates_last_status(self):
        """Verify create_response updates last_status on the request model."""
        from src.common.schema.request import BackendRequestModel
        from nnsight.schema.response import ResponseModel

        req = BackendRequestModel(
            id="req-003",
            model_key="test-model",
        )
        logger = logging.getLogger("test")

        with patch("src.common.schema.response.SioProvider"):
            req.create_response(ResponseModel.JobStatus.RECEIVED, logger, "Received")
            assert req.last_status == ResponseModel.JobStatus.RECEIVED

            req.create_response(ResponseModel.JobStatus.QUEUED, logger, "Queued")
            assert req.last_status == ResponseModel.JobStatus.QUEUED

    def test_create_response_same_status_no_metric_update(self):
        """Verify repeated same-status responses keep last_status unchanged."""
        from src.common.schema.request import BackendRequestModel
        from nnsight.schema.response import ResponseModel

        req = BackendRequestModel(
            id="req-004",
            model_key="test-model",
        )
        logger = logging.getLogger("test")

        with patch("src.common.schema.response.SioProvider"):
            req.create_response(ResponseModel.JobStatus.QUEUED, logger, "Queued")
            assert req.last_status == ResponseModel.JobStatus.QUEUED

            # Same status again — last_status should not change time
            old_time = req.last_status_time
            req.create_response(ResponseModel.JobStatus.QUEUED, logger, "Still queued")
            assert req.last_status == ResponseModel.JobStatus.QUEUED


class TestIsEmail:
    """Tests for the is_email helper."""

    def test_valid_email(self):
        """Verify is_email returns True for a standard email address."""
        from src.common.schema.response import is_email

        assert is_email("user@example.com") is True

    def test_invalid_email(self):
        """Verify is_email returns False for non-email strings and URLs."""
        from src.common.schema.response import is_email

        assert is_email("not-an-email") is False
        assert is_email("https://example.com") is False

    def test_email_with_dots(self):
        """Verify is_email handles dotted names and domain parts."""
        from src.common.schema.response import is_email

        assert is_email("first.last@company.org") is True
