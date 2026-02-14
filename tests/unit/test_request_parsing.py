"""Unit tests for request parsing pipeline.

Tests:
- BackendRequestModel.from_request() — header extraction and model construction
- validate_request() — orchestration of validators + from_request + hotswapping
- BackendRequestModel.deserialize() — body deserialization
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: build a mock FastAPI Request
# ---------------------------------------------------------------------------

def _make_request(
    headers: dict = None,
    client_host: str = "127.0.0.1",
    body: bytes = b"fake-body",
    client_is_none: bool = False,
):
    """Build a mock fastapi.Request with configurable headers and client."""
    req = MagicMock()
    req.headers = headers or {}

    if client_is_none:
        req.client = None
    else:
        req.client = MagicMock()
        req.client.host = client_host

    # body() must return a coroutine
    async def _body():
        return body

    req.body = _body
    return req


# ===========================================================================
# Class 1: TestFromRequest
# ===========================================================================


class TestFromRequest:
    """Tests for BackendRequestModel.from_request()."""

    def test_all_headers_populated(self):
        """All headers provided → all fields correctly populated."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(
            headers={
                "nnsight-model-key": "openai-community/gpt2",
                "ndif-session_id": "sess-abc",
                "ndif-api-key": "key-123",
                "ndif-callback": "https://example.com/cb",
                "nnsight-compress": "true",
                "python-version": "3.12.1",
                "nnsight-version": "0.5.16",
                "content-length": "4096",
                "user-agent": "nnsight/0.5.16",
                "ndif-timestamp": "1700000000.5",
                "ndif-request_id": "req-fixed-id",
            },
            client_host="10.0.0.1",
        )
        result = BackendRequestModel.from_request(req)

        assert result.model_key == "openai-community/gpt2"
        assert result.session_id == "sess-abc"
        assert result.api_key == "key-123"
        assert result.callback == "https://example.com/cb"
        assert result.compress is True  # Pydantic coerces "true" string to bool
        assert result.python_version == "3.12.1"
        assert result.nnsight_version == "0.5.16"
        assert result.content_length == 4096
        assert result.user_agent == "nnsight/0.5.16"
        assert result.last_status_time == 1700000000.5
        assert result.id == "req-fixed-id"
        assert result.ip_address == "10.0.0.1"

    def test_model_key_extracted(self):
        """model_key extracted from nnsight-model-key header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"nnsight-model-key": "meta-llama/Llama-2-7b"})
        result = BackendRequestModel.from_request(req)
        assert result.model_key == "meta-llama/Llama-2-7b"

    def test_session_id_extracted(self):
        """session_id from ndif-session_id header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"ndif-session_id": "my-session"})
        result = BackendRequestModel.from_request(req)
        assert result.session_id == "my-session"

    def test_api_key_extracted(self):
        """api_key from ndif-api-key header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"ndif-api-key": "secret-key"})
        result = BackendRequestModel.from_request(req)
        assert result.api_key == "secret-key"

    def test_callback_extracted(self):
        """callback from ndif-callback header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"ndif-callback": "https://hook.example.com"})
        result = BackendRequestModel.from_request(req)
        assert result.callback == "https://hook.example.com"

    def test_compress_extracted(self):
        """compress from nnsight-compress header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"nnsight-compress": "false"})
        result = BackendRequestModel.from_request(req)
        assert result.compress is False  # Pydantic coerces "false" string to bool

    def test_python_version_extracted(self):
        """python_version from python-version header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"python-version": "3.11.5"})
        result = BackendRequestModel.from_request(req)
        assert result.python_version == "3.11.5"

    def test_nnsight_version_extracted(self):
        """nnsight_version from nnsight-version header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"nnsight-version": "0.6.0"})
        result = BackendRequestModel.from_request(req)
        assert result.nnsight_version == "0.6.0"

    def test_content_length_parsed_as_int(self):
        """content_length parsed as int from content-length header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"content-length": "12345"})
        result = BackendRequestModel.from_request(req)
        assert result.content_length == 12345
        assert isinstance(result.content_length, int)

    def test_ip_address_from_client_host(self):
        """ip_address from request.client.host."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(client_host="192.168.1.100")
        result = BackendRequestModel.from_request(req)
        assert result.ip_address == "192.168.1.100"

    def test_user_agent_extracted(self):
        """user_agent from user-agent header."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"user-agent": "Mozilla/5.0"})
        result = BackendRequestModel.from_request(req)
        assert result.user_agent == "Mozilla/5.0"

    # -- Defaults when headers missing --

    def test_missing_timestamp_gives_none(self):
        """Missing ndif-timestamp → last_status_time is None."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={})
        result = BackendRequestModel.from_request(req)
        assert result.last_status_time is None

    def test_missing_request_id_generates_uuid(self):
        """Missing ndif-request_id → UUID auto-generated."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={})
        result = BackendRequestModel.from_request(req)
        # Should be a valid UUID string
        parsed = uuid.UUID(result.id)
        assert str(parsed) == result.id

    def test_provided_request_id_used(self):
        """Provided ndif-request_id header → uses that value."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"ndif-request_id": "custom-id-xyz"})
        result = BackendRequestModel.from_request(req)
        assert result.id == "custom-id-xyz"

    def test_missing_model_key_gives_none(self):
        """Missing nnsight-model-key → model_key is None."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={})
        result = BackendRequestModel.from_request(req)
        assert result.model_key is None

    def test_missing_session_id_gives_none(self):
        """Missing ndif-session_id → session_id is None."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={})
        result = BackendRequestModel.from_request(req)
        assert result.session_id is None

    def test_missing_callback_gives_empty_string(self):
        """Missing ndif-callback → callback is empty string."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={})
        result = BackendRequestModel.from_request(req)
        assert result.callback == ""

    def test_missing_content_length_gives_zero(self):
        """Missing content-length → content_length is 0."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={})
        result = BackendRequestModel.from_request(req)
        assert result.content_length == 0

    def test_client_none_gives_empty_ip(self):
        """Missing request.client (None) → ip_address is empty string."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(client_is_none=True)
        result = BackendRequestModel.from_request(req)
        assert result.ip_address == ""

    # -- model_key normalization --

    def test_model_key_main_revision_replaced(self):
        """'revision': 'main' replaced with 'revision': null."""
        from src.common.schema.request import BackendRequestModel

        key_with_main = '{"repo": "gpt2", "revision": "main"}'
        req = _make_request(headers={"nnsight-model-key": key_with_main})
        result = BackendRequestModel.from_request(req)
        assert '"revision": null' in result.model_key
        assert '"revision": "main"' not in result.model_key

    def test_model_key_non_main_revision_unchanged(self):
        """Model key without 'main' revision left unchanged."""
        from src.common.schema.request import BackendRequestModel

        key_no_main = '{"repo": "gpt2", "revision": "v1.0"}'
        req = _make_request(headers={"nnsight-model-key": key_no_main})
        result = BackendRequestModel.from_request(req)
        assert result.model_key == key_no_main

    # -- Timestamp parsing --

    def test_valid_timestamp_parsed(self):
        """Valid float timestamp string → parsed to float."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(headers={"ndif-timestamp": "1700000000.123"})
        result = BackendRequestModel.from_request(req)
        assert result.last_status_time == 1700000000.123

    # -- Request body --

    @pytest.mark.asyncio
    async def test_request_body_stored_as_coroutine(self):
        """request field stores result of request.body() (coroutine)."""
        from src.common.schema.request import BackendRequestModel

        req = _make_request(body=b"test-payload")
        result = BackendRequestModel.from_request(req)
        # The stored value is a coroutine; awaiting it gives the body bytes
        body = await result.request
        assert body == b"test-payload"


# ===========================================================================
# Class 2: TestValidateRequest
# ===========================================================================


class TestValidateRequest:
    """Tests for validate_request() dependency."""

    def _make_valid_headers(self):
        return {
            "ndif-api-key": "valid-key",
            "nnsight-version": "0.5.16",
            "python-version": "3.12.1",
            "nnsight-model-key": "openai-community/gpt2",
            "ndif-session_id": "sess-1",
            "content-length": "100",
        }

    @pytest.mark.asyncio
    async def test_happy_path_returns_populated_model(self):
        """Valid headers → returns fully populated BackendRequestModel with hotswapping."""
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True

        from src.services.api.src.dependencies import validate_request

        req = _make_request(headers=self._make_valid_headers(), client_host="10.0.0.1")
        result = await validate_request(req)

        assert result.api_key == "valid-key"
        assert result.model_key == "openai-community/gpt2"
        assert result.session_id == "sess-1"
        assert result.hotswapping is True  # dev_mode → True

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self):
        """Invalid API key → raises 401."""
        from fastapi import HTTPException
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False

        mock_store = MagicMock()
        mock_store.api_key_exists.return_value = False
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import validate_request

            req = _make_request(headers=self._make_valid_headers())
            with pytest.raises(HTTPException) as exc_info:
                await validate_request(req)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_nnsight_version_raises_400(self):
        """Invalid nnsight version → raises 400."""
        from fastapi import HTTPException
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_nnsight_version = "0.6.0"

        mock_store = MagicMock()
        mock_store.api_key_exists.return_value = True
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import validate_request

            headers = self._make_valid_headers()
            headers["nnsight-version"] = "0.5.0"
            req = _make_request(headers=headers)
            with pytest.raises(HTTPException) as exc_info:
                await validate_request(req)
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_python_version_raises_400(self):
        """Invalid python version → raises 400."""
        from fastapi import HTTPException
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_python_version = "3.12"
        AppConfig.min_nnsight_version = "0.5.0"

        mock_store = MagicMock()
        mock_store.api_key_exists.return_value = True
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import validate_request

            headers = self._make_valid_headers()
            headers["python-version"] = "3.10.1"
            req = _make_request(headers=headers)
            with pytest.raises(HTTPException) as exc_info:
                await validate_request(req)
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_dev_mode_bypasses_validation(self):
        """Dev mode bypasses all validation, still creates the model."""
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True

        from src.services.api.src.dependencies import validate_request

        headers = {
            "ndif-api-key": "",
            "nnsight-version": "",
            "python-version": "",
            "nnsight-model-key": "test/model",
        }
        req = _make_request(headers=headers)
        result = await validate_request(req)

        assert result.model_key == "test/model"
        assert result.hotswapping is True  # dev_mode → True

    @pytest.mark.asyncio
    async def test_hotswapping_from_check(self):
        """Hotswapping populated from check_hotswapping_access result."""
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_nnsight_version = "0.5.0"
        AppConfig.min_python_version = "3.11"

        mock_store = MagicMock()
        mock_store.api_key_exists.return_value = True
        mock_store.key_has_hotswapping_access.return_value = False
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import validate_request

            req = _make_request(headers=self._make_valid_headers())
            result = await validate_request(req)
            assert result.hotswapping is False

    @pytest.mark.asyncio
    async def test_validators_called_in_order(self):
        """Validators are called in order: API key, nnsight, python."""
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True

        call_order = []

        async def mock_auth(key):
            call_order.append("auth")
            return key

        async def mock_nnsight(ver):
            call_order.append("nnsight")
            return ver

        async def mock_python(ver):
            call_order.append("python")
            return ver

        async def mock_hotswap(key):
            call_order.append("hotswap")
            return False

        with patch("src.services.api.src.dependencies.authenticate_api_key", mock_auth), \
             patch("src.services.api.src.dependencies.validate_nnsight_version", mock_nnsight), \
             patch("src.services.api.src.dependencies.validate_python_version", mock_python), \
             patch("src.services.api.src.dependencies.check_hotswapping_access", mock_hotswap):
            from src.services.api.src.dependencies import validate_request

            req = _make_request(headers=self._make_valid_headers())
            await validate_request(req)

        assert call_order == ["auth", "nnsight", "python", "hotswap"]


# ===========================================================================
# Class 3: TestDeserialize
# ===========================================================================


class TestDeserialize:
    """Tests for BackendRequestModel.deserialize()."""

    def _setup_ray_objectref(self):
        """Set ray.ObjectRef to a real class so isinstance() works."""
        import ray

        class _FakeObjectRef:
            pass

        ray.ObjectRef = _FakeObjectRef
        return _FakeObjectRef

    def test_bytes_request_calls_deserialize(self):
        """Bytes request → calls RequestModel.deserialize() with body and compress flag."""
        from src.common.schema.request import BackendRequestModel
        from nnsight.schema.request import RequestModel

        self._setup_ray_objectref()

        body = b"serialized-data"
        model = BackendRequestModel(id="req-1", request=body, compress=True)

        mock_result = MagicMock()
        RequestModel.deserialize = MagicMock(return_value=mock_result)

        result = model.deserialize()
        RequestModel.deserialize.assert_called_once_with(body, None, True)
        assert result is mock_result

    def test_ray_object_ref_calls_ray_get(self):
        """Ray ObjectRef → calls ray.get() first, then deserializes."""
        import ray
        from src.common.schema.request import BackendRequestModel
        from nnsight.schema.request import RequestModel

        FakeRef = self._setup_ray_objectref()
        ref = FakeRef()

        resolved_bytes = b"resolved-data"
        ray.get = MagicMock(return_value=resolved_bytes)

        mock_result = MagicMock()
        RequestModel.deserialize = MagicMock(return_value=mock_result)

        model = BackendRequestModel(id="req-2", request=ref, compress=False)
        result = model.deserialize()

        ray.get.assert_called_once_with(ref)
        RequestModel.deserialize.assert_called_once_with(resolved_bytes, None, False)
        assert result is mock_result

    def test_persistent_objects_passed_through(self):
        """Passes persistent_objects through to RequestModel.deserialize()."""
        from src.common.schema.request import BackendRequestModel
        from nnsight.schema.request import RequestModel

        self._setup_ray_objectref()

        body = b"data"
        persistent = {"key": "value"}
        mock_result = MagicMock()
        RequestModel.deserialize = MagicMock(return_value=mock_result)

        model = BackendRequestModel(id="req-3", request=body, compress=True)
        result = model.deserialize(persistent_objects=persistent)

        RequestModel.deserialize.assert_called_once_with(body, persistent, True)
        assert result is mock_result

    def test_compress_flag_forwarded(self):
        """Compress flag forwarded correctly."""
        from src.common.schema.request import BackendRequestModel
        from nnsight.schema.request import RequestModel

        self._setup_ray_objectref()

        body = b"data"
        mock_result = MagicMock()
        RequestModel.deserialize = MagicMock(return_value=mock_result)

        model = BackendRequestModel(id="req-4", request=body, compress=False)
        model.deserialize()

        RequestModel.deserialize.assert_called_once_with(body, None, False)
