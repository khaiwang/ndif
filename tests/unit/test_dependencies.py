"""Unit tests for API dependency validation functions.

Tests: src/services/api/src/dependencies.py
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


class TestAuthenticateApiKey:
    """Tests for authenticate_api_key()."""

    @pytest.mark.asyncio
    async def test_dev_mode_bypasses_validation(self):
        from src.services.api.src.dependencies import authenticate_api_key
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True
        result = await authenticate_api_key("any-key")
        assert result == "any-key"

    @pytest.mark.asyncio
    async def test_dev_mode_empty_key(self):
        from src.services.api.src.dependencies import authenticate_api_key
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True
        result = await authenticate_api_key("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_no_key_store_raises_401(self):
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        with patch("src.services.api.src.dependencies.api_key_store", None):
            from src.services.api.src.dependencies import authenticate_api_key

            with pytest.raises(HTTPException) as exc_info:
                await authenticate_api_key("test-key")
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_raises_401(self):
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        mock_store = MagicMock()
        mock_store.api_key_exists.return_value = False
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import authenticate_api_key

            with pytest.raises(HTTPException) as exc_info:
                await authenticate_api_key("bad-key")
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_returns_key(self):
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        mock_store = MagicMock()
        mock_store.api_key_exists.return_value = True
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import authenticate_api_key

            result = await authenticate_api_key("valid-key")
            assert result == "valid-key"


class TestValidatePythonVersion:
    """Tests for validate_python_version()."""

    @pytest.mark.asyncio
    async def test_dev_mode_bypasses(self):
        from src.services.api.src.dependencies import validate_python_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True
        result = await validate_python_version("3.10")
        assert result == "3.10"

    @pytest.mark.asyncio
    async def test_compatible_version(self):
        from src.services.api.src.dependencies import validate_python_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_python_version = "3.11"
        result = await validate_python_version("3.12.5")
        assert result == "3.12"

    @pytest.mark.asyncio
    async def test_exact_minimum_version(self):
        from src.services.api.src.dependencies import validate_python_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_python_version = "3.12"
        result = await validate_python_version("3.12.0")
        assert result == "3.12"

    @pytest.mark.asyncio
    async def test_incompatible_version_raises_400(self):
        from src.services.api.src.dependencies import validate_python_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_python_version = "3.12"
        with pytest.raises(HTTPException) as exc_info:
            await validate_python_version("3.10.1")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_version_raises_400(self):
        from src.services.api.src.dependencies import validate_python_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        with pytest.raises(HTTPException) as exc_info:
            await validate_python_version("")
        assert exc_info.value.status_code == 400


class TestValidateNnsightVersion:
    """Tests for validate_nnsight_version()."""

    @pytest.mark.asyncio
    async def test_dev_mode_bypasses(self):
        from src.services.api.src.dependencies import validate_nnsight_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True
        result = await validate_nnsight_version("0.1.0")
        assert result == "0.1.0"

    @pytest.mark.asyncio
    async def test_compatible_version(self):
        from src.services.api.src.dependencies import validate_nnsight_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_nnsight_version = "0.5.10"
        result = await validate_nnsight_version("0.5.16")
        assert result == "0.5.16"

    @pytest.mark.asyncio
    async def test_incompatible_version_raises_400(self):
        from src.services.api.src.dependencies import validate_nnsight_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        AppConfig.min_nnsight_version = "0.5.16"
        with pytest.raises(HTTPException) as exc_info:
            await validate_nnsight_version("0.5.10")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_version_raises_400(self):
        from src.services.api.src.dependencies import validate_nnsight_version
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        with pytest.raises(HTTPException) as exc_info:
            await validate_nnsight_version("")
        assert exc_info.value.status_code == 400


class TestCheckHotswappingAccess:
    """Tests for check_hotswapping_access()."""

    @pytest.mark.asyncio
    async def test_dev_mode_returns_true(self):
        from src.services.api.src.dependencies import check_hotswapping_access
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = True
        result = await check_hotswapping_access("any-key")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_key_store_returns_false(self):
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        with patch("src.services.api.src.dependencies.api_key_store", None):
            from src.services.api.src.dependencies import check_hotswapping_access

            result = await check_hotswapping_access("test-key")
            assert result is False

    @pytest.mark.asyncio
    async def test_key_with_access(self):
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        mock_store = MagicMock()
        mock_store.key_has_hotswapping_access.return_value = True
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import check_hotswapping_access

            result = await check_hotswapping_access("premium-key")
            assert result is True

    @pytest.mark.asyncio
    async def test_key_without_access(self):
        from src.services.api.src.config import AppConfig

        AppConfig.dev_mode = False
        mock_store = MagicMock()
        mock_store.key_has_hotswapping_access.return_value = False
        with patch("src.services.api.src.dependencies.api_key_store", mock_store):
            from src.services.api.src.dependencies import check_hotswapping_access

            result = await check_hotswapping_access("free-key")
            assert result is False


class TestRequireRayConnection:
    """Tests for require_ray_connection()."""

    @pytest.mark.asyncio
    async def test_connected_passes(self):
        from src.services.api.src.dependencies import require_ray_connection

        mock_client = AsyncMock()
        mock_client.get.return_value = b"1"
        with patch("src.services.api.src.dependencies.RedisProvider") as mock_redis:
            mock_redis.async_client = mock_client
            await require_ray_connection()

    @pytest.mark.asyncio
    async def test_disconnected_raises_503(self):
        from src.services.api.src.dependencies import require_ray_connection

        mock_client = AsyncMock()
        mock_client.get.return_value = None
        with patch("src.services.api.src.dependencies.RedisProvider") as mock_redis:
            mock_redis.async_client = mock_client
            with pytest.raises(HTTPException) as exc_info:
                await require_ray_connection()
            assert exc_info.value.status_code == 503
