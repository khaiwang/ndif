"""Component tests for the API service.

Uses httpx.AsyncClient with ASGITransport to test the FastAPI app
without starting a real server. External dependencies (Redis, Ray,
MinIO) are mocked.
"""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Ensure dev mode and mock dependencies before importing app
os.environ["NDIF_DEV_MODE"] = "true"

# Mock the SocketManager and Redis before app import
patch("socketio.AsyncRedisManager", return_value=Mock()).start()
patch("redis.asyncio.Redis.from_url", return_value=AsyncMock()).start()
patch("fastapi_socketio.SocketManager", return_value=Mock()).start()


@pytest.fixture
def app():
    """Import and return the FastAPI app with mocked dependencies."""
    from src.services.api.src.app import app as fastapi_app
    return fastapi_app


@pytest.fixture
async def client(app):
    """Create an async test client."""
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestPingEndpoint:
    """Test the /ping health check endpoint."""

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self, client):
        """Verify that GET /ping returns 200 with 'pong'."""
        response = await client.get("/ping")
        assert response.status_code == 200
        assert response.json() == "pong"


class TestConnectedEndpoint:
    """Test the /connected endpoint."""

    @pytest.mark.asyncio
    async def test_connected_when_ray_up(self, client):
        """Verify that GET /connected returns 200 when Redis reports Ray is connected."""
        mock_client = AsyncMock()
        mock_client.get.return_value = b"1"
        with patch("src.services.api.src.dependencies.RedisProvider") as mock_redis:
            mock_redis.async_client = mock_client
            response = await client.get("/connected")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_disconnected_when_ray_down(self, client):
        """Verify that GET /connected returns 503 when Redis reports Ray is disconnected."""
        mock_client = AsyncMock()
        mock_client.get.return_value = None
        with patch("src.services.api.src.dependencies.RedisProvider") as mock_redis:
            mock_redis.async_client = mock_client
            response = await client.get("/connected")
            assert response.status_code == 503


class TestResponseEndpoint:
    """Test the /response/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_response_not_found(self, client):
        """Verify that GET /response/{id} raises when the response ID does not exist."""
        with patch("src.services.api.src.app.BackendResponseModel") as mock_resp_cls:
            mock_resp_cls.load.side_effect = Exception("Not found")
            # ASGITransport propagates unhandled exceptions as Python exceptions
            with pytest.raises(Exception, match="Not found"):
                await client.get("/response/nonexistent-id")

    @pytest.mark.asyncio
    async def test_response_found(self, client):
        """Verify that GET /response/{id} loads and returns a valid response."""
        from src.common.schema.response import BackendResponseModel

        mock_resp = MagicMock(spec=BackendResponseModel)
        mock_resp.model_dump.return_value = {
            "id": "valid-id",
            "session_id": None,
            "status": "RECEIVED",
            "description": "test",
            "callback": "",
        }
        # Make it JSON serializable for FastAPI response validation
        with patch("src.services.api.src.app.BackendResponseModel") as mock_resp_cls:
            mock_resp_cls.load.return_value = mock_resp
            # FastAPI will try to serialize via response_model;
            # use model_validate to produce a real instance
            mock_resp_cls.model_validate.return_value = mock_resp
            response = await client.get("/response/valid-id")
            mock_resp_cls.load.assert_called_once_with("valid-id")
