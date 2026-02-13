"""Pytest configuration for NDIF tests.

Sets up shared module mocking and markers for all test layers.
"""

import os
import sys
from enum import Enum
from typing import ClassVar, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

# ===========================================================================
# Environment variables — set BEFORE any ndif code is imported
# ===========================================================================
os.environ.setdefault("NDIF_BROKER_URL", "redis://localhost:6379")
os.environ.setdefault("NDIF_DEV_MODE", "true")
os.environ.setdefault("NDIF_API_URL", "http://localhost:5001")
os.environ.setdefault("NDIF_RAY_ADDRESS", "ray://localhost:10001")

# ===========================================================================
# Module-level mocks — prevent external service connections on import
# ===========================================================================

# ---- Redis ----
_mock_redis_sync = MagicMock()
_mock_redis_async = AsyncMock()
patch("redis.Redis.from_url", return_value=_mock_redis_sync).start()
patch("redis.asyncio.Redis.from_url", return_value=_mock_redis_async).start()

# ---- Ray (not installed in test env) ----
_mock_ray = MagicMock()
_ray_modules = {
    "ray": _mock_ray,
    "ray.util": MagicMock(),
    "ray.util.client": MagicMock(),
    "ray.util.client.ray": MagicMock(),
    "ray.util.client.common": MagicMock(),
    "ray.util.state": MagicMock(),
    "ray._private": MagicMock(),
    "ray._private.services": MagicMock(),
    "ray._private.state": MagicMock(),
    "ray._raylet": MagicMock(),
    "ray.serve": MagicMock(),
}
patch.dict("sys.modules", _ray_modules).start()

# ---- nnsight (circular import with accelerate in this env) ----
# ResponseModel must be a real pydantic BaseModel for metaclass compatibility
# with BackendResponseModel(ResponseModel, ObjectStorageMixin, TelemetryMixin).


class _JobStatus(str, Enum):
    RECEIVED = "RECEIVED"
    QUEUED = "QUEUED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    LOG = "LOG"


class _ResponseModel(BaseModel):
    """Stub for nnsight.schema.response.ResponseModel."""
    JobStatus: ClassVar = _JobStatus
    id: str = ""
    session_id: Optional[str] = None
    status: _JobStatus = _JobStatus.RECEIVED
    description: str = ""

    def pickle(self):
        """Stub for nnsight ResponseModel.pickle()."""
        import pickle as _pickle
        return _pickle.dumps(self.model_dump())


class _RequestModel(BaseModel):
    """Stub for nnsight.schema.request.RequestModel."""
    id: str = ""
    model_key: str = ""


_mock_nnsight = MagicMock()
_mock_nnsight.__version__ = "0.5.15"
_mock_nnsight_schema_response = MagicMock()
_mock_nnsight_schema_response.ResponseModel = _ResponseModel
_mock_nnsight_schema_request = MagicMock()
_mock_nnsight_schema_request.RequestModel = _RequestModel
_nnsight_modules = {
    "nnsight": _mock_nnsight,
    "nnsight.modeling": MagicMock(),
    "nnsight.modeling.mixins": MagicMock(),
    "nnsight.modeling.mixins.remoteable": MagicMock(),
    "nnsight.modeling.language": MagicMock(),
    "nnsight.schema": MagicMock(),
    "nnsight.schema.request": _mock_nnsight_schema_request,
    "nnsight.schema.response": _mock_nnsight_schema_response,
    "nnsight.intervention": MagicMock(),
    "nnsight.intervention.backends": MagicMock(),
    "nnsight.intervention.backends.remote": MagicMock(),
    "nnsight.intervention.tracing": MagicMock(),
    "nnsight.intervention.tracing.globals": MagicMock(),
    "nnsight.intervention.tracing.tracer": MagicMock(),
    "nnsight.intervention.tracing.util": MagicMock(),
    "nnsight.util": MagicMock(),
}
patch.dict("sys.modules", _nnsight_modules).start()

# ---- transformers (version mismatch — missing _get_device_map) ----
_mock_transformers = MagicMock()
_transformers_modules = {
    "transformers": _mock_transformers,
    "transformers.modeling_utils": MagicMock(),
    "transformers.utils": MagicMock(),
}
patch.dict("sys.modules", _transformers_modules).start()

# ---- psycopg2 (not installed in test env) ----
sys.modules.setdefault("psycopg2", MagicMock())

# ---- boto3 — prevent real S3 connections ----
patch("boto3.client", return_value=Mock()).start()


# ===========================================================================
# Pytest hooks and shared fixtures
# ===========================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--ndif-host",
        action="store",
        default="http://localhost:5001",
        help="NDIF API host URL",
    )
    parser.addoption(
        "--run-remote",
        action="store_true",
        default=False,
        help="Run tests that require a remote NDIF server",
    )


@pytest.fixture(scope="session")
def ndif_host(request):
    """Get the NDIF host from command line or environment."""
    return request.config.getoption("--ndif-host") or os.environ.get(
        "NDIF_HOST", "http://localhost:5001"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-apply markers based on test directory and skip remote tests."""

    # Auto-apply markers by directory
    for item in items:
        test_path = str(item.fspath)
        if "/tests/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/component/" in test_path:
            item.add_marker(pytest.mark.component)
        elif "/tests/contract/" in test_path:
            item.add_marker(pytest.mark.contract)
        else:
            item.add_marker(pytest.mark.integration)

    # Skip remote tests unless --run-remote is specified
    if config.getoption("--run-remote"):
        return

    skip_remote = pytest.mark.skip(reason="Need --run-remote option to run")

    # Test classes that require remote NDIF server
    remote_test_classes = {
        "TestAllowedOperations",
        "TestBlockedOperations",
        "TestBasicTracing",
        "TestGeneration",
        "TestActivationModification",
        "TestGradients",
        "TestSessions",
        "TestCaching",
        "TestInvokers",
        "TestIteration",
        "TestAdhocModules",
        "TestEdgeCases",
        "TestPrintAndDebug",
    }

    for item in items:
        if item.cls and item.cls.__name__ in remote_test_classes:
            item.add_marker(skip_remote)
