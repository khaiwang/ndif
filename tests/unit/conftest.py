"""Shared fixtures for unit tests.

All module-level mocking is handled by the root tests/conftest.py.
This file provides unit-test-specific factory fixtures.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# Dispatcher patch targets
# ---------------------------------------------------------------------------

_DISPATCHER_MODULE = "src.services.api.src.queue.dispatcher"

DISPATCHER_PATCH_RAY = f"{_DISPATCHER_MODULE}.RayProvider"
DISPATCHER_PATCH_REDIS = f"{_DISPATCHER_MODULE}.RedisProvider"
DISPATCHER_PATCH_OBJ = f"{_DISPATCHER_MODULE}.ObjectStoreProvider"
DISPATCHER_PATCH_LOGGER = f"{_DISPATCHER_MODULE}.set_logger"
DISPATCHER_PATCH_PATCH = f"{_DISPATCHER_MODULE}.patch"
DISPATCHER_PATCH_PROCESSOR = f"{_DISPATCHER_MODULE}.Processor"
DISPATCHER_PATCH_CONTROLLER = f"{_DISPATCHER_MODULE}.controller_handle"
DISPATCHER_PATCH_SUBMIT = f"{_DISPATCHER_MODULE}.submit"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis_sync():
    """Mocked synchronous Redis client."""
    from tests.conftest import _mock_redis_sync
    _mock_redis_sync.reset_mock()
    return _mock_redis_sync


@pytest.fixture
def mock_redis_async():
    """Mocked async Redis client."""
    from tests.conftest import _mock_redis_async
    _mock_redis_async.reset_mock()
    return _mock_redis_async


@pytest.fixture
def make_resources():
    """Factory for creating Resources instances with sensible defaults."""
    from src.services.ray.src.ray.deployments.controller.cluster.node import Resources

    def _make(
        total_gpus=4,
        gpu_type="A100",
        gpu_memory_bytes=80 * 1024**3,  # 80 GiB
        cpu_memory_bytes=256 * 1024**3,  # 256 GiB
        available_cpu_memory_bytes=None,
        available_gpus=None,
        gpu_memory_available_bytes_by_id=None,
    ):
        if available_cpu_memory_bytes is None:
            available_cpu_memory_bytes = cpu_memory_bytes
        if available_gpus is None:
            available_gpus = list(range(total_gpus))
        if gpu_memory_available_bytes_by_id is None:
            gpu_memory_available_bytes_by_id = {
                gpu_id: gpu_memory_bytes for gpu_id in range(total_gpus)
            }

        return Resources(
            total_gpus=total_gpus,
            gpu_type=gpu_type,
            gpu_memory_bytes=gpu_memory_bytes,
            cpu_memory_bytes=cpu_memory_bytes,
            available_cpu_memory_bytes=available_cpu_memory_bytes,
            available_gpus=available_gpus,
            gpu_memory_available_bytes_by_id=gpu_memory_available_bytes_by_id,
        )

    return _make


@pytest.fixture
def make_node(make_resources):
    """Factory for creating Node instances."""
    from src.services.ray.src.ray.deployments.controller.cluster.node import Node

    def _make(
        node_id="node-1",
        name="test-node",
        resources=None,
        minimum_deployment_time_seconds=None,
        **resource_kwargs,
    ):
        if resources is None:
            resources = make_resources(**resource_kwargs)
        return Node(node_id, name, resources, minimum_deployment_time_seconds)

    return _make


@pytest.fixture
def make_deployment():
    """Factory for creating Deployment instances."""
    from src.services.ray.src.ray.deployments.controller.cluster.deployment import (
        Deployment,
        DeploymentLevel,
    )

    def _make(
        model_key="test-model",
        replica_id="replica-1",
        deployment_level=DeploymentLevel.HOT,
        gpu_mem_bytes_by_id=None,
        gpu_memory_fraction=None,
        size_bytes=10 * 1024**3,
        dedicated=False,
        node_id="node-1",
    ):
        if gpu_mem_bytes_by_id is None:
            gpu_mem_bytes_by_id = {0: 80 * 1024**3}
        return Deployment(
            model_key=model_key,
            replica_id=replica_id,
            deployment_level=deployment_level,
            gpu_mem_bytes_by_id=gpu_mem_bytes_by_id,
            gpu_memory_fraction=gpu_memory_fraction,
            size_bytes=size_bytes,
            dedicated=dedicated,
            node_id=node_id,
        )

    return _make


@pytest.fixture
def make_request():
    """Factory for creating mock BackendRequestModel instances."""

    def _make(
        request_id="req-001",
        model_key="meta-llama/Llama-2-7b",
        session_id="session-1",
        api_key="test-key",
        hotswapping=False,
    ):
        req = MagicMock()
        req.id = request_id
        req.model_key = model_key
        req.session_id = session_id
        req.api_key = api_key
        req.hotswapping = hotswapping
        req.create_response.return_value = MagicMock()
        req.create_response.return_value.respond.return_value = None
        return req

    return _make


@pytest.fixture
def make_processor(make_request):
    """Factory for creating Processor instances with mocked dependencies."""
    from src.services.api.src.queue.processor import Processor

    def _make(
        model_key="meta-llama/Llama-2-7b",
        replica_count=1,
    ):
        eviction_queue = asyncio.Queue()
        error_queue = asyncio.Queue()
        processor = Processor(
            model_key=model_key,
            eviction_queue=eviction_queue,
            error_queue=error_queue,
            replica_count=replica_count,
        )
        return processor

    return _make


# ---------------------------------------------------------------------------
# Dispatcher fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_event():
    """Factory for building Redis-stream event_data dicts with bytes keys/values."""

    def _make(**overrides) -> dict:
        defaults = {
            "event_type": "",
            "model_key": "meta-llama/Llama-2-7b",
            "replicas": "",
            "response_key": "response:123",
            "request_id": "req-001",
            "replica_id": "",
        }
        defaults.update(overrides)
        return {k.encode(): v.encode() for k, v in defaults.items()}

    return _make


@pytest.fixture
def dispatcher_deps():
    """Patch all external dependencies and yield a dict of mocks.

    The returned dict contains:
        ray, redis, obj, logger, patch_fn
    and the ready-to-use ``dispatcher`` instance.
    """
    with (
        patch(DISPATCHER_PATCH_RAY) as mock_ray,
        patch(DISPATCHER_PATCH_REDIS) as mock_redis,
        patch(DISPATCHER_PATCH_OBJ) as mock_obj,
        patch(DISPATCHER_PATCH_LOGGER) as mock_log,
        patch(DISPATCHER_PATCH_PATCH) as mock_patch_fn,
    ):
        # Make connected() return True on first call so the while-loop
        # in connect() exits immediately.
        mock_ray.connected.return_value = True
        mock_log.return_value = MagicMock()

        # Provide async_client and sync_client stubs.
        mock_redis.sync_client = MagicMock()
        mock_redis.async_client = AsyncMock()

        from src.services.api.src.queue.dispatcher import Dispatcher

        dispatcher = Dispatcher()

        yield {
            "dispatcher": dispatcher,
            "ray": mock_ray,
            "redis": mock_redis,
            "obj": mock_obj,
            "logger": mock_log,
            "patch_fn": mock_patch_fn,
        }


@pytest.fixture
def dispatcher(dispatcher_deps):
    """Shorthand: return just the dispatcher instance."""
    return dispatcher_deps["dispatcher"]


@pytest.fixture
def mock_ray(dispatcher_deps):
    return dispatcher_deps["ray"]


@pytest.fixture
def mock_redis(dispatcher_deps):
    return dispatcher_deps["redis"]
