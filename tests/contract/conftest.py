"""Contract test fixtures.

Provides shared fixtures needed by contract tests that exercise
both the API and Ray sides of the boundary.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def make_resources():
    """Factory for creating Resources instances with sensible defaults."""
    from src.services.ray.src.ray.deployments.controller.cluster.node import Resources

    def _make(
        total_gpus=4,
        gpu_type="A100",
        gpu_memory_bytes=80 * 1024**3,
        cpu_memory_bytes=256 * 1024**3,
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
