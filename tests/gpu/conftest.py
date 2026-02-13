"""GPU test fixtures — auto-detection, model factories, real hardware specs.

GPU tests are split into two files that must run in separate pytest sessions:

    # Single-GPU tests (exactly 1 visible GPU):
    CUDA_VISIBLE_DEVICES=0 pytest tests/gpu/test_single_gpu.py -v

    # Multi-GPU tests (2+ visible GPUs):
    pytest tests/gpu/test_multi_gpu.py -v

All tests auto-skip if the GPU environment doesn't match their requirements.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# GPU detection (evaluated once at import time)
# ---------------------------------------------------------------------------

HAS_CUDA = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if HAS_CUDA else 0
if HAS_CUDA:
    GPU_MEM_BYTES = torch.cuda.get_device_properties(0).total_memory
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    GPU_MEM_BYTES = 0
    GPU_NAME = None

GIB = 1024**3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpu_specs():
    """Real GPU specs from the current machine."""
    return {
        "count": GPU_COUNT,
        "memory_bytes": GPU_MEM_BYTES,
        "gpu_name": GPU_NAME,
    }


@pytest.fixture
def make_test_model():
    """Factory for simple models used in GPU placement tests.

    Default: 4-layer MLP (~1MB at float32). Lightweight enough to load
    instantly, large enough to verify device placement.
    """

    def _make(hidden=256, layers=4, dtype=torch.float32):
        model = torch.nn.Sequential(
            *[torch.nn.Linear(hidden, hidden) for _ in range(layers)]
        )
        return model.to(dtype)

    return _make


@pytest.fixture
def make_node_with_real_specs():
    """Factory for Node instances using real GPU specs from this machine."""
    from src.services.ray.src.ray.deployments.controller.cluster.node import (
        Node,
        Resources,
    )

    def _make(
        node_id="node-0",
        name="test-node",
        total_gpus=None,
        gpu_memory_bytes=None,
        cpu_memory_bytes=None,
    ):
        if total_gpus is None:
            total_gpus = GPU_COUNT
        if gpu_memory_bytes is None:
            gpu_memory_bytes = GPU_MEM_BYTES
        if cpu_memory_bytes is None:
            cpu_memory_bytes = 64 * GIB  # reasonable default

        resources = Resources(
            total_gpus=total_gpus,
            gpu_type=GPU_NAME or "unknown",
            gpu_memory_bytes=gpu_memory_bytes,
            cpu_memory_bytes=cpu_memory_bytes,
            available_cpu_memory_bytes=cpu_memory_bytes,
            available_gpus=list(range(total_gpus)),
            gpu_memory_available_bytes_by_id={
                i: gpu_memory_bytes for i in range(total_gpus)
            },
        )
        return Node(node_id, name, resources)

    return _make


# ---------------------------------------------------------------------------
# Standalone reimplementations of BaseModelDeployment methods
# ---------------------------------------------------------------------------
# These replicate the exact logic from base.py so we can test them without
# instantiating the full ModelActor (which requires Ray, providers, etc.).


def build_max_memory(gpu_mem_bytes_by_id):
    """Standalone version of BaseModelDeployment._build_max_memory().

    See: src/services/ray/src/ray/deployments/modeling/base.py:125-143
    """
    if not gpu_mem_bytes_by_id or not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    max_memory = {}
    for i in range(num_gpus):
        if i in gpu_mem_bytes_by_id:
            total = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = min(gpu_mem_bytes_by_id[i], total)
        else:
            max_memory[i] = 0
    return max_memory


def get_device_placement(module):
    """Get the set of devices a module's parameters reside on.

    See: src/services/ray/src/ray/deployments/modeling/base.py:145-167
    """
    devices = set()
    for param in module.parameters():
        devices.add(f"{param.device.type}:{param.device.index}")
    return devices
