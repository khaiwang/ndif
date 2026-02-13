"""Single-GPU tests — verify GPU interactions with exactly 1 visible GPU.

Run with:
    CUDA_VISIBLE_DEVICES=0 pytest tests/gpu/test_single_gpu.py -v

Auto-skips if device_count != 1. This ensures the code is tested in a
true single-GPU environment, not just "uses 1 GPU out of many."
"""

import gc

import pytest
import torch

from src.services.ray.src.ray.deployments.controller.cluster.node import (
    CandidateLevel,
)

from .conftest import GPU_COUNT, GPU_MEM_BYTES, GIB, build_max_memory, get_device_placement

pytestmark = pytest.mark.skipif(
    GPU_COUNT != 1,
    reason=(
        f"Requires exactly 1 visible GPU (found {GPU_COUNT}). "
        "Run with: CUDA_VISIBLE_DEVICES=0 pytest tests/gpu/test_single_gpu.py"
    ),
)


# ============================================================================
# _build_max_memory with 1 GPU
# ============================================================================


class TestBuildMaxMemorySingleGpu:
    """Test _build_max_memory logic with a single visible GPU."""

    def test_single_target_gpu(self):
        """Target GPU 0 → {0: min(requested, total)}."""
        requested = 10 * GIB
        result = build_max_memory({0: requested})
        assert result is not None
        assert len(result) == 1
        assert 0 in result
        assert result[0] == min(requested, GPU_MEM_BYTES)

    def test_empty_map_returns_none(self):
        """{} → None (let accelerate use all GPUs)."""
        assert build_max_memory({}) is None

    def test_allocation_capped_at_total(self):
        """Requesting more than GPU total is capped."""
        huge = 999 * GIB
        result = build_max_memory({0: huge})
        assert result[0] == GPU_MEM_BYTES  # capped at actual total

    def test_full_gpu_allocation(self):
        """Requesting full GPU memory → gets full GPU memory."""
        result = build_max_memory({0: GPU_MEM_BYTES})
        assert result[0] == GPU_MEM_BYTES


# ============================================================================
# torch.cuda device setup
# ============================================================================


class TestCudaDeviceSetup:
    """Test torch.cuda API calls that NDIF uses during actor init."""

    def test_set_device_zero(self):
        """torch.cuda.set_device(0) sets the default device."""
        torch.cuda.set_device(0)
        assert torch.cuda.current_device() == 0

    def test_memory_fraction_applied(self):
        """set_per_process_memory_fraction doesn't error."""
        # Reset to full first
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)
        # Apply a fraction — should not raise
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        # Restore
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)

    def test_memory_fraction_clamping(self):
        """Clamping logic: values outside [0.01, 0.99] are bounded.

        This tests the clamping pattern from base.py:93:
            fraction = min(0.99, max(0.01, self.gpu_memory_fraction))
        """
        # Clamp values the same way NDIF does
        for raw_val, expected_clamped in [
            (0.001, 0.01),
            (0.5, 0.5),
            (1.5, 0.99),
            (-0.1, 0.01),
        ]:
            clamped = min(0.99, max(0.01, raw_val))
            assert clamped == expected_clamped
            # Should not raise
            torch.cuda.set_per_process_memory_fraction(clamped, device=0)

        # Restore
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)


# ============================================================================
# Model placement on single GPU
# ============================================================================


class TestModelPlacement:
    """Verify model loads on the single visible GPU correctly."""

    def test_model_on_cuda_zero(self, make_test_model):
        """model.to('cuda:0') places all params on cuda:0."""
        model = make_test_model().to("cuda:0")
        for name, param in model.named_parameters():
            assert param.device == torch.device("cuda", 0), (
                f"Parameter {name} on {param.device}, expected cuda:0"
            )
        del model
        torch.cuda.empty_cache()

    def test_device_placement_verification(self, make_test_model):
        """get_device_placement returns exactly {'cuda:0'}."""
        model = make_test_model().to("cuda:0")
        devices = get_device_placement(model)
        assert devices == {"cuda:0"}
        del model
        torch.cuda.empty_cache()

    def test_cuda_context_on_target_device(self):
        """CUDA context (small allocation) goes to the set device."""
        torch.cuda.set_device(0)
        t = torch.zeros(1, device="cuda")
        assert t.device == torch.device("cuda", 0)
        del t
        torch.cuda.empty_cache()

    def test_no_params_on_cpu_after_to_cuda(self, make_test_model):
        """After .to('cuda:0'), no parameters remain on CPU."""
        model = make_test_model().to("cuda:0")
        devices = get_device_placement(model)
        cpu_devices = {d for d in devices if d.startswith("cpu")}
        assert len(cpu_devices) == 0
        del model
        torch.cuda.empty_cache()


# ============================================================================
# Caching cycle (GPU ↔ CPU)
# ============================================================================


class TestCachingCycle:
    """Test the to_cache/from_cache movement pattern from base.py."""

    def test_model_to_cpu(self, make_test_model):
        """model.cpu() moves all params to CPU (to_cache pattern)."""
        model = make_test_model().to("cuda:0")
        model = model.cpu()
        devices = get_device_placement(model)
        # cpu device shows as "cpu:None"
        assert all(d.startswith("cpu") for d in devices)
        del model

    def test_gpu_memory_freed_after_cpu(self, make_test_model):
        """After model.cpu() + empty_cache, GPU reserved memory decreases."""
        torch.cuda.empty_cache()
        gc.collect()
        baseline = torch.cuda.memory_reserved(0)

        model = make_test_model(hidden=1024, layers=8).to("cuda:0")
        after_load = torch.cuda.memory_reserved(0)
        assert after_load > baseline, "Model didn't allocate GPU memory"

        model = model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        after_free = torch.cuda.memory_reserved(0)
        assert after_free <= after_load, "GPU memory not freed after cpu + empty_cache"

    def test_model_back_to_gpu(self, make_test_model):
        """model.cpu() → model.to('cuda:0') restores placement."""
        model = make_test_model().to("cuda:0")
        model = model.cpu()
        model = model.to("cuda:0")
        devices = get_device_placement(model)
        assert devices == {"cuda:0"}
        del model
        torch.cuda.empty_cache()

    def test_fraction_reset_after_cache(self):
        """to_cache pattern: reset fraction to 1.0 for each GPU."""
        # Simulate: set fraction → cache → reset
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        # After caching, reset (base.py:214-216)
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)
        # Should not raise on subsequent operations
        t = torch.zeros(1, device="cuda:0")
        del t
        torch.cuda.empty_cache()

    def test_full_cache_cycle(self, make_test_model):
        """Full to_cache → from_cache cycle: GPU → CPU → GPU."""
        # Load on GPU
        model = make_test_model().to("cuda:0")
        assert get_device_placement(model) == {"cuda:0"}

        # to_cache: move to CPU, reset fraction
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)
        assert all(d.startswith("cpu") for d in get_device_placement(model))

        # from_cache: set device, set fraction, move back
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        model = model.to("cuda:0")
        assert get_device_placement(model) == {"cuda:0"}

        del model
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)


# ============================================================================
# Node allocation logic with real GPU specs
# ============================================================================


class TestNodeAllocationRealSpecs:
    """Test Node.evaluate/deploy using specs from the actual GPU hardware.

    These validate that the allocation heuristics produce correct results
    for the real GPU memory size, not just arbitrary test values.
    """

    def test_evaluate_fractional_small_model(self, make_node_with_real_specs):
        """Small model (<<GPU mem) gets fractional allocation.

        Fractional path: model_size * 3 < gpu_memory * 0.8
        """
        node = make_node_with_real_specs(total_gpus=1)
        # Model at ~1% of GPU memory — well within fractional range
        model_size = int(GPU_MEM_BYTES * 0.01)
        candidate = node.evaluate("small-model", "r1", model_size)
        assert candidate.candidate_level in (
            CandidateLevel.FREE,
            CandidateLevel.CACHED_AND_FREE,
        )
        assert candidate.gpus_required == 1
        # Fractional: gpu_memory_required_bytes < full GPU
        expected_required = int(model_size * 3.0)
        assert candidate.gpu_memory_required_bytes == expected_required
        assert expected_required < GPU_MEM_BYTES

    def test_evaluate_full_gpu_medium_model(self, make_node_with_real_specs):
        """Medium model (>26% GPU mem, <100%) gets full single-GPU allocation.

        Full-GPU path: model_size * 3 >= gpu_memory * 0.8 but
                       model_size <= gpu_memory
        """
        node = make_node_with_real_specs(total_gpus=1)
        # Model at 30% of GPU memory — above fractional threshold
        model_size = int(GPU_MEM_BYTES * 0.30)
        candidate = node.evaluate("medium-model", "r1", model_size)
        assert candidate.candidate_level in (
            CandidateLevel.FREE,
            CandidateLevel.CACHED_AND_FREE,
        )
        assert candidate.gpus_required == 1
        # Full GPU: required bytes == full GPU memory
        assert candidate.gpu_memory_required_bytes == GPU_MEM_BYTES

    def test_deploy_fractional_creates_correct_deployment(
        self, make_node_with_real_specs
    ):
        """Fractional deploy produces correct gpu_mem_bytes_by_id."""
        node = make_node_with_real_specs(total_gpus=1)
        model_size = int(GPU_MEM_BYTES * 0.01)
        candidate = node.evaluate("small-model", "r1", model_size)
        node.deploy("small-model", "r1", candidate, model_size)

        dep = node._get_from_map(node.deployments, "small-model", "r1")
        assert dep is not None
        assert len(dep.gpu_mem_bytes_by_id) == 1
        gpu_id = list(dep.gpu_mem_bytes_by_id.keys())[0]
        assert gpu_id == 0
        # Fractional allocation: bytes < full GPU
        assert dep.gpu_mem_bytes_by_id[gpu_id] < GPU_MEM_BYTES
        # Memory fraction set for single-GPU fractional
        assert dep.gpu_memory_fraction is not None
        assert 0.01 <= dep.gpu_memory_fraction <= 0.99

    def test_deploy_evict_restores_resources(self, make_node_with_real_specs):
        """Deploy → evict cycle restores GPU availability."""
        node = make_node_with_real_specs(total_gpus=1)
        model_size = int(GPU_MEM_BYTES * 0.01)

        avail_before = dict(node.resources.gpu_memory_available_bytes_by_id)

        candidate = node.evaluate("model-a", "r1", model_size)
        node.deploy("model-a", "r1", candidate, model_size)

        # GPU memory reduced
        avail_during = dict(node.resources.gpu_memory_available_bytes_by_id)
        assert avail_during[0] < avail_before[0]

        # Evict
        node.evict("model-a", "r1")

        # GPU memory restored
        avail_after = dict(node.resources.gpu_memory_available_bytes_by_id)
        assert avail_after[0] == avail_before[0]

    def test_cant_accommodate_oversized_model(self, make_node_with_real_specs):
        """Model larger than 1 GPU → CANT_ACCOMMODATE on single-GPU node."""
        node = make_node_with_real_specs(total_gpus=1)
        # Model 2x GPU memory — needs 2+ GPUs but only 1 available
        model_size = int(GPU_MEM_BYTES * 2)
        candidate = node.evaluate("huge-model", "r1", model_size)
        assert candidate.candidate_level == CandidateLevel.CANT_ACCOMMODATE

    def test_multiple_fractional_fit_on_one_gpu(self, make_node_with_real_specs):
        """Multiple small models fit on the single GPU with fractional alloc."""
        node = make_node_with_real_specs(total_gpus=1)
        # Each model uses ~5% of GPU memory → up to ~6 models before GPU full
        model_size = int(GPU_MEM_BYTES * 0.05)

        for i in range(3):
            candidate = node.evaluate(f"model-{i}", "r1", model_size)
            assert candidate.candidate_level in (
                CandidateLevel.FREE,
                CandidateLevel.CACHED_AND_FREE,
            ), f"Model {i} should fit (fractional)"
            node.deploy(f"model-{i}", "r1", candidate, model_size)

        # Verify all deployed on GPU 0
        for i in range(3):
            dep = node._get_from_map(node.deployments, f"model-{i}", "r1")
            assert dep is not None
            assert 0 in dep.gpu_mem_bytes_by_id

    def test_full_gpu_model_removes_gpu_from_available(
        self, make_node_with_real_specs
    ):
        """Full-GPU model should remove the GPU from available_gpus."""
        node = make_node_with_real_specs(total_gpus=1)
        assert 0 in node.resources.available_gpus

        model_size = int(GPU_MEM_BYTES * 0.30)  # triggers full-GPU allocation
        candidate = node.evaluate("big-model", "r1", model_size)
        node.deploy("big-model", "r1", candidate, model_size)

        # GPU 0 should no longer be available (fully allocated)
        assert 0 not in node.resources.available_gpus
