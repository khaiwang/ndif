"""Multi-GPU tests — verify GPU interactions with 2+ visible GPUs.

Run with:
    pytest tests/gpu/test_multi_gpu.py -v

Auto-skips if device_count < 2.
"""

import gc

import pytest
import torch

from src.services.ray.src.ray.deployments.controller.cluster.node import (
    CandidateLevel,
)

from .conftest import GPU_COUNT, GPU_MEM_BYTES, GIB, build_max_memory, get_device_placement

pytestmark = pytest.mark.skipif(
    GPU_COUNT < 2,
    reason=f"Requires 2+ visible GPUs (found {GPU_COUNT})",
)


# ============================================================================
# _build_max_memory with multiple GPUs
# ============================================================================


class TestBuildMaxMemoryMultiGpu:
    """Test _build_max_memory logic with multiple visible GPUs."""

    def test_multi_target_gpus(self):
        """Target GPUs 0,1 → {0: X, 1: X, others: 0}."""
        requested = 10 * GIB
        result = build_max_memory({0: requested, 1: requested})
        assert result is not None
        assert len(result) == GPU_COUNT
        # Target GPUs get their allocation (capped at total)
        for gpu_id in [0, 1]:
            assert result[gpu_id] == min(requested, GPU_MEM_BYTES)
        # Non-target GPUs get 0
        for gpu_id in range(2, GPU_COUNT):
            assert result[gpu_id] == 0

    def test_non_target_gpus_get_zero(self):
        """Non-target GPUs get exactly 0 bytes — no leaks."""
        result = build_max_memory({0: GPU_MEM_BYTES})
        for gpu_id in range(1, GPU_COUNT):
            assert result[gpu_id] == 0, (
                f"GPU {gpu_id} should get 0 bytes but got {result[gpu_id]}"
            )

    def test_single_non_zero_target(self):
        """Targeting only GPU 1 (not GPU 0) works correctly."""
        result = build_max_memory({1: 10 * GIB})
        assert result[0] == 0, "GPU 0 should get 0 when not targeted"
        assert result[1] == min(10 * GIB, GPU_MEM_BYTES)

    def test_all_gpus_targeted(self):
        """All GPUs targeted → all get allocation."""
        target = {i: GPU_MEM_BYTES for i in range(GPU_COUNT)}
        result = build_max_memory(target)
        for gpu_id in range(GPU_COUNT):
            assert result[gpu_id] == GPU_MEM_BYTES


# ============================================================================
# torch.cuda device setup with multiple GPUs
# ============================================================================


class TestCudaDeviceSetupMultiGpu:
    """Test torch.cuda API calls with multiple GPUs."""

    def test_set_device_non_zero(self):
        """set_device(1) makes GPU 1 the default."""
        torch.cuda.set_device(1)
        assert torch.cuda.current_device() == 1
        # Restore
        torch.cuda.set_device(0)

    def test_current_device_after_set(self):
        """After set_device(N), new tensors go to GPU N."""
        torch.cuda.set_device(1)
        t = torch.zeros(1, device="cuda")
        assert t.device == torch.device("cuda", 1)
        del t
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    def test_memory_fraction_per_gpu(self):
        """Can set different memory fractions on different GPUs."""
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        torch.cuda.set_per_process_memory_fraction(0.8, device=1)
        # Restore
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)
        torch.cuda.set_per_process_memory_fraction(1.0, device=1)


# ============================================================================
# Model placement on multiple GPUs
# ============================================================================


class TestModelPlacementMultiGpu:
    """Verify model placement on non-default and multiple GPUs."""

    def test_model_on_non_default_gpu(self, make_test_model):
        """model.to('cuda:1') places all params on GPU 1, not GPU 0."""
        model = make_test_model().to("cuda:1")
        devices = get_device_placement(model)
        assert devices == {"cuda:1"}, (
            f"Expected all params on cuda:1, got {devices}"
        )
        del model
        torch.cuda.empty_cache()

    def test_model_split_across_gpus(self, make_test_model):
        """Manually split model layers across GPUs 0 and 1."""
        model = make_test_model(layers=4)
        # First half on GPU 0, second half on GPU 1
        for i in range(2):
            model[i] = model[i].to("cuda:0")
        for i in range(2, 4):
            model[i] = model[i].to("cuda:1")

        devices = get_device_placement(model)
        assert "cuda:0" in devices, "Expected some params on cuda:0"
        assert "cuda:1" in devices, "Expected some params on cuda:1"
        # No other devices
        assert devices == {"cuda:0", "cuda:1"}
        del model
        torch.cuda.empty_cache()

    def test_device_placement_detects_multi_device(self, make_test_model):
        """get_device_placement correctly identifies multi-device spread."""
        model = make_test_model(layers=2)
        model[0] = model[0].to("cuda:0")
        model[1] = model[1].to("cuda:1")

        devices = get_device_placement(model)
        assert len(devices) == 2
        assert "cuda:0" in devices
        assert "cuda:1" in devices
        del model
        torch.cuda.empty_cache()

    def test_set_device_does_not_move_existing_params(self, make_test_model):
        """set_device(1) doesn't move existing params from GPU 0."""
        model = make_test_model().to("cuda:0")
        torch.cuda.set_device(1)
        # Model should still be on GPU 0
        assert get_device_placement(model) == {"cuda:0"}
        del model
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)


# ============================================================================
# Caching cycle with multiple GPUs
# ============================================================================


class TestCachingCycleMultiGpu:
    """Test GPU ↔ CPU movement with multi-GPU scenarios."""

    def test_model_cpu_from_non_zero_gpu(self, make_test_model):
        """Model on cuda:1 → .cpu() moves all params to CPU."""
        model = make_test_model().to("cuda:1")
        model = model.cpu()
        devices = get_device_placement(model)
        assert all(d.startswith("cpu") for d in devices)
        del model

    def test_repin_to_different_gpu(self, make_test_model):
        """GPU 0 → CPU → GPU 1: from_cache re-pinning pattern."""
        model = make_test_model().to("cuda:0")
        assert get_device_placement(model) == {"cuda:0"}

        # to_cache
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # from_cache to different GPU
        torch.cuda.set_device(1)
        model = model.to("cuda:1")
        assert get_device_placement(model) == {"cuda:1"}

        del model
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    def test_fraction_reset_all_gpus(self):
        """to_cache pattern: reset fraction to 1.0 on ALL allocated GPUs.

        base.py:214-216 iterates gpu_mem_bytes_by_id and resets each.
        """
        # Simulate multi-GPU allocation
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        torch.cuda.set_per_process_memory_fraction(0.5, device=1)

        # to_cache: reset all
        gpu_mem_bytes_by_id = {0: GPU_MEM_BYTES, 1: GPU_MEM_BYTES}
        for gpu_id in gpu_mem_bytes_by_id:
            torch.cuda.set_per_process_memory_fraction(1.0, device=gpu_id)

        # Verify: can still allocate on both GPUs
        t0 = torch.zeros(1, device="cuda:0")
        t1 = torch.zeros(1, device="cuda:1")
        del t0, t1
        torch.cuda.empty_cache()

    def test_from_cache_new_gpu_allocation(self, make_test_model):
        """Full from_cache flow: set device, set fractions, move model.

        Tests the pattern from base.py:221-286.
        """
        model = make_test_model().to("cuda:0")

        # to_cache
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)

        # from_cache to GPU 1
        new_gpu_map = {1: GPU_MEM_BYTES}
        torch.cuda.set_device(list(new_gpu_map.keys())[0])

        for gpu_id, bytes_cap in new_gpu_map.items():
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            fraction = min(0.99, max(0.01, bytes_cap / total))
            torch.cuda.set_per_process_memory_fraction(fraction, device=gpu_id)

        model = model.to("cuda:1")
        assert get_device_placement(model) == {"cuda:1"}

        del model
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device=1)
        torch.cuda.set_device(0)

    def test_multi_gpu_to_cache_and_back(self, make_test_model):
        """Model split across GPUs 0,1 → CPU → back to GPUs 0,1."""
        model = make_test_model(layers=4)
        for i in range(2):
            model[i] = model[i].to("cuda:0")
        for i in range(2, 4):
            model[i] = model[i].to("cuda:1")
        assert get_device_placement(model) == {"cuda:0", "cuda:1"}

        # to_cache
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        assert all(d.startswith("cpu") for d in get_device_placement(model))

        # from_cache: put back on same GPUs
        for i in range(2):
            model[i] = model[i].to("cuda:0")
        for i in range(2, 4):
            model[i] = model[i].to("cuda:1")
        assert get_device_placement(model) == {"cuda:0", "cuda:1"}

        del model
        torch.cuda.empty_cache()


# ============================================================================
# Node allocation logic with real multi-GPU specs
# ============================================================================


class TestNodeAllocationMultiGpu:
    """Test Node.evaluate/deploy for multi-GPU scenarios."""

    def test_evaluate_multi_gpu_model(self, make_node_with_real_specs):
        """Model larger than 1 GPU → multi-GPU allocation."""
        node = make_node_with_real_specs()
        # Model 1.5x GPU memory → needs 2 GPUs
        model_size = int(GPU_MEM_BYTES * 1.5)
        candidate = node.evaluate("large-model", "r1", model_size)
        assert candidate.candidate_level in (
            CandidateLevel.FREE,
            CandidateLevel.CACHED_AND_FREE,
        )
        assert candidate.gpus_required >= 2

    def test_deploy_assigns_multiple_gpus(self, make_node_with_real_specs):
        """Multi-GPU deploy allocates correct number of GPUs."""
        node = make_node_with_real_specs()
        model_size = int(GPU_MEM_BYTES * 1.5)
        candidate = node.evaluate("large-model", "r1", model_size)
        node.deploy("large-model", "r1", candidate, model_size)

        dep = node._get_from_map(node.deployments, "large-model", "r1")
        assert dep is not None
        assert len(dep.gpu_mem_bytes_by_id) >= 2
        # Each GPU gets full allocation
        for gpu_id, mem in dep.gpu_mem_bytes_by_id.items():
            assert mem == GPU_MEM_BYTES
        # No memory fraction for multi-GPU (full GPUs)
        assert dep.gpu_memory_fraction is None

    def test_evict_multi_gpu_frees_all(self, make_node_with_real_specs):
        """Evicting multi-GPU deployment frees ALL GPUs."""
        node = make_node_with_real_specs()
        model_size = int(GPU_MEM_BYTES * 1.5)
        candidate = node.evaluate("large-model", "r1", model_size)
        node.deploy("large-model", "r1", candidate, model_size)

        dep = node._get_from_map(node.deployments, "large-model", "r1")
        gpu_ids_used = list(dep.gpu_mem_bytes_by_id.keys())
        assert len(gpu_ids_used) >= 2

        # Remember available GPUs before eviction
        avail_before = set(node.resources.available_gpus)

        node.evict("large-model", "r1")

        # All used GPUs should be back in available_gpus
        avail_after = set(node.resources.available_gpus)
        for gpu_id in gpu_ids_used:
            assert gpu_id in avail_after, (
                f"GPU {gpu_id} not restored after multi-GPU evict"
            )
        # Available GPUs increased
        assert len(avail_after) > len(avail_before)

    def test_multi_gpu_and_fractional_coexist(self, make_node_with_real_specs):
        """Node can host both a multi-GPU model and fractional models.

        Only possible if the node has enough GPUs for both.
        """
        if GPU_COUNT < 3:
            pytest.skip("Need 3+ GPUs for coexistence test")

        node = make_node_with_real_specs()

        # Deploy a multi-GPU model (uses 2 GPUs)
        large_model_size = int(GPU_MEM_BYTES * 1.5)
        c1 = node.evaluate("large-model", "r1", large_model_size)
        assert c1.gpus_required >= 2
        node.deploy("large-model", "r1", c1, large_model_size)

        # Deploy a small fractional model on the remaining GPU(s)
        small_model_size = int(GPU_MEM_BYTES * 0.01)
        c2 = node.evaluate("small-model", "r1", small_model_size)
        assert c2.candidate_level in (
            CandidateLevel.FREE,
            CandidateLevel.CACHED_AND_FREE,
        ), "Small model should fit on remaining GPU"
        node.deploy("small-model", "r1", c2, small_model_size)

        # Both should be deployed
        assert node._get_from_map(node.deployments, "large-model", "r1") is not None
        assert node._get_from_map(node.deployments, "small-model", "r1") is not None

        # Small model should be on a different GPU than the large model
        large_dep = node._get_from_map(node.deployments, "large-model", "r1")
        small_dep = node._get_from_map(node.deployments, "small-model", "r1")
        large_gpus = set(large_dep.gpu_mem_bytes_by_id.keys())
        small_gpus = set(small_dep.gpu_mem_bytes_by_id.keys())
        assert large_gpus.isdisjoint(small_gpus), (
            f"Multi-GPU ({large_gpus}) and fractional ({small_gpus}) should not overlap"
        )

    def test_cant_accommodate_exceeds_all_gpus(self, make_node_with_real_specs):
        """Model too large for all GPUs combined → CANT_ACCOMMODATE."""
        node = make_node_with_real_specs()
        model_size = int(GPU_MEM_BYTES * (GPU_COUNT + 1))
        candidate = node.evaluate("impossible-model", "r1", model_size)
        assert candidate.candidate_level == CandidateLevel.CANT_ACCOMMODATE

    def test_gpus_required_calculation(self, make_node_with_real_specs):
        """Verify gpus_required matches expected for real GPU memory."""
        node = make_node_with_real_specs()
        # Model exactly 1 GPU → gpus_required should be 2 (// + 1 formula)
        model_size = GPU_MEM_BYTES
        gpus_needed = node.resources.gpus_required(model_size)
        assert gpus_needed == 2, (
            f"Model exactly 1 GPU should need 2 GPUs (floor div + 1), got {gpus_needed}"
        )

        # Model half a GPU → gpus_required should be 1
        model_size_half = GPU_MEM_BYTES // 2
        gpus_needed_half = node.resources.gpus_required(model_size_half)
        assert gpus_needed_half == 1
