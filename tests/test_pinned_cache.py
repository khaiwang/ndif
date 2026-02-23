"""Tests for the adaptive pinned warm cache (mirror + relay fallback)."""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

from src.services.ray.src.ray.deployments.modeling.pinned_pool import (  # noqa: E402
    PinnedBufferPool,
    unique_named_tensors,
)
from src.services.ray.src.ray.deployments.modeling.relay_buffer import (  # noqa: E402
    RelayBuffer,
)


# ---------------------------------------------------------------------------
# PinnedBufferPool tests
# ---------------------------------------------------------------------------


class TestPinnedBufferPool:
    def test_preallocate_and_transfer(self):
        model = nn.Linear(16, 8, bias=True).float()
        weight_orig = model.weight.data.clone()
        bias_orig = model.bias.data.clone()

        pool = PinnedBufferPool()
        pool.preallocate(model)
        assert pool.is_ready
        assert pool.matches(model)

        # Copy CPU data into pinned views
        for name, tensor in unique_named_tensors(model):
            pinned = pool.get(name)
            assert pinned is not None
            pinned.copy_(tensor.data)

        # Transfer to GPU
        device = torch.device("cuda:0")
        gpu_views = pool.transfer_to_device(device)

        assert torch.equal(gpu_views["weight"].cpu(), weight_orig)
        assert torch.equal(gpu_views["bias"].cpu(), bias_orig)

        pool.release()

    def test_matches_rejects_different_model(self):
        model_a = nn.Linear(16, 8).float()
        model_b = nn.Linear(32, 16).float()

        pool = PinnedBufferPool()
        pool.preallocate(model_a)
        assert pool.is_ready
        assert pool.matches(model_a)
        assert not pool.matches(model_b)

        pool.release()

    def test_reuse_across_cycles(self):
        """Pool can be transferred to GPU multiple times (evict/reload cycle)."""
        model = nn.Linear(8, 4, bias=True).float()

        pool = PinnedBufferPool()
        pool.preallocate(model)
        assert pool.is_ready

        for name, tensor in unique_named_tensors(model):
            pool.get(name).copy_(tensor.data)

        device = torch.device("cuda:0")

        # First transfer
        gpu_views_1 = pool.transfer_to_device(device)
        assert gpu_views_1["weight"].device.type == "cuda"

        # Second transfer (simulating another reload)
        gpu_views_2 = pool.transfer_to_device(device)
        assert torch.equal(gpu_views_2["weight"].cpu(), gpu_views_1["weight"].cpu())

        pool.release()

    def test_async_preallocate(self):
        model = nn.Linear(8, 4).float()

        pool = PinnedBufferPool()
        pool.preallocate_async(model)
        assert pool.wait_ready(timeout=10.0)
        assert pool.matches(model)

        pool.release()


# ---------------------------------------------------------------------------
# RelayBuffer tests
# ---------------------------------------------------------------------------


class TestRelayBuffer:
    def test_gpu_to_cpu(self):
        model = nn.Linear(16, 8, bias=True).cuda().float()
        weight_orig = model.weight.data.clone()
        bias_orig = model.bias.data.clone()

        relay = RelayBuffer(relay_size_bytes=1024)
        relay.ensure_allocated()

        cpu_views = relay.gpu_to_cpu(model)

        assert cpu_views["weight"].device.type == "cpu"
        assert torch.equal(cpu_views["weight"], weight_orig.cpu())
        assert torch.equal(cpu_views["bias"], bias_orig.cpu())

        relay.release()

    def test_cpu_to_gpu(self):
        model = nn.Linear(16, 8, bias=True).float()
        weight_orig = model.weight.data.clone()

        relay = RelayBuffer(relay_size_bytes=1024)
        relay.ensure_allocated()

        device = torch.device("cuda:0")
        gpu_views = relay.cpu_to_gpu(model, device)

        assert gpu_views["weight"].device.type == "cuda"
        assert torch.equal(gpu_views["weight"].cpu(), weight_orig)

        relay.release()

    def test_round_trip(self):
        """GPU → relay → CPU → relay → GPU preserves values."""
        model = (
            nn.Sequential(
                nn.Linear(16, 8, bias=True),
                nn.Linear(8, 4, bias=False),
            )
            .cuda()
            .float()
        )

        originals = {n: p.data.clone() for n, p in model.named_parameters()}

        relay = RelayBuffer(relay_size_bytes=1024)
        relay.ensure_allocated()

        # Evict: GPU → CPU
        cpu_views = relay.gpu_to_cpu(model)
        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        # Reload: CPU → GPU
        gpu_views = relay.cpu_to_gpu(model, torch.device("cuda:0"))
        for name, orig in originals.items():
            assert torch.equal(gpu_views[name], orig), f"Mismatch: {name}"

        relay.release()


# ---------------------------------------------------------------------------
# Mirror lifecycle tests
# ---------------------------------------------------------------------------


class TestMirrorLifecycle:
    def test_mirror_dispatch_and_zero_copy_eviction(self):
        """Simulate dispatch_to_gpu mirror path and zero-copy eviction."""
        model = nn.Linear(16, 8, bias=True).float()
        weight_orig = model.weight.data.clone()

        # Step 1: Preallocate pool (simulates background alloc during __init__)
        pool = PinnedBufferPool()
        pool.preallocate(model)
        assert pool.is_ready

        # Step 2: Copy mmap'd CPU → pinned views (simulates dispatch_to_gpu)
        for name, tensor in unique_named_tensors(model):
            pool.get(name).copy_(tensor.data)

        # Step 3: Transfer to GPU
        device = torch.device("cuda:0")
        gpu_views = pool.transfer_to_device(device)
        torch.cuda.synchronize()

        # Verify GPU data
        assert torch.equal(gpu_views["weight"].cpu(), weight_orig)

        # Step 4: Zero-copy eviction — just swap param.data to pool views
        # (simulates to_cache with mirror)
        for name, param in model.named_parameters():
            pinned = pool.get(name)
            if pinned is not None:
                param.data = pinned

        # Verify data is on CPU (pinned)
        assert model.weight.data.device.type == "cpu"
        assert torch.equal(model.weight.data, weight_orig)

        # Step 5: Reload from pinned pool
        gpu_views_2 = pool.transfer_to_device(device)
        torch.cuda.synchronize()

        assert torch.equal(gpu_views_2["weight"].cpu(), weight_orig)

        pool.release()

    def test_fallback_relay_then_async_pin(self):
        """Simulate eviction via relay (no mirror), then async pin upgrade."""
        model = nn.Linear(16, 8, bias=True).cuda().float()
        weight_orig = model.weight.data.clone()

        # Evict via relay buffer (simulates to_cache without mirror)
        relay = RelayBuffer(relay_size_bytes=1024)
        relay.ensure_allocated()
        cpu_views = relay.gpu_to_cpu(model)

        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        # Simulate async pin upgrade
        pool = PinnedBufferPool()
        pool.preallocate(model)
        assert pool.is_ready

        for name, tensor in unique_named_tensors(model):
            pinned = pool.get(name)
            if pinned is not None:
                pinned.copy_(tensor.data)

        # Swap to pinned
        for name, param in model.named_parameters():
            pinned = pool.get(name)
            if pinned is not None:
                param.data = pinned

        # Now reload from pinned
        device = torch.device("cuda:0")
        gpu_views = pool.transfer_to_device(device)
        torch.cuda.synchronize()

        assert torch.equal(gpu_views["weight"].cpu(), weight_orig.cpu())

        pool.release()
        relay.release()
