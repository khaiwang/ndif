"""Tests for the dual relay buffer GPU↔CPU transfer pipeline."""

import pytest
import torch
import torch.nn as nn

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton between tests."""
    import src.services.ray.src.ray.deployments.modeling.relay_buffer as rb

    old = rb._relay_buffer_instance
    rb._relay_buffer_instance = None
    yield
    # Restore (or clean up)
    if rb._relay_buffer_instance is not None:
        rb._relay_buffer_instance.release()
    rb._relay_buffer_instance = old


def _make_relay():
    from src.services.ray.src.ray.deployments.modeling.relay_buffer import RelayBuffer

    return RelayBuffer(relay_size_bytes=1024)  # 1 KiB for fast tests


class TestRelayBufferLifecycle:
    def test_allocate_and_release(self):
        relay = _make_relay()
        assert not relay.is_allocated
        relay.ensure_allocated()
        assert relay.is_allocated
        assert relay.total_pinned_bytes == 2048
        relay.release()
        assert not relay.is_allocated

    def test_ensure_allocated_idempotent(self):
        relay = _make_relay()
        relay.ensure_allocated()
        relay.ensure_allocated()  # should not raise
        assert relay.is_allocated
        relay.release()

    def test_singleton(self):
        from src.services.ray.src.ray.deployments.modeling.relay_buffer import (
            get_relay_buffer,
        )

        a = get_relay_buffer()
        b = get_relay_buffer()
        assert a is b
        a.release()


class TestGPUToCPU:
    def test_small_linear(self):
        relay = _make_relay()
        relay.ensure_allocated()

        model = nn.Linear(8, 4, bias=True).cuda().float()
        weight_orig = model.weight.data.clone()
        bias_orig = model.bias.data.clone()

        cpu_views = relay.gpu_to_cpu(model)

        assert "weight" in cpu_views
        assert "bias" in cpu_views
        assert cpu_views["weight"].device.type == "cpu"
        assert cpu_views["bias"].device.type == "cpu"
        assert torch.equal(cpu_views["weight"], weight_orig.cpu())
        assert torch.equal(cpu_views["bias"], bias_orig.cpu())

        # Verify they are NOT pinned (unpinned CPU memory)
        assert not cpu_views["weight"].is_pinned()
        assert not cpu_views["bias"].is_pinned()

        relay.release()

    def test_tensor_larger_than_relay(self):
        """Test with a tensor whose byte size exceeds the relay buffer size (1 KiB)."""
        relay = _make_relay()  # 1 KiB relay
        relay.ensure_allocated()

        # 512 float32 values = 2048 bytes > 1024 byte relay
        model = nn.Linear(512, 1, bias=False).cuda().float()
        weight_orig = model.weight.data.clone()

        cpu_views = relay.gpu_to_cpu(model)
        assert torch.equal(cpu_views["weight"], weight_orig.cpu())

        relay.release()


class TestCPUToGPU:
    def test_small_linear(self):
        relay = _make_relay()
        relay.ensure_allocated()

        model = nn.Linear(8, 4, bias=True).float()
        # Start on CPU
        weight_orig = model.weight.data.clone()
        bias_orig = model.bias.data.clone()

        device = torch.device("cuda:0")
        gpu_views = relay.cpu_to_gpu(model, device)

        assert "weight" in gpu_views
        assert "bias" in gpu_views
        assert gpu_views["weight"].device.type == "cuda"
        assert gpu_views["bias"].device.type == "cuda"
        assert torch.equal(gpu_views["weight"].cpu(), weight_orig)
        assert torch.equal(gpu_views["bias"].cpu(), bias_orig)

        relay.release()

    def test_tensor_larger_than_relay(self):
        """Test with a tensor whose byte size exceeds the relay buffer size."""
        relay = _make_relay()  # 1 KiB relay
        relay.ensure_allocated()

        model = nn.Linear(512, 1, bias=False).float()
        weight_orig = model.weight.data.clone()

        device = torch.device("cuda:0")
        gpu_views = relay.cpu_to_gpu(model, device)
        assert torch.equal(gpu_views["weight"].cpu(), weight_orig)

        relay.release()


class TestRoundTrip:
    def test_gpu_to_cpu_to_gpu(self):
        """Full eviction → reload cycle: GPU → CPU → GPU preserves values."""
        relay = _make_relay()
        relay.ensure_allocated()

        model = nn.Sequential(
            nn.Linear(16, 8, bias=True),
            nn.Linear(8, 4, bias=False),
        ).cuda().float()

        # Snapshot original GPU values
        originals = {
            name: p.data.clone() for name, p in model.named_parameters()
        }

        # Evict: GPU → CPU
        cpu_views = relay.gpu_to_cpu(model)

        # Verify all on CPU
        for name, t in cpu_views.items():
            assert t.device.type == "cpu"

        # Simulate reassignment (as base.py would do)
        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        # Reload: CPU → GPU
        device = torch.device("cuda:0")
        gpu_views = relay.cpu_to_gpu(model, device)

        # Verify roundtrip fidelity
        for name, orig in originals.items():
            assert torch.equal(gpu_views[name], orig), f"Mismatch for {name}"

        relay.release()

    def test_bfloat16(self):
        """Ensure bfloat16 tensors survive the roundtrip."""
        relay = _make_relay()
        relay.ensure_allocated()

        model = nn.Linear(8, 4).to(dtype=torch.bfloat16, device="cuda")
        orig = model.weight.data.clone()

        cpu_views = relay.gpu_to_cpu(model)
        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        gpu_views = relay.cpu_to_gpu(model, torch.device("cuda:0"))
        assert torch.equal(gpu_views["weight"], orig)

        relay.release()
