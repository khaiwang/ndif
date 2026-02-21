"""Tests for the pipelined lazy-pin GPU↔CPU transfer."""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

from src.services.ray.src.ray.deployments.modeling.lazy_pin import gpu_to_cpu, cpu_to_gpu


class TestGPUToCPU:
    def test_small_linear(self):
        model = nn.Linear(8, 4, bias=True).cuda().float()
        weight_orig = model.weight.data.clone()
        bias_orig = model.bias.data.clone()

        cpu_views = gpu_to_cpu(model, chunk_size=1024)

        assert "weight" in cpu_views
        assert "bias" in cpu_views
        assert cpu_views["weight"].device.type == "cpu"
        assert cpu_views["bias"].device.type == "cpu"
        assert torch.equal(cpu_views["weight"], weight_orig.cpu())
        assert torch.equal(cpu_views["bias"], bias_orig.cpu())
        assert not cpu_views["weight"].is_pinned()
        assert not cpu_views["bias"].is_pinned()

    def test_multi_chunk(self):
        """Tensor larger than chunk_size forces multiple pipeline iterations."""
        model = nn.Linear(512, 1, bias=False).cuda().float()
        weight_orig = model.weight.data.clone()

        # 512 float32 = 2048 bytes, chunk = 1024 → 2 chunks
        cpu_views = gpu_to_cpu(model, chunk_size=1024)
        assert torch.equal(cpu_views["weight"], weight_orig.cpu())

    def test_multi_param(self):
        model = nn.Sequential(
            nn.Linear(32, 16, bias=True),
            nn.Linear(16, 8, bias=True),
        ).cuda().float()

        originals = {n: p.data.clone() for n, p in model.named_parameters()}
        cpu_views = gpu_to_cpu(model, chunk_size=1024)

        for name, orig in originals.items():
            assert torch.equal(cpu_views[name], orig.cpu()), f"Mismatch: {name}"

    def test_empty_model(self):
        model = nn.Module().cuda()
        cpu_views = gpu_to_cpu(model)
        assert cpu_views == {}


class TestCPUToGPU:
    def test_small_linear(self):
        model = nn.Linear(8, 4, bias=True).float()
        weight_orig = model.weight.data.clone()
        bias_orig = model.bias.data.clone()

        device = torch.device("cuda:0")
        gpu_views = cpu_to_gpu(model, device, chunk_size=1024)

        assert gpu_views["weight"].device.type == "cuda"
        assert gpu_views["bias"].device.type == "cuda"
        assert torch.equal(gpu_views["weight"].cpu(), weight_orig)
        assert torch.equal(gpu_views["bias"].cpu(), bias_orig)

    def test_multi_chunk(self):
        model = nn.Linear(512, 1, bias=False).float()
        weight_orig = model.weight.data.clone()

        gpu_views = cpu_to_gpu(model, torch.device("cuda:0"), chunk_size=1024)
        assert torch.equal(gpu_views["weight"].cpu(), weight_orig)


class TestRoundTrip:
    def test_gpu_cpu_gpu(self):
        """Full eviction → reload preserves values."""
        model = nn.Sequential(
            nn.Linear(16, 8, bias=True),
            nn.Linear(8, 4, bias=False),
        ).cuda().float()

        originals = {n: p.data.clone() for n, p in model.named_parameters()}

        cpu_views = gpu_to_cpu(model, chunk_size=1024)

        # Simulate base.py reassignment
        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        gpu_views = cpu_to_gpu(model, torch.device("cuda:0"), chunk_size=1024)

        for name, orig in originals.items():
            assert torch.equal(gpu_views[name], orig), f"Mismatch: {name}"

    def test_bfloat16(self):
        model = nn.Linear(8, 4).to(dtype=torch.bfloat16, device="cuda")
        orig = model.weight.data.clone()

        cpu_views = gpu_to_cpu(model)
        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        gpu_views = cpu_to_gpu(model, torch.device("cuda:0"))
        assert torch.equal(gpu_views["weight"], orig)

    def test_many_chunks(self):
        """Stress the pipeline with many small chunks."""
        model = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 64, bias=True),
            nn.Linear(64, 32, bias=False),
        ).cuda().float()

        originals = {n: p.data.clone() for n, p in model.named_parameters()}

        cpu_views = gpu_to_cpu(model, chunk_size=4096)
        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        gpu_views = cpu_to_gpu(model, torch.device("cuda:0"), chunk_size=4096)
        for name, orig in originals.items():
            assert torch.equal(gpu_views[name], orig), f"Mismatch: {name}"
