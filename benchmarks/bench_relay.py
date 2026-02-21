#!/usr/bin/env python3
"""Microbenchmark for lazy-pin pipelined GPU↔CPU transfers.

Measures individual transfer legs and the full pipeline:
  D_g2c  — GPU→pinned DMA (CUDA events)
  D_c2g  — pinned→GPU DMA (CUDA events)
  M_faulting   — pinned→fresh-unpinned memcpy (page faults)
  M_prefaulted — pinned→pre-touched-unpinned memcpy
  Pipeline     — end-to-end gpu_to_cpu / cpu_to_gpu via lazy_pin

Usage:
    python benchmarks/bench_relay.py [--size-mb 512] [--repeats 5] [--warmup 2]
"""

import argparse
import time

import torch
import torch.nn as nn


def _bandwidth_str(nbytes: int, seconds: float) -> str:
    gb = nbytes / (1024**3)
    return f"{gb / seconds:6.2f} GB/s"


def bench_dma_g2c(buf_gpu, buf_pinned, repeats, warmup):
    device = buf_gpu.device
    stream = torch.cuda.Stream(device=device)
    times = []
    for i in range(warmup + repeats):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_ev.record(stream)
            buf_pinned.copy_(buf_gpu, non_blocking=True)
            end_ev.record(stream)
        stream.synchronize()
        if i >= warmup:
            times.append(start_ev.elapsed_time(end_ev) / 1000.0)
    return times


def bench_dma_c2g(buf_pinned, buf_gpu, repeats, warmup):
    device = buf_gpu.device
    stream = torch.cuda.Stream(device=device)
    times = []
    for i in range(warmup + repeats):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_ev.record(stream)
            buf_gpu.copy_(buf_pinned, non_blocking=True)
            end_ev.record(stream)
        stream.synchronize()
        if i >= warmup:
            times.append(start_ev.elapsed_time(end_ev) / 1000.0)
    return times


def bench_memcpy_faulting(buf_pinned, repeats, warmup):
    times = []
    for i in range(warmup + repeats):
        dst = torch.empty_like(buf_pinned, device="cpu")
        t0 = time.perf_counter()
        dst.copy_(buf_pinned)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
        del dst
    return times


def bench_memcpy_prefaulted(buf_pinned, repeats, warmup):
    times = []
    for i in range(warmup + repeats):
        dst = torch.empty_like(buf_pinned, device="cpu")
        dst[::4096] = 0
        t0 = time.perf_counter()
        dst.copy_(buf_pinned)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
        del dst
    return times


def bench_pipeline(size_mb, repeats, warmup):
    """End-to-end lazy-pin gpu_to_cpu and cpu_to_gpu pipeline."""
    from src.services.ray.src.ray.deployments.modeling.lazy_pin import (
        gpu_to_cpu, cpu_to_gpu,
    )

    dim = int((size_mb * 1024 * 1024) / 4)
    side = max(1, int(dim**0.5))
    model = nn.Linear(side, side, bias=False).cuda().float()

    model_bytes = sum(
        p.data.numel() * p.data.element_size() for p in model.parameters()
    )

    g2c_times = []
    c2g_times = []

    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu_views = gpu_to_cpu(model)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        for name, param in model.named_parameters():
            if name in cpu_views:
                param.data = cpu_views[name]

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        gpu_views = cpu_to_gpu(model, torch.device("cuda:0"))
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        for name, param in model.named_parameters():
            if name in gpu_views:
                param.data = gpu_views[name]

        if i >= warmup:
            g2c_times.append(t1 - t0)
            c2g_times.append(t3 - t2)

    return {"gpu_to_cpu": g2c_times, "cpu_to_gpu": c2g_times, "model_bytes": model_bytes}


def _stats(times):
    return sum(times) / len(times), min(times), max(times)


def main():
    parser = argparse.ArgumentParser(description="Lazy-pin transfer benchmark")
    parser.add_argument("--size-mb", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-pipeline", action="store_true")
    args = parser.parse_args()

    nbytes = args.size_mb * 1024 * 1024
    numel = nbytes

    print(f"Lazy-pin benchmark: {args.size_mb} MiB, "
          f"{args.repeats} repeats, {args.warmup} warmup\n")

    buf_gpu = torch.empty(numel, dtype=torch.uint8, device="cuda:0")
    buf_gpu.random_()
    buf_pinned = torch.empty(numel, dtype=torch.uint8, device="cpu").pin_memory()

    results = {}

    times = bench_dma_g2c(buf_gpu, buf_pinned, args.repeats, args.warmup)
    results["D_g2c"] = times

    times = bench_dma_c2g(buf_pinned, buf_gpu, args.repeats, args.warmup)
    results["D_c2g"] = times

    times = bench_memcpy_faulting(buf_pinned, args.repeats, args.warmup)
    results["M_faulting"] = times

    times = bench_memcpy_prefaulted(buf_pinned, args.repeats, args.warmup)
    results["M_prefaulted"] = times

    print(f"{'Benchmark':<22} {'Mean (ms)':>10} {'Min (ms)':>10} "
          f"{'Max (ms)':>10} {'Bandwidth':>12}")
    print("-" * 68)
    for name, times in results.items():
        mean, mn, mx = _stats(times)
        bw = _bandwidth_str(nbytes, mean)
        print(f"{name:<22} {mean*1000:10.1f} {mn*1000:10.1f} "
              f"{mx*1000:10.1f} {bw:>12}")

    if not args.skip_pipeline:
        print(f"\n{'Pipeline (end-to-end)':<22}")
        print("-" * 68)
        pipe = bench_pipeline(args.size_mb, args.repeats, args.warmup)
        model_bytes = pipe["model_bytes"]
        for name in ("gpu_to_cpu", "cpu_to_gpu"):
            times = pipe[name]
            mean, mn, mx = _stats(times)
            bw = _bandwidth_str(model_bytes, mean)
            print(f"{name:<22} {mean*1000:10.1f} {mn*1000:10.1f} "
                  f"{mx*1000:10.1f} {bw:>12}")

    del buf_gpu, buf_pinned
    print("\nDone.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
