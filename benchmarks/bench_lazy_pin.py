#!/usr/bin/env python3
"""Microbenchmark for cudaHostRegister-based lazy pinning.

Measures the critical unknowns for the 3-stage pipelined lazy pinning approach:
  1. cudaHostRegister on cold (freshly allocated, untouched) pages
  2. cudaHostRegister on pre-touched (faulted-in) pages
  3. cudaHostUnregister
  4. DMA after registration (to confirm it's as fast as cudaHostAlloc'd memory)
  5. Per-chunk timing at 512 MiB granularity
  6. Full 3-stage pipeline simulation

Usage:
    python benchmarks/bench_lazy_pin.py [--size-mb 512] [--repeats 5]
"""

import argparse
import time
from collections import defaultdict

import torch
from cuda import cudart  # pip install cuda-python


def _check(result):
    """Check CUDA runtime call succeeded."""
    err = result[0] if isinstance(result, tuple) else result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)}")
    return result


def _bw(nbytes: int, seconds: float) -> str:
    return f"{nbytes / (1024**3) / seconds:.2f} GB/s"


def bench_register_cold(size_mb: int, repeats: int) -> list[float]:
    """cudaHostRegister on freshly-allocated (never touched) memory."""
    nbytes = size_mb * 1024 * 1024
    times = []
    for _ in range(repeats):
        buf = torch.empty(nbytes, dtype=torch.uint8)
        ptr = buf.data_ptr()
        t0 = time.perf_counter()
        _check(cudart.cudaHostRegister(ptr, nbytes, cudart.cudaHostRegisterDefault))
        t1 = time.perf_counter()
        _check(cudart.cudaHostUnregister(ptr))
        times.append(t1 - t0)
        del buf
    return times


def bench_register_pretouched(size_mb: int, repeats: int) -> list[float]:
    """cudaHostRegister on pre-touched (faulted-in) memory."""
    nbytes = size_mb * 1024 * 1024
    times = []
    for _ in range(repeats):
        buf = torch.empty(nbytes, dtype=torch.uint8)
        buf.fill_(0)  # fault in all pages
        ptr = buf.data_ptr()
        t0 = time.perf_counter()
        _check(cudart.cudaHostRegister(ptr, nbytes, cudart.cudaHostRegisterDefault))
        t1 = time.perf_counter()
        _check(cudart.cudaHostUnregister(ptr))
        times.append(t1 - t0)
        del buf
    return times


def bench_unregister(size_mb: int, repeats: int) -> list[float]:
    """cudaHostUnregister timing."""
    nbytes = size_mb * 1024 * 1024
    times = []
    for _ in range(repeats):
        buf = torch.empty(nbytes, dtype=torch.uint8)
        buf.fill_(0)
        ptr = buf.data_ptr()
        _check(cudart.cudaHostRegister(ptr, nbytes, cudart.cudaHostRegisterDefault))
        t0 = time.perf_counter()
        _check(cudart.cudaHostUnregister(ptr))
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del buf
    return times


def bench_pretouch(size_mb: int, repeats: int) -> list[float]:
    """Time just the pre-touch (fill_) step."""
    nbytes = size_mb * 1024 * 1024
    times = []
    for _ in range(repeats):
        buf = torch.empty(nbytes, dtype=torch.uint8)
        t0 = time.perf_counter()
        buf.fill_(0)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del buf
    return times


def bench_dma_after_register(size_mb: int, repeats: int) -> list[float]:
    """GPU→registered-pinned DMA to confirm speed matches cudaHostAlloc."""
    nbytes = size_mb * 1024 * 1024
    gpu_buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
    gpu_buf.random_()
    stream = torch.cuda.Stream()
    times = []
    for _ in range(repeats):
        cpu_buf = torch.empty(nbytes, dtype=torch.uint8)
        cpu_buf.fill_(0)
        ptr = cpu_buf.data_ptr()
        _check(cudart.cudaHostRegister(ptr, nbytes, cudart.cudaHostRegisterDefault))

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_ev.record(stream)
            cpu_buf.copy_(gpu_buf, non_blocking=True)
            end_ev.record(stream)
        stream.synchronize()
        times.append(start_ev.elapsed_time(end_ev) / 1000.0)

        _check(cudart.cudaHostUnregister(ptr))
        del cpu_buf
    del gpu_buf
    return times


def bench_dma_cudahostalloc(size_mb: int, repeats: int) -> list[float]:
    """GPU→cudaHostAlloc'd pinned DMA (baseline reference)."""
    nbytes = size_mb * 1024 * 1024
    gpu_buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
    gpu_buf.random_()
    cpu_buf = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    stream = torch.cuda.Stream()
    times = []
    for _ in range(repeats):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_ev.record(stream)
            cpu_buf.copy_(gpu_buf, non_blocking=True)
            end_ev.record(stream)
        stream.synchronize()
        times.append(start_ev.elapsed_time(end_ev) / 1000.0)
    del gpu_buf, cpu_buf
    return times


def bench_pipeline_simulation(total_mb: int, chunk_mb: int, repeats: int) -> list[dict]:
    """Simulate the 3-stage pipeline: touch → register → DMA.

    Returns per-repeat dicts with stage timings.
    """
    total_bytes = total_mb * 1024 * 1024
    chunk_bytes = chunk_mb * 1024 * 1024
    n_chunks = (total_bytes + chunk_bytes - 1) // chunk_bytes

    gpu_buf = torch.empty(total_bytes, dtype=torch.uint8, device="cuda:0")
    gpu_buf.random_()
    stream = torch.cuda.Stream()

    results = []
    for _ in range(repeats):
        # Allocate all destination chunks upfront (unpinned)
        chunks = []
        for c in range(n_chunks):
            offset = c * chunk_bytes
            length = min(chunk_bytes, total_bytes - offset)
            cpu_chunk = torch.empty(length, dtype=torch.uint8)
            gpu_chunk = gpu_buf.narrow(0, offset, length)
            chunks.append((cpu_chunk, gpu_chunk))

        stage_times = defaultdict(float)

        # Sequential 3-stage pipeline (single-threaded simulation)
        # In production, these would be on separate threads
        t_total_start = time.perf_counter()

        for i, (cpu_chunk, gpu_chunk) in enumerate(chunks):
            # Stage 1: pre-touch
            t0 = time.perf_counter()
            cpu_chunk.fill_(0)
            t1 = time.perf_counter()
            stage_times["touch"] += t1 - t0

            # Stage 2: register
            ptr = cpu_chunk.data_ptr()
            t2 = time.perf_counter()
            _check(cudart.cudaHostRegister(ptr, cpu_chunk.numel(),
                                           cudart.cudaHostRegisterDefault))
            t3 = time.perf_counter()
            stage_times["register"] += t3 - t2

            # Stage 3: DMA
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start_ev.record(stream)
                cpu_chunk.copy_(gpu_chunk, non_blocking=True)
                end_ev.record(stream)
            stream.synchronize()
            stage_times["dma"] += start_ev.elapsed_time(end_ev) / 1000.0

        t_total_end = time.perf_counter()
        stage_times["total_sequential"] = t_total_end - t_total_start

        # Pipelined estimate: max(touch, register, dma) * n_chunks + drain
        bottleneck = max(
            stage_times["touch"] / n_chunks,
            stage_times["register"] / n_chunks,
            stage_times["dma"] / n_chunks,
        )
        stage_times["estimated_pipelined"] = bottleneck * n_chunks + 2 * bottleneck
        stage_times["bottleneck_stage"] = max(
            ("touch", stage_times["touch"]),
            ("register", stage_times["register"]),
            ("dma", stage_times["dma"]),
            key=lambda x: x[1],
        )[0]
        stage_times["n_chunks"] = n_chunks

        # Cleanup: unregister all
        for cpu_chunk, _ in chunks:
            _check(cudart.cudaHostUnregister(cpu_chunk.data_ptr()))

        results.append(dict(stage_times))

    del gpu_buf
    return results


def _stats(times):
    return sum(times) / len(times), min(times), max(times)


def main():
    parser = argparse.ArgumentParser(description="Lazy pinning microbenchmark")
    parser.add_argument("--size-mb", type=int, default=512,
                        help="Per-operation buffer size in MiB (default: 512)")
    parser.add_argument("--total-mb", type=int, default=2048,
                        help="Total size for pipeline simulation in MiB (default: 2048)")
    parser.add_argument("--chunk-mb", type=int, default=512,
                        help="Chunk size for pipeline simulation in MiB (default: 512)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Measurement repeats (default: 5)")
    args = parser.parse_args()

    nbytes = args.size_mb * 1024 * 1024

    print(f"Lazy pinning microbenchmark: {args.size_mb} MiB per-op, "
          f"{args.repeats} repeats\n")

    header = f"{'Benchmark':<32} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'Bandwidth':>12}"
    print(header)
    print("-" * len(header))

    # Pre-touch
    times = bench_pretouch(args.size_mb, args.repeats)
    mean, mn, mx = _stats(times)
    print(f"{'pre-touch (fill_)':<32} {mean*1000:10.1f} {mn*1000:10.1f} {mx*1000:10.1f} {_bw(nbytes, mean):>12}")

    # Register cold
    times = bench_register_cold(args.size_mb, args.repeats)
    mean, mn, mx = _stats(times)
    print(f"{'cudaHostRegister (cold)':<32} {mean*1000:10.1f} {mn*1000:10.1f} {mx*1000:10.1f} {_bw(nbytes, mean):>12}")

    # Register pre-touched
    times = bench_register_pretouched(args.size_mb, args.repeats)
    mean, mn, mx = _stats(times)
    print(f"{'cudaHostRegister (pre-touched)':<32} {mean*1000:10.1f} {mn*1000:10.1f} {mx*1000:10.1f} {_bw(nbytes, mean):>12}")

    # Unregister
    times = bench_unregister(args.size_mb, args.repeats)
    mean, mn, mx = _stats(times)
    print(f"{'cudaHostUnregister':<32} {mean*1000:10.1f} {mn*1000:10.1f} {mx*1000:10.1f} {_bw(nbytes, mean):>12}")

    # DMA after register
    times = bench_dma_after_register(args.size_mb, args.repeats)
    mean, mn, mx = _stats(times)
    print(f"{'DMA GPU→registered pinned':<32} {mean*1000:10.1f} {mn*1000:10.1f} {mx*1000:10.1f} {_bw(nbytes, mean):>12}")

    # DMA cudaHostAlloc (baseline)
    times = bench_dma_cudahostalloc(args.size_mb, args.repeats)
    mean, mn, mx = _stats(times)
    print(f"{'DMA GPU→cudaHostAlloc (ref)':<32} {mean*1000:10.1f} {mn*1000:10.1f} {mx*1000:10.1f} {_bw(nbytes, mean):>12}")

    # Pipeline simulation
    print(f"\n{'Pipeline simulation':<32} {args.total_mb} MiB total, "
          f"{args.chunk_mb} MiB chunks\n")

    results = bench_pipeline_simulation(args.total_mb, args.chunk_mb, args.repeats)

    print(f"{'Stage':<32} {'Total (ms)':>10} {'Per-chunk (ms)':>14}")
    print("-" * 60)

    for key in ("touch", "register", "dma"):
        vals = [r[key] for r in results]
        mean = sum(vals) / len(vals)
        n = results[0]["n_chunks"]
        print(f"{key:<32} {mean*1000:10.1f} {mean/n*1000:14.1f}")

    seq_vals = [r["total_sequential"] for r in results]
    pipe_vals = [r["estimated_pipelined"] for r in results]
    seq_mean = sum(seq_vals) / len(seq_vals)
    pipe_mean = sum(pipe_vals) / len(pipe_vals)
    bottleneck = results[0]["bottleneck_stage"]

    print(f"\n{'Sequential total':<32} {seq_mean*1000:10.1f} ms")
    print(f"{'Estimated pipelined':<32} {pipe_mean*1000:10.1f} ms")
    print(f"{'Bottleneck stage':<32} {bottleneck}")
    print(f"{'Speedup vs sequential':<32} {seq_mean/pipe_mean:.2f}x")

    # Compare with relay buffer benchmark numbers
    total_bytes = args.total_mb * 1024 * 1024
    print(f"\n{'Comparison at ' + str(args.total_mb) + ' MiB:'}")
    print(f"  Pipelined lazy pin (est.):  {pipe_mean*1000:.0f} ms  ({_bw(total_bytes, pipe_mean)})")
    print(f"  Sequential lazy pin:        {seq_mean*1000:.0f} ms  ({_bw(total_bytes, seq_mean)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
