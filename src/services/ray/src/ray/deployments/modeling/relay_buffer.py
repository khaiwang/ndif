"""Fixed-size dual relay buffer for GPU↔CPU transfers.

Instead of allocating pinned CPU memory equal to the entire model size,
this module provides two small fixed-size pinned "relay" buffers that
alternate (double-buffering / ping-pong).  GPU DMA and CPU memcpy overlap
via CUDA streams, keeping throughput comparable while reducing pinned
memory from O(model_size) to a fixed ~512 MiB (configurable).

Usage::

    relay = get_relay_buffer()
    relay.ensure_allocated()

    # Eviction: GPU → relay → unpinned CPU
    cpu_tensors = relay.gpu_to_cpu(module)

    # Reload: unpinned CPU → relay → GPU
    gpu_tensors = relay.cpu_to_gpu(module, torch.device("cuda:0"))
"""

import logging
import os
import threading
from typing import Dict

import torch
import torch.nn as nn

from .pinned_pool import _alloc_pinned_chunk, _release_chunk_alloc, unique_named_tensors

logger = logging.getLogger("ndif")

_DEFAULT_RELAY_MB = int(os.environ.get("NDIF_RELAY_BUFFER_SIZE_MB", "256"))

# Module-level singleton
_relay_buffer_instance: "RelayBuffer | None" = None
_relay_lock = threading.Lock()


def get_relay_buffer() -> "RelayBuffer":
    """Return the per-process singleton :class:`RelayBuffer`."""
    global _relay_buffer_instance
    if _relay_buffer_instance is None:
        with _relay_lock:
            if _relay_buffer_instance is None:
                _relay_buffer_instance = RelayBuffer()
    return _relay_buffer_instance


class RelayBuffer:
    """Two fixed-size pinned buffers for pipelined GPU↔CPU transfers."""

    def __init__(self, relay_size_bytes: int | None = None) -> None:
        if relay_size_bytes is None:
            relay_size_bytes = _DEFAULT_RELAY_MB * 1024 * 1024
        self._relay_size = relay_size_bytes
        self._buf_a: torch.Tensor | None = None
        self._buf_b: torch.Tensor | None = None
        self._alloc_a = None
        self._alloc_b = None
        self._lock = threading.Lock()

    @property
    def total_pinned_bytes(self) -> int:
        return 2 * self._relay_size

    @property
    def is_allocated(self) -> bool:
        return self._buf_a is not None

    def ensure_allocated(self) -> None:
        """Lazily allocate both pinned relay buffers (thread-safe)."""
        if self._buf_a is not None:
            return
        with self._lock:
            if self._buf_a is not None:
                return
            self._buf_a, self._alloc_a = _alloc_pinned_chunk(self._relay_size)
            self._buf_b, self._alloc_b = _alloc_pinned_chunk(self._relay_size)
            logger.info(
                f"RelayBuffer allocated 2 × {self._relay_size / (1024**2):.0f} MiB "
                f"pinned buffers [{self._alloc_a.kind}, {self._alloc_b.kind}]"
            )

    def release(self) -> None:
        """Free both pinned relay buffers."""
        with self._lock:
            self._buf_a = None
            self._buf_b = None
            if self._alloc_a is not None:
                _release_chunk_alloc(self._alloc_a)
                self._alloc_a = None
            if self._alloc_b is not None:
                _release_chunk_alloc(self._alloc_b)
                self._alloc_b = None

    # ------------------------------------------------------------------
    # GPU → CPU  (eviction)
    # ------------------------------------------------------------------

    def gpu_to_cpu(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """Pipeline GPU tensors to unpinned CPU memory via relay buffers.

        Returns a dict mapping parameter/buffer names to new unpinned CPU
        tensors that hold copies of the GPU data.
        """
        assert self._buf_a is not None, "call ensure_allocated() first"
        relay_a = self._buf_a
        relay_b = self._buf_b
        relay_size = self._relay_size

        # Collect work items: list of (name, gpu_tensor_flat_u8, cpu_dest_flat_u8)
        work = []
        cpu_tensors: Dict[str, torch.Tensor] = {}

        for name, tensor in unique_named_tensors(module):
            gpu_flat = tensor.data.contiguous().view(torch.uint8).reshape(-1)
            cpu_dest = torch.empty(
                tensor.data.shape, dtype=tensor.data.dtype, device="cpu"
            )
            cpu_tensors[name] = cpu_dest
            cpu_flat = cpu_dest.view(torch.uint8).reshape(-1)
            work.append((name, gpu_flat, cpu_flat))

        # Build a flat list of (gpu_slice, cpu_dest_slice) chunks across all tensors
        slices = []
        for _name, gpu_flat, cpu_flat in work:
            total = gpu_flat.numel()
            offset = 0
            while offset < total:
                length = min(relay_size, total - offset)
                slices.append(
                    (
                        gpu_flat.narrow(0, offset, length),
                        cpu_flat.narrow(0, offset, length),
                    )
                )
                offset += length

        if not slices:
            return cpu_tensors

        device = slices[0][0].device
        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)

        # Pipeline: DMA on one stream while CPU memcpy for previous chunk
        # We process slices in pairs using ping-pong between relay_a and relay_b.
        pending_cpu_copy = None  # (relay_buf, cpu_dest_slice)

        for i, (gpu_slice, cpu_dest_slice) in enumerate(slices):
            cur_relay = relay_a if (i % 2 == 0) else relay_b
            cur_stream = stream_a if (i % 2 == 0) else stream_b

            relay_view = cur_relay.narrow(0, 0, gpu_slice.numel())

            # Start async DMA: GPU → pinned relay
            with torch.cuda.stream(cur_stream):
                relay_view.copy_(gpu_slice, non_blocking=True)

            # While DMA is in flight, do the CPU memcpy for the *previous* chunk
            if pending_cpu_copy is not None:
                prev_relay, prev_cpu = pending_cpu_copy
                prev_cpu.copy_(prev_relay)

            # Wait for current DMA to complete before we hand off this relay
            cur_stream.synchronize()
            pending_cpu_copy = (relay_view, cpu_dest_slice)

        # Final CPU memcpy for the last chunk
        if pending_cpu_copy is not None:
            prev_relay, prev_cpu = pending_cpu_copy
            prev_cpu.copy_(prev_relay)

        return cpu_tensors

    # ------------------------------------------------------------------
    # CPU → GPU  (reload)
    # ------------------------------------------------------------------

    def cpu_to_gpu(
        self, module: nn.Module, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Pipeline unpinned CPU tensors to GPU via relay buffers.

        Returns a dict mapping parameter/buffer names to new GPU tensors.
        """
        assert self._buf_a is not None, "call ensure_allocated() first"
        relay_a = self._buf_a
        relay_b = self._buf_b
        relay_size = self._relay_size

        # Collect work items
        work = []
        gpu_tensors: Dict[str, torch.Tensor] = {}

        for name, tensor in unique_named_tensors(module):
            cpu_flat = tensor.data.contiguous().view(torch.uint8).reshape(-1)
            gpu_dest = torch.empty(
                tensor.data.shape, dtype=tensor.data.dtype, device=device
            )
            gpu_tensors[name] = gpu_dest
            gpu_flat = gpu_dest.view(torch.uint8).reshape(-1)
            work.append((name, cpu_flat, gpu_flat))

        # Build slice list
        slices = []
        for _name, cpu_flat, gpu_flat in work:
            total = cpu_flat.numel()
            offset = 0
            while offset < total:
                length = min(relay_size, total - offset)
                slices.append(
                    (
                        cpu_flat.narrow(0, offset, length),
                        gpu_flat.narrow(0, offset, length),
                    )
                )
                offset += length

        if not slices:
            return gpu_tensors

        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)

        # Prime: load first chunk into relay_a
        cpu_slice_0, gpu_slice_0 = slices[0]
        relay_view_0 = relay_a.narrow(0, 0, cpu_slice_0.numel())
        relay_view_0.copy_(cpu_slice_0)  # CPU memcpy: unpinned → pinned

        # Start DMA for first chunk
        with torch.cuda.stream(stream_a):
            gpu_slice_0.copy_(relay_view_0, non_blocking=True)

        for i in range(1, len(slices)):
            cpu_slice, gpu_slice = slices[i]
            cur_relay = relay_b if (i % 2 == 1) else relay_a
            cur_stream = stream_b if (i % 2 == 1) else stream_a
            prev_stream = stream_a if (i % 2 == 1) else stream_b

            relay_view = cur_relay.narrow(0, 0, cpu_slice.numel())

            # CPU memcpy while previous DMA is in flight
            relay_view.copy_(cpu_slice)

            # Wait for previous DMA before reusing that stream
            prev_stream.synchronize()

            # Start DMA: pinned relay → GPU
            with torch.cuda.stream(cur_stream):
                gpu_slice.copy_(relay_view, non_blocking=True)

        # Wait for the last DMA
        stream_a.synchronize()
        stream_b.synchronize()

        return gpu_tensors
